# 核心机制

## 1. AsyncStream  trait 体系

`shoes` 的核心设计是将所有传输层统一为少数几个 trait，使得代理链的各层可以像洋葱一样包裹，而双向拷贝逻辑完全与协议解耦。

```
AsyncStream = AsyncRead + AsyncWrite + AsyncPing + Unpin + Send + Sync
     │
     ├─ tokio::net::TcpStream
     ├─ CryptoTlsStream          (TLS / REALITY)
     ├─ QuicStream               (QUIC bidirectional stream)
     ├─ TcpConnection            (TUN smoltcp 桥接)
     ├─ ShadowsocksStream
     ├─ VmessStream
     ├─ WebsocketStream
     └─ ... 每个协议都有自己的 Stream 包装器

AsyncMessageStream = 消息/数据报语义
     │
     ├─ tokio::net::UdpSocket
     ├─ XudpMessageStream
     ├─ UotV1ServerStream
     └─ ...

AsyncTargetedMessageStream    读 = (payload, target_addr)
AsyncSessionMessageStream     读 = (payload, session_id)
```

**Blanket impl**：为 `Box<T>` 和 `&mut T` 实现所有 trait，因此可以很方便地在堆上擦除类型或做可变引用传递。

---

## 2. TCP Server Handler 与 Client Handler 的分工

- **Server Handler**（入站）：负责协议握手，解析出目标地址，返回 `TcpServerSetupResult`
- **Client Handler**（出站）：负责在已建立的传输流上发送代理协议握手，将流转换为可直接发送应用数据的形式，返回 `TcpClientSetupResult`

两者通过 `proxy_connector.rs` / `socket_connector.rs` 组合成 `ClientProxyChain`。

---

## 3. 代理链构造与执行

代理链由 `chain_builder.rs` 从配置构建为运行时结构：

```
ClientChainGroup
    └─ [ ClientProxyChain, ClientProxyChain, ... ]   (round-robin)

ClientProxyChain
    ├─ initial_hop: Vec<InitialHopEntry>
    │     ├─ Direct(SocketConnectorImpl)
    │     └─ Proxy { socket, proxy_handler }
    │
    └─ subsequent_hops: Vec<Vec<Box<dyn ProxyConnector>>>
```

**连接执行逻辑**（`client_proxy_chain.rs:connect_tcp`）：
1. 轮询选择 `initial_hop` 中的一个入口
2. 轮询选择 `subsequent_hops` 每一层的一个 handler
3. 确定第一跳目标：
   - 如果有后续 hop → 第一跳 proxy 的地址
   - 否则 → 最终目标地址
4. `socket.connect()` 建立底层连接
5. 若 initial hop 是 Proxy，则在其上执行 `setup_client_tcp_stream()`
6. 逐层执行后续 hop 的 `setup_client_tcp_stream(stream, next_target)`
7. 返回最终 `client_stream`

**UDP 代理链**（`connect_udp_bidirectional`）：
- 检查链的最终 hop 是否支持 UDP（`udp_final_hop_indices`）
- 若后续 hop 为空：直接选择支持 UDP 的 initial hop
- 若后续 hop 存在：先建立 TCP 层到倒数第二跳，最后跳使用支持 UDP 的 handler

---

## 4. 规则路由与缓存

`ClientProxySelector` 维护有序的 `Vec<ConnectRule>`，每条规则包含：
- `masks: Vec<NetLocationMask>`：CIDR、主机名（子域匹配）、端口
- `action: ConnectAction`：Allow（指定 chain_group + 可选地址覆盖）或 Block

**决策流程**（`judge()`）：
1. 若启用缓存，先查 `RoutingCache`
2. 缓存未命中：线性扫描规则，调用 `match_mask()`
3. 主机名规则可能需要先解析 DNS（`resolve_rule_hostnames = true`）
4. 匹配到第一条生效规则即停止
5. 无匹配 → Block
6. 结果写入缓存

**缓存策略**：
- 当规则数 > 16 或 `resolve_rule_hostnames = true` 时自动启用
- 键：`ResolvedLocation`（地址 + 端口）
- 值：`Allow(rule_index)` 或 `Block`
- LRU 容量固定，不随配置热重载自动清空（设计权衡）

---

## 5. DNS 系统的分层设计

```
应用层查询
    │
    ├─ Resolver trait
    │     ├─ NativeResolver          → tokio::net::lookup_host
    │     ├─ CachingNativeResolver   → NativeResolver + 1h TTL cache
    │     ├─ TimeoutResolver         → 包装任意 Resolver，加超时
    │     ├─ RefreshingResolver      → 检测到空闲/错误时重建内部 Resolver
    │     ├─ CompositeResolver       → 顺序尝试多个 resolver
    │     └─ HickoryResolver         → hickory-dns，支持 DoT/DoH/DoH3
    │
    └─ DnsRegistry
          ├─ 每个 named DNS group 对应一个 Arc<dyn Resolver>
          └─ 默认 fallback = CachingNativeResolver
```

**关键设计**：`ProxyRuntimeProvider` 让 hickory 的 TCP 连接通过 `ClientChainGroup.connect_tcp()` 发出，因此 **DoT/DoH/DoH3 上游可以完全走代理链**。

**限制**：UDP DNS 和 H3 DNS 只能直连（受限于 `bind_interface`），不能走代理链。

---

## 6. TUN 模式的三线程协作模型

TUN 模式下有三个主要执行单元协作：

1. **smoltcp OS 线程**（`tcp_stack_direct.rs`）：
   - 阻塞 `select()` 在 TUN fd 上
   - batch 读取 IP 包（最多 64 个）
   - 对 TCP：维护 smoltcp socket 状态，与 `TcpConnection` 通过 ring buffer + waker 交换数据
   - 对 UDP：直接将包通过 channel 丢给 tokio UDP handler，**不经过 smoltcp**

2. **Tokio TCP handler 任务**（`tun/mod.rs`）：
   - 从 channel 接收 `NewTcpConnection`
   - 调用 `proxy_selector.judge()` → `chain_group.connect_tcp()`
   - `tokio::io::copy_bidirectional()` 桥接 `TcpConnection` 与代理链 outbound stream

3. **Tokio UDP manager 任务**（`tun/udp_manager.rs`）：
   - `TunUdpManager` 用 LRU 缓存按**本地地址**分 session
   - 每个 session 内再按目标地址分 destination task
   - destination task 持有 `AsyncMessageStream`，`select!` 读写 + 120s 超时

**线程安全要点**：
- `TcpConnectionControl` 用 `AtomicBool`、`Mutex<RingBuffer>`、`Option<Waker>` 实现跨线程同步
- smoltcp 线程在数据就绪时通过 `unpark()` 唤醒 async task；async task 写数据或关闭时通过 `unpark()` 唤醒 smoltcp 线程
- fd 生命周期由 `TcpStackDirect` 统一管理，关闭时设置 `running = false` 让线程退出

---

## 7. QUIC 服务器协议分发

`start_quic_servers()` 根据 `ServerProxyConfig` 做一级分发：

- `Hysteria2` → `hysteria2_server::start_hysteria2_server()`
- `TuicV5` → `tuic_server::start_tuic_server()`
- 其他 TCP 协议（VLESS/VMess/Shadowsocks/...）→ 通用 QUIC 适配层

通用 QUIC 适配层的工作方式：
1. Quinn endpoint accept 到 `Incoming` connection
2. `process_connection()` 等待连接建立，然后循环 `accept_bi()`
3. 每个 bidirectional stream 被包装为 `QuicStream`（实现 `AsyncStream`）
4. 调用对应协议的 `TcpServerHandler::setup_server_stream()`
5. 后续处理与 TCP 完全一致：`TcpForward` → `setup_client_tcp_stream()` → `copy_bidirectional()`

这意味着 **任何 TCP 协议 handler 无需修改即可运行在 QUIC 传输之上**，得益于 `AsyncStream` 抽象。

---

## 8. 配置热重载机制

1. 启动时：`notify::RecommendedWatcher` 监听每个配置文件路径的 `EventKind::Modify`
2. 触发后：发送 `ConfigChanged` 信号
3. `main()` 中的循环收到信号后：
   - `abort()` 所有 server join handle
   - sleep 3s 等待资源释放
   - drain 可能积压的多次修改事件
   - 重新执行 `load_configs()` → `create_server_configs()` → 启动服务器

**注意**：热重载会重建整个运行时（resolver、selector、server handler），但 TUN fd 如果是外部传入（移动端）则不会被关闭。

---

## 9. 移动端 Socket Protect

防止 VPN 路由循环的关键机制：

1. Android：Java 层实现 `protect(int fd)`，通过 JNI 传入 Rust
2. iOS：Swift/ObjC 层实现 C 回调 `protect_socket(fd)`
3. Rust 层：全局 `GLOBAL_SOCKET_PROTECTOR`（`RwLock<Option<Arc<dyn SocketProtector>>>`）
4. `SocketConnectorImpl::connect()` 和 `ProxyRuntimeProvider::bind_udp()` 等位置在创建 socket 后调用 `protect_socket(fd)`

所有需要保护的 socket 都通过 `socket_util.rs` 中的辅助函数创建，确保 `protect` 被统一调用。
