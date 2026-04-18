# 功能模块全景

本文档按源码目录梳理每个模块的职责、对外暴露的关键类型与入口函数。

---

## 一、入口与生命周期

### `src/main.rs`
- **职责**：CLI 二进制入口
- **关键流程**：参数解析 -> 配置加载 -> 校验 -> 启动 TCP/QUIC/TUN 服务器 -> 热重载监听
- **关键函数**：
  - `main()`：线程数设置、日志初始化、配置加载循环
  - `start_notify_thread()`：`notify` 库监听配置文件 `Modify` 事件

### `src/lib.rs`
- **职责**：Library 入口，为移动 FFI 提供可链接的库
- **特点**：使用 `#![allow(dead_code)]`，因为服务端代码在纯 lib 构建中看似未被使用

---

## 二、配置系统（`src/config/`）

### `src/config/mod.rs`
- `load_configs()`：读取多个 YAML 文件，解析为 `Vec<Config>`
- `convert_cert_paths()`：将 PEM 证书路径扩展为内联数据

### `src/config/types/`
所有 Serde 反序列化类型：
- `client.rs`：`ClientConfig`、`ClientProxyConfig`（所有出站协议枚举）
- `server.rs`：`ServerConfig`、`BindLocation`
- `rules.rs`：`RuleConfig`、匹配条件与动作
- `dns.rs`：`DnsConfig`、`DnsConfigGroup`、`DnsServerSpec`
- `groups.rs`：`ClientChain`、`ClientChainHop`、代理组引用
- `transport.rs`：`Transport` 枚举（Tcp / Quic / Udp）
- `tun.rs`：`TunConfig`、`TunServerConfig`
- `common.rs`：共享辅助类型

### `src/config/validate.rs`
- `create_server_configs()`：核心校验函数
- 拓扑排序解析 client 组（Kahn 算法，防循环）
- 提取内联 DNS 并生成匿名组
- 展开 DNS 组的组合引用
- 按 bootstrap 依赖拓扑排序 DNS 组
- 校验 URL、client chain、协议兼容性

### `src/config/pem.rs`
- PEM 文件读取与内联嵌入

---

## 三、TCP 核心（`src/tcp/`）

### `src/tcp/tcp_handler.rs`
- **核心 Trait**：
  - `TcpServerHandler`：`setup_server_stream()` → `TcpServerSetupResult`
  - `TcpClientHandler`：`setup_client_tcp_stream()` → `TcpClientSetupResult`
- **`TcpServerSetupResult`** 变体：
  - `TcpForward`：标准 TCP 转发
  - `BidirectionalUdp`：UDP-over-TCP 隧道
  - `MultiDirectionalUdp`：多目标 UDP 路由
  - `SessionBasedUdp`：基于会话 ID 的 UDP（XUDP）
  - `AlreadyHandled`：handler 已自行处理（如 REALITY fallback）

### `src/tcp/tcp_server.rs`
- `run_tcp_server()`：TCP 监听 accept 循环
- `process_stream()`：单连接生命周期管理（setup → 路由 → 建链 → 双向拷贝）
- `start_servers()` / `start_tcp_servers()`：根据 `ServerConfig` 批量启动

### `src/tcp/tcp_server_handler_factory.rs`
- `create_tcp_server_handler()`：根据 `ServerProxyConfig` 构造入站协议 handler

### `src/tcp/tcp_client_handler_factory.rs`
- `create_tcp_client_handler()`：根据 `ClientProxyConfig` 构造出站协议 handler
- `create_tcp_client_proxy_selector()`：将规则配置转换为 `ClientProxySelector`

### `src/tcp/chain_builder.rs`
- `build_client_proxy_chain()`：将配置中的多跳链转换为运行时 `ClientProxyChain`
- `build_client_chain_group()`：为负载均衡构造多链组

### `src/tcp/proxy_connector.rs` / `proxy_connector_impl.rs`
- `ProxyConnector` trait：在已有流上叠加一层代理协议
- `ProxyConnectorImpl`：包装协议-specific `TcpClientHandler`

### `src/tcp/socket_connector.rs` / `socket_connector_impl.rs`
- `SocketConnector` trait：建立底层 socket（TCP / QUIC）
- `SocketConnectorImpl`：处理 `bind_interface`、keepalive、nodelay、QUIC endpoint 轮询

---

## 四、路由与代理链（根 `src/`）

### `src/client_proxy_selector.rs`
- `ClientProxySelector`：规则匹配引擎
- `judge()`：对目标地址做决策（Allow / Block）
- `ConnectRule` + `NetLocationMask`：CIDR / 主机名 / 端口匹配
- `RoutingCache`：LRU 缓存（当规则数 > 16 或启用主机名解析时激活）

### `src/client_proxy_chain.rs`
- `ClientProxyChain`：单条多跳代理链的运行时表示
- `ClientChainGroup`：多链负载均衡（round-robin）
- `connect_tcp()` / `connect_udp_bidirectional()`：执行链式连接
- `InitialHopEntry`：首跳可以是 Direct 或 Proxy socket

---

## 五、流抽象与数据拷贝（根 `src/`）

### `src/async_stream.rs`
- `AsyncStream`：`AsyncRead + AsyncWrite + AsyncPing`，所有 TCP-like 流的统一接口
- `AsyncMessageStream`：消息/数据报语义（UDP-like）
- `AsyncTargetedMessageStream` / `AsyncSessionMessageStream`：带目标地址或会话 ID 的 UDP 流
- 为 `Box<T>`、`&mut T`、`tokio::net::TcpStream`、`tokio::net::UdpSocket` 提供 blanket impl

### `src/copy_bidirectional.rs`
- `copy_bidirectional()`：双向流拷贝入口
- `CopyBidirectional`：管理两个方向的 `CopyBuffer`
- `CopyBuffer`：自定义环形缓冲区（forked from tokio），支持合作式调度 `coop::poll_proceed` 与 ping 保活

### `src/copy_bidirectional_message.rs`
- 针对 `AsyncMessageStream` 的双向消息拷贝

---

## 六、DNS 系统（`src/dns/`）

### `src/dns/mod.rs`
- 模块说明：支持 system、UDP、TCP、DoT、DoH、DoH3

### `src/dns/builder.rs`
- `build_dns_registry()`：从 `ExpandedDnsGroup` 构建 `DnsRegistry`
- `build_entry_and_plan()`：解析 URL → 构造 client chain → 构造 bootstrap resolver → 解析 DNS 服务器主机名
- `HickoryResolverPlan`：可重建的 hickory resolver 构建计划（用于 `RefreshingResolver`）

### `src/dns/hickory_resolver.rs`
- `HickoryResolver`：封装 `hickory_resolver::Resolver<ProxyRuntimeProvider>`
- 工厂方法：`udp()`、`tcp()`、`tls()`、`https()`、`h3()`

### `src/dns/proxy_runtime.rs`
- `ProxyRuntimeProvider`：实现 hickory 的 `RuntimeProvider`
- `connect_tcp()` 将 DNS 上游的 TCP 连接**通过代理链**路由出去
- `bind_udp()` 支持 `bind_interface`

### `src/dns/composite_resolver.rs`
- `CompositeResolver`：顺序尝试多个 resolver，直到成功

### `src/dns/parsed.rs`
- `ParsedDnsUrl`、`ParsedDnsServer`、`IpStrategy`

### `src/resolver.rs`
- `Resolver` trait
- `NativeResolver` / `CachingNativeResolver` / `RefreshingResolver` / `TimeoutResolver`
- `ResolverCache`：基于 `Shared` future 的并发查询去重

---

## 七、TUN / VPN（`src/tun/`）

### `src/tun/mod.rs`
- `run_tun_server()`：TUN 主入口，创建 fd，启动 smoltcp 线程、TCP handler 任务、UDP handler 任务
- `run_tun_from_config()`：从 `TunConfig` 构建 `TunServerConfig` 并启动
- `handle_tcp_connection()`：将 smoltcp TCP 连接路由到代理链

### `src/tun/tcp_stack_direct.rs`
- `TcpStackDirect`：在独立线程中运行 smoltcp 接口
- `run_direct_stack_thread()`：核心循环（batch 读包 -> 协议过滤 -> TCP socket 管理 -> UDP 透传 -> select 等待）
- `DirectDevice`：smoltcp `Device` trait 实现，裸 fd 读写
- `PooledBuffer`：全局缓冲池（最大 64 个），减少分配

### `src/tun/tcp_conn.rs`
- `TcpConnection`：为 smoltcp socket 实现 `tokio::io::AsyncRead` + `AsyncWrite`
- `TcpConnectionControl`：跨线程共享状态（ring buffer、waker、socket 状态）

### `src/tun/udp_handler.rs`
- `UdpHandler` / `UdpReader` / `UdpWriter`：将原始 IP 包解析为 UDP payload，或反向组装
- `parse_udp_packet()` / `build_udp_packet()`：基于 `smoltcp` wire type 和 `etherparse`

### `src/tun/udp_manager.rs`
- `TunUdpManager`：基于 LRU 的 UDP 会话管理（最大 256 会话，300s 超时）
- `session_task()`：每个本地地址一个任务，管理多目标 destination
- `destination_task()`：每个目标一个任务，持有 `AsyncMessageStream`，`select!` 读写 + 超时
- `create_connection()`：通过 `ClientProxySelector` 和 `ClientChainGroup` 建立 UDP 代理连接

### `src/tun/platform.rs`
- `SocketProtector` trait：Android/iOS 排除 VPN 路由循环
- `PlatformCallbacks` / `PlatformInterface`：移动端生命周期与流量统计回调
- 全局 `GLOBAL_SOCKET_PROTECTOR`：FFI 设置后，所有 socket 创建时自动调用 `protect(fd)`

---

## 八、FFI / 移动端（`src/ffi/`）

### `src/ffi/common.rs`
- `TunServiceHandle`：持有 `tokio::runtime::Runtime` 与 shutdown sender
- `start_from_config()`：FFI 统一启动逻辑
- `stop_service()`：信号关闭，最多等待 5s

### `src/ffi/android.rs`
- JNI 函数：`Java_com_shoesproxy_ShoesNative_init` / `_start` / `_stop` / `_isRunning`
- `FnSocketProtector`：通过 `JavaVM` attach 到 Java 层调用 `protect(int fd)`

### `src/ffi/ios.rs`
- C FFI：`shoes_init` / `shoes_start` / `shoes_stop` / `shoes_is_running` / `shoes_set_log_file`
- `IosSocketProtector`：调用全局 C 函数指针 `ProtectSocketCallback`

### `src/ffi/stub.rs`
- 非移动平台的 no-op stub

---

## 九、TLS / Crypto 抽象（`src/crypto/`、`src/tls_*`）

### `src/crypto/crypto_connection.rs`
- `CryptoConnection` enum：统一 `rustls::ClientConnection` / `rustls::ServerConnection` / `RealityClientConnection` / `RealityServerConnection`

### `src/crypto/crypto_handshake.rs`
- `perform_crypto_handshake()`：在 `AsyncStream` 上驱动 sans-I/O 握手

### `src/crypto/crypto_tls_stream.rs`
- `CryptoTlsStream`：将 `AsyncStream` + `CryptoConnection` 再次包装为 `AsyncStream`

### `src/tls_client_handler.rs`
- `TlsClientHandler`：为内层 client handler 包裹 TLS
- `TlsInnerClientHandler`：`Default` 或 `VisionVless`（VLESS XTLS VISION 特殊处理）

### `src/tls_server_handler.rs`
- `TlsServerHandler`：peek ClientHello，按 SNI 路由到：
  - `TlsServerTarget::Tls`（标准 TLS 终止）
  - `TlsServerTarget::ShadowTls`
  - `TlsServerTarget::Reality`
- `InnerProtocol`：`Normal`、`VisionVless`、`Naive`

### `src/reality_client_handler.rs`
- `RealityClientHandler`：REALITY 客户端封装，支持 `VisionVless` 内层

### `src/rustls_config_util.rs` / `src/rustls_connection_util.rs`
- rustls 配置与连接辅助函数，支持证书 fingerprint pinning（`client_fingerprints`）

---

## 十、传输层服务器（根 `src/`）

### `src/quic_server.rs`
- `start_quic_servers()`：解析配置，按协议分发到 Hysteria2 / TUIC / 通用 QUIC-over-TCP
- `start_quic_server()`：创建 Quinn endpoint，spawn acceptor 任务
- `process_connection()` / `process_streams()`：QUIC 连接/流生命周期

### `src/quic_stream.rs`
- `QuicStream`：将 `quinn::SendStream` + `quinn::RecvStream` 包装为 `AsyncStream`

---

## 十一、入站协议 Handler（根 `src/`）

| 文件 | Handler | 说明 |
|------|---------|------|
| `src/http_handler.rs` | `HttpTcpServerHandler` / `HttpTcpClientHandler` | HTTP CONNECT / 正向代理，可选 Basic Auth |
| `src/socks_handler.rs` | `SocksTcpServerHandler` / `SocksTcpClientHandler` | SOCKS5（CONNECT + UDP ASSOCIATE），用户名密码认证 |
| `src/mixed_handler.rs` | `MixedTcpServerHandler` | 自动探测 HTTP/SOCKS5 |
| `src/port_forward_handler.rs` | `PortForwardServerHandler` | 固定目标端口转发（轮询） |

---

## 十二、协议实现目录

### `src/vless/`
- `vless_client_handler.rs` / `vless_server_handler.rs`
- VISION：`vision_stream.rs`、`vision_pad.rs`、`vision_unpad.rs`、`vision_filter.rs`
- TLS 记录层：`tls_deframer.rs`、`tls_fuzzy_deframer.rs`、`tls_handshake_util.rs`
- 消息流：`vless_message_stream.rs`、`vless_response_stream.rs`（用于 XUDP / UoT V2）

### `src/vmess/`
- `vmess_handler.rs`、`vmess_stream.rs`
- 自研 crypto 原语：`crc32.rs`、`fnv1a.rs`、`md5.rs`、`nonce.rs`、`sha2.rs`、`typed.rs`

### `src/shadowsocks/`
- `shadowsocks_tcp_handler.rs`、`shadowsocks_stream.rs`
- Cipher：`shadowsocks_cipher.rs`（AES-GCM、ChaCha20-Poly1305）
- Key：`shadowsocks_key.rs`、`default_key.rs`、`blake3_key.rs`
- 抗重放：`salt_checker.rs`、`timed_salt_checker.rs`

### `src/trojan_handler.rs`
- `TrojanTcpHandler`：同时实现 server 与 client
- SHA224 密码哈希，可选 Shadowsocks AEAD 外层加密

### `src/snell/`
- `snell_handler.rs`、`snell_fixed_target_stream.rs`、`snell_udp_stream.rs`

### `src/naiveproxy/`
- `naive_client_handler.rs`、`naive_server_handler.rs`（hyper HTTP/2）
- `naive_hyper_service.rs`、`naive_padding_stream.rs`、`h2_multi_stream.rs`
- `user_lookup.rs`：Basic Auth 用户查找

### `src/anytls/`
- `anytls_client_handler.rs`、`anytls_server_handler.rs`
- Session 复用：`anytls_client_session.rs`、`anytls_server_session.rs`
- `anytls_stream.rs`、`anytls_padding.rs`（首 N 包填充）

### `src/h2mux/`
- `h2mux_client_handler.rs`
- Session：`h2mux_client_session.rs`、`h2mux_server_session.rs`
- Stream：`h2mux_client_stream.rs`、`h2mux_server_stream.rs`、`h2mux_stream.rs`
- `h2mux_protocol.rs`、padding、activity tracker / `prepend_stream.rs`

### `src/websocket/`
- `websocket_handler.rs`：client + server
- `websocket_stream.rs`：`WebsocketStream` 包装

### `src/reality/`
- `reality_client_connection.rs` / `reality_server_connection.rs`：sans-I/O 状态机
- `reality_client_handler.rs` / `reality_server_handler.rs`
- AEAD、认证、证书、TLS1.3 密钥派生、消息构造、reader/writer

### `src/shadow_tls/`
- `shadow_tls_client_handler.rs`、`shadow_tls_server_handler.rs`
- `shadow_tls_stream.rs`、`shadow_tls_hmac.rs`

### `src/uot/`
- `uot_common.rs`、`uot_v1_server_stream.rs`、`socks_addr.rs`
- UoT V1 / V2 magic address（`sp.udp-over-tcp.arpa`、`sp.v2.udp-over-tcp.arpa`）

### `src/xudp/`
- `frame.rs`、`message_stream.rs`：`XudpMessageStream`，基于 `u16` session ID 的 UDP 多路复用

---

## 十三、UDP 服务器协议（根 `src/`）

### `src/hysteria2_server.rs`
- `start_hysteria2_server()`：HTTP/3 基础，auth 头验证
- UDP：QUIC datagram 分片/重组；TCP：双向 stream

### `src/tuic_server.rs`
- `start_tuic_server()`：TUIC v5
- Auth：exported keying material + UUID/password
- 双向 stream（TCP）、单向 stream / datagram（UDP）、心跳、会话清理

---

## 十四、辅助模块（根 `src/`）

| 文件 | 职责 |
|------|------|
| `src/address.rs` | `Address`、`NetLocation`、`ResolvedLocation`、`NetLocationMask`、`AddressMask`（CIDR 匹配） |
| `src/socket_util.rs` | TCP keepalive、原始 fd 转换、监听器创建 |
| `src/stream_reader.rs` | `StreamReader`：带 peek 功能的缓冲读取，用于协议探测 |
| `src/logging.rs` | 多输出（stderr + 文件）日志后端 |
| `src/uuid_util.rs` | UUID 生成辅助 |
