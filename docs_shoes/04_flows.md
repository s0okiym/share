# 关键流程与流程图

本文档用 Mermaid 流程图展示 `shoes` 中最重要的端到端流程。

> **语法约定**：所有流程图节点文本均用双引号包裹，确保渲染兼容性。

---

## TCP 代理链完整流程

从外部客户端建立一条 TCP 连接，到通过代理链访问远程目标，再到双向数据转发的完整生命周期。

```mermaid
flowchart TD
    START["客户端发起 TCP 连接"] --> ACCEPT["run_tcp_server()<br/>listener.accept()"]
    ACCEPT --> SPAWN["tokio::spawn<br/>process_stream()"]
    SPAWN --> SETUP_TIMEOUT["timeout(60s)<br/>setup_server_stream()"]
    SETUP_TIMEOUT --> SERVER_HANDLER["server_handler.setup_server_stream()<br/>协议握手: SOCKS5 / HTTP / Shadowsocks / VMess / VLESS / ..."]
    SERVER_HANDLER --> RESULT{"TcpServerSetupResult"}

    RESULT -->|TcpForward| TCP_FORWARD["提取 remote_location<br/>proxy_selector<br/>need_initial_flush<br/>initial_remote_data"]
    RESULT -->|BidirectionalUdp| UDP_BIDIR["UDP-over-TCP 分支"]
    RESULT -->|AlreadyHandled| ALREADY["handler 已自行处理<br/>流程结束"]

    TCP_FORWARD --> JUDGE["client_proxy_selector.judge()"]
    JUDGE --> CACHE{"查 RoutingCache"}
    CACHE -->|命中| DECISION["返回缓存决策"]
    CACHE -->|未命中| MATCH_RULE["线性扫描 ConnectRule<br/>match_mask: CIDR / 主机名 / 端口"]
    MATCH_RULE --> DECISION
    DECISION --> ALLOW_BLOCK{"ConnectDecision"}

    ALLOW_BLOCK -->|Block| CLOSE_BLOCK["关闭连接<br/>流程结束"]
    ALLOW_BLOCK -->|Allow| CHAIN_GROUP["chain_group.connect_tcp()"]

    CHAIN_GROUP --> SELECT_HOP["select_initial_hop_entry()<br/>round-robin"]
    SELECT_HOP --> INIT_HOP{"InitialHopEntry"}
    INIT_HOP -->|Direct| DIRECT_CONNECT["socket.connect(target)"]
    INIT_HOP -->|Proxy| PROXY_CONNECT["socket.connect(proxy_loc)<br/>proxy.setup_tcp_stream()"]

    DIRECT_CONNECT --> SUBSEQ["逐层 subsequent_hops<br/>proxy.setup_tcp_stream(stream, next_target)"]
    PROXY_CONNECT --> SUBSEQ
    SUBSEQ --> CLIENT_RESULT["返回 TcpClientSetupResult<br/>client_stream + early_data"]

    CLIENT_RESULT --> WRITE_RESP["写 connection_success_response<br/>到 server_stream"]
    WRITE_RESP --> WRITE_EARLY["写 initial_remote_data / early_data<br/>到对端"]
    WRITE_EARLY --> COPY["copy_bidirectional()<br/>CopyBidirectional 双向并发拷贝"]
    COPY --> SHUTDOWN["shutdown() 两端流<br/>流程结束"]

    UDP_BIDIR --> JUDGE_UDP["proxy_selector.judge()<br/>获取 chain_group"]
    JUDGE_UDP --> UDP_CONNECT["chain_group.connect_udp_bidirectional()"]
    UDP_CONNECT --> UDP_COPY["copy_bidirectional_message()<br/>流程结束"]
```

### 关键文件与行号参考

| 阶段 | 文件 | 关键函数/行 |
|------|------|------------|
| Accept | `src/tcp/tcp_server.rs` | `run_tcp_server` ~L30 |
| Per-connection | `src/tcp/tcp_server.rs` | `process_stream` ~L121 |
| Server setup | `src/tcp/tcp_handler.rs` | `TcpServerHandler` ~L72 |
| Routing | `src/client_proxy_selector.rs` | `judge` ~L283 |
| Chain build | `src/tcp/chain_builder.rs` | `build_client_proxy_chain` ~L19 |
| Chain exec | `src/client_proxy_chain.rs` | `connect_tcp` ~L242 |
| Bidirectional copy | `src/copy_bidirectional.rs` | `copy_bidirectional` ~L294 |

---

## 配置加载与校验流程

从 YAML 文件到运行时 `ValidatedConfigs` 的完整转换链。

```mermaid
flowchart TD
    CLI["main() 参数解析"] --> LOAD["load_configs(&paths)"]
    LOAD --> PARSE["serde_yaml::from_str::<Vec<Config>><br/>每个文件 -> Vec<Config>"]
    PARSE --> CONVERT_CERT["convert_cert_paths()<br/>PEM 路径 -> 内联数据"]
    CONVERT_CERT --> VALIDATE["create_server_configs()<br/>src/config/validate.rs"]

    VALIDATE --> COLLECT["收集 raw_client_groups<br/>rule_groups / dns_groups<br/>server_configs / tun_configs"]
    COLLECT --> TOPO_CLIENT["resolve_client_groups_topologically()<br/>Kahn 算法 + 循环检测"]
    TOPO_CLIENT --> EMBED_PEM["embed PEM 到 client configs"]
    EMBED_PEM --> EXTRACT_INLINE["extract_inline_dns()<br/>内联 DNS -> 生成 __inline_dns_N 组"]
    EXTRACT_INLINE --> EXPAND_DNS_COMP["expand_dns_groups_composition()<br/>拓扑展开组引用"]
    EXPAND_DNS_COMP --> SORT_BOOTSTRAP["topological_sort_dns_groups_by_bootstrap()<br/>按 bootstrap 依赖排序"]
    SORT_BOOTSTRAP --> EXPAND_DNS_SPEC["expand_dns_specs()<br/>解析 URL / 展开 client_chains / 校验协议兼容性"]
    EXPAND_DNS_SPEC --> VALIDATE_SERVERS["validate_server_config()<br/>validate_tun_config()"]
    VALIDATE_SERVERS --> RETURN["返回 ValidatedConfigs<br/>{ configs, dns_groups }"]

    RETURN --> BUILD_REGISTRY["dns::build_dns_registry(dns_groups)"]
    BUILD_REGISTRY --> START_SERVERS["启动 TCP / QUIC / TUN 服务器"]
    START_SERVERS --> WATCH["notify::RecommendedWatcher<br/>监听配置文件 Modify"]
    WATCH --> MODIFIED{"文件修改?"}
    MODIFIED -->|是| RELOAD["abort 所有任务<br/>sleep 3s<br/>drain 事件<br/>回到 load_configs()"]
    MODIFIED -->|否| RUNNING["正常运行"]
    RELOAD --> LOAD
```

---

## DNS 解析与代理链路由流程

一条 DNS 查询如何被解析，以及 DNS 上游如何走代理链。

```mermaid
flowchart TD
    QUERY["应用查询域名 example.com"] --> RESOLVER["Resolver::resolve_location()"]
    RESOLVER --> WHICH{"Resolver 类型?"}

    WHICH -->|NativeResolver| OS_LOOKUP["tokio::net::lookup_host()"]
    WHICH -->|CachingNativeResolver| CHECK_CACHE["查 FxHashMap 缓存<br/>TTL = 1h"]
    CHECK_CACHE -->|命中| RETURN_CACHE["返回缓存结果"]
    CHECK_CACHE -->|未命中| OS_LOOKUP
    WHICH -->|HickoryResolver| HICKORY["hickory_resolver.lookup_ip()"]
    WHICH -->|CompositeResolver| TRY_SEQ["顺序尝试多个 resolver<br/>直到成功"]

    HICKORY --> RUNTIME["ProxyRuntimeProvider<br/>提供 TCP/UDP/QUIC 连接"]
    RUNTIME --> CONNECT_TCP["connect_tcp(target)"]
    CONNECT_TCP --> CHAIN_GROUP["chain_group.connect_tcp(target, bootstrap_resolver)"]
    CHAIN_GROUP --> PROXY_CHAIN["代理链建立<br/>同 TCP 代理链流程"]
    PROXY_CHAIN --> DNS_UPSTREAM["到达 DNS 上游服务器<br/>返回解析结果"]

    OS_LOOKUP --> RETURN_RESULT["返回 Vec<SocketAddr>"]
    DNS_UPSTREAM --> RETURN_RESULT
    TRY_SEQ --> RETURN_RESULT
    RETURN_RESULT --> USE_ADDR["后续用于 connect_tcp()<br/>或作为代理链目标"]

    subgraph "DNS 注册表构建"
        direction TB
        BUILD["build_dns_registry()"] --> PER_GROUP["对每个 ExpandedDnsGroup"]
        PER_GROUP --> ENTRY["build_entry_and_plan()"]
        ENTRY --> PARSE_URL["ParsedDnsUrl::parse()"]
        ENTRY --> BUILD_CHAIN["build_client_chain_group()"]
        ENTRY --> BOOTSTRAP["构造 bootstrap resolver<br/>组名查找 或 NativeResolver"]
        BOOTSTRAP --> RESOLVE_HOST["bootstrap.resolve_location()<br/>解析 DNS 服务器主机名"]
        RESOLVE_HOST --> SAVE["注册到 DnsRegistry"]
    end
```

---

## TUN TCP 流程

TUN 设备模式下，一个 TCP  SYN 包如何被拦截、经过代理链、到达远端。

```mermaid
flowchart TD
    PACKET["应用发送 IP 包到 TUN fd"] --> STACK_THREAD["smoltcp OS 线程<br/>run_direct_stack_thread()"]
    STACK_THREAD --> READ["DirectDevice::try_recv()<br/>batch 读最多 64 包"]
    READ --> FILTER["should_filter_packet()<br/>丢弃广播/多播/无效包"]
    FILTER --> PROTO{"IP 协议?"}

    PROTO -->|TCP| TCP_INFO["extract_tcp_info()<br/>解析 src/dst + 检测 SYN"]
    PROTO -->|UDP| UDP_TX["udp_tx.send(pkt)<br/>直接透传给 tokio UDP handler"]
    PROTO -->|ICMP| ICMP["iface.poll()<br/>smoltcp 处理"]
    PROTO -->|其他| DROP["丢弃"]

    TCP_INFO --> NEW_SYN{"is_syn 且<br/>非已有连接?"}
    NEW_SYN -->|否| EXISTING["查找已有 TcpConnection<br/>enqueue_recv_data / dequeue_send_data"]
    NEW_SYN -->|是| CHECK_LIMIT["检查 MAX_CONCURRENT_CONNECTIONS<br/>当前 1024"]
    CHECK_LIMIT -->|超限| REJECT_SYN["丢弃 SYN"]
    CHECK_LIMIT -->|通过| CREATE["create_tcp_connection()<br/>创建 smoltcp TcpSocket + TcpConnection"]
    CREATE --> SEND_CHAN["new_conn_tx.send(NewTcpConnection)"]

    SEND_CHAN --> TOKIO_TCP["tokio 任务<br/>handle_tcp_connection()"]
    TOKIO_TCP --> TCP_JUDGE["proxy_selector.judge(target, resolver)"]
    TCP_JUDGE --> TCP_ALLOW{"Allow?"}
    TCP_ALLOW -->|Block| TCP_DROP["丢弃连接<br/>通知 smoltcp close"]
    TCP_ALLOW -->|Allow| TCP_CHAIN["chain_group.connect_tcp(remote_location, resolver)"]
    TCP_CHAIN --> TCP_COPY["tokio::io::copy_bidirectional()<br/>TcpConnection <-> client_stream"]
    TCP_COPY --> TCP_SHUTDOWN["connection.shutdown()<br/>通知 smoltcp 关闭 socket"]

    EXISTING --> WAKE["wake_receiver / wake_sender<br/>unpark() 唤醒 async task"]
```

---

## TUN UDP 流程

TUN 模式下 UDP 包不走 smoltcp socket，而是直接通过 session-manager 路由。

```mermaid
flowchart TD
    UDP_PACKET["UDP IP 包到达 TUN"] --> STACK_READ["smoltcp 线程读取"]
    STACK_READ --> UDP_TX2["udp_tx.send(pkt) -> tokio"]
    UDP_TX2 --> UDP_READER["UdpReader::parse_udp_packet()<br/>提取 payload, src_addr, dst_addr"]
    UDP_READER --> UDP_MANAGER["TunUdpManager::run()<br/>select! 接收"]
    UDP_MANAGER --> HANDLE["handle_packet(local_addr, remote_addr, payload)"]
    HANDLE --> LRU_LOOKUP["LruCache<SocketAddr, Session><br/>按 local_addr 查找"]
    LRU_LOOKUP -->|无 session| CREATE_SESSION["create_session()<br/>spawn session_task"]
    LRU_LOOKUP -->|有 session| FORWARD_SESSION["tx.try_send()<br/>转发到 session_task"]

    CREATE_SESSION --> SESSION_TASK["session_task()"]
    SESSION_TASK --> DEST_LOOKUP["HashMap<NetLocation, DestinationEntry><br/>按 remote_addr 查找"]
    DEST_LOOKUP -->|无 destination| CREATE_DEST["create_connection()<br/>proxy_selector.judge()<br/>chain_group.connect_udp_bidirectional()"]
    DEST_LOOKUP -->|有 destination| WRITE_DEST["write_tx.try_send()<br/>转发到 destination_task"]

    CREATE_DEST --> DEST_TASK["destination_task()<br/>持有 AsyncMessageStream"]
    DEST_TASK --> DEST_LOOP["select! 循环"]
    DEST_LOOP -->|Action::Read| READ_MSG["stream.poll_read_message()<br/>读到后 response_tx -> manager"]
    DEST_LOOP -->|Action::Write| WRITE_MSG["stream.poll_write_message()<br/>发送 UDP payload"]
    DEST_LOOP -->|Action::Timeout| DEST_TIMEOUT["120s 无活动<br/>self-terminate"]

    READ_MSG --> MANAGER_WRITE["TunUdpManager::write_to_tun()<br/>build_udp_packet() -> UdpWriter"]
    MANAGER_WRITE --> BACK_TO_TUN["写回 TUN fd<br/>应用收到响应"]

    WRITE_DEST --> DEST_LOOP
```

---

## QUIC 服务器连接处理流程

QUIC endpoint 接受连接后，如何复用现有 TCP 协议 handler 处理 bidirectional streams。

```mermaid
flowchart TD
    START_QUIC["start_quic_servers()"] --> PARSE_CONFIG["解析 ServerConfig<br/>ALPN / 证书 / 密钥"]
    PARSE_CONFIG --> DISPATCH{"ServerProxyConfig?"}
    DISPATCH -->|Hysteria2| HY2["hysteria2_server::start_hysteria2_server()"]
    DISPATCH -->|TuicV5| TUIC["tuic_server::start_tuic_server()"]
    DISPATCH -->|其他| GENERIC["start_quic_server()<br/>创建 quinn::Endpoint"]

    GENERIC --> ACCEPT_LOOP["endpoint.accept().await<br/>spawn acceptor 任务"]
    ACCEPT_LOOP --> PROCESS_CONN["process_connection()"]
    PROCESS_CONN --> AWAIT_CONN["conn.await?"]
    AWAIT_CONN --> STREAM_LOOP["connection.accept_bi().await<br/>循环接收 bidirectional streams"]
    STREAM_LOOP --> PROCESS_STREAMS["process_streams()"]

    PROCESS_STREAMS --> WRAP["QuicStream::from(send, recv)<br/>包装为 AsyncStream"]
    WRAP --> SERVER_SETUP["server_handler.setup_server_stream(quic_stream)<br/>60s 超时"]
    SERVER_SETUP --> QUIC_RESULT{"TcpServerSetupResult"}

    QUIC_RESULT -->|TcpForward| Q_FORWARD["setup_client_tcp_stream()<br/>写 connection_success_response<br/>写 initial_remote_data"]
    Q_FORWARD --> Q_COPY["copy_bidirectional()<br/>QuicStream <-> outbound stream"]
    Q_COPY --> Q_SHUTDOWN["shutdown()<br/>流结束"]

    QUIC_RESULT -->|BidirectionalUdp| Q_UDP_BIDIR["chain_group.connect_udp_bidirectional()<br/>copy_bidirectional_message()"]
    QUIC_RESULT -->|MultiDirectionalUdp| Q_MULTI_UDP["run_udp_routing()<br/>ServerStream::Targeted"]
    QUIC_RESULT -->|SessionBasedUdp| Q_SESSION_UDP["run_udp_routing()<br/>ServerStream::Session"]
    QUIC_RESULT -->|AlreadyHandled| Q_ALREADY["无操作"]

    HY2 --> HY2_END["HTTP/3 + QUIC datagram<br/>分片重组 + auth<br/>流程结束"]
    TUIC --> TUIC_END["TUIC v5 会话管理<br/>stream / datagram / heartbeat<br/>流程结束"]
```

---

## 代理链连接建立时序

以一条两跳代理链（直连 -> Shadowsocks -> 目标）为例，展示时序。

```mermaid
sequenceDiagram
    autonumber
    participant C as "客户端"
    participant S as "shoes 入站"
    participant H1 as "InitialHop<br/>Direct Socket"
    participant P1 as "Hop 1<br/>Shadowsocks ClientHandler"
    participant SS as "Shadowsocks 服务端"
    participant T as "最终目标"

    C->>S: "建立 TCP 连接"
    S->>S: "server_handler.setup_server_stream()<br/>解析目标地址"
    S->>S: "client_proxy_selector.judge()<br/>决定走 chain_group[0]"
    S->>H1: "socket.connect(proxy_server_addr)"
    H1->>SS: "TCP 三次握手"
    SS-->>H1: "连接建立"
    H1->>P1: "proxy.setup_tcp_stream(stream, target_addr)"
    P1->>SS: "Shadowsocks 握手 + 目标地址"
    SS->>T: "连接目标"
    T-->>SS: "连接建立"
    SS-->>P1: "握手完成"
    P1-->>S: "返回 client_stream"
    S->>S: "copy_bidirectional()"
    S->>P1: "应用数据"
    P1->>SS: "加密数据"
    SS->>T: "明文数据"
    T-->>SS: "响应"
    SS-->>P1: "加密响应"
    P1-->>S: "解密响应"
    S-->>C: "响应"
```

---

## TUN / VPN 三线程数据交换

展示 smoltcp 线程、TCP handler task、UDP manager task 之间的协作关系。

```mermaid
flowchart LR
    subgraph "OS Thread: smoltcp"
        direction TB
        TUN_FD["TUN fd"]
        DEVICE["DirectDevice"]
        TCP_STACK["smoltcp Interface<br/>TcpSocket 管理"]
        RING_BUF["RingBuffer<br/>send/recv"]

        TUN_FD -->|"read"| DEVICE
        DEVICE -->|"store_packet"| TCP_STACK
        TCP_STACK -->|"enqueue_recv<br/>dequeue_send"| RING_BUF
    end

    subgraph "Tokio: TCP Handler"
        direction TB
        TCP_CHAN["mpsc channel<br/>NewTcpConnection"]
        TCP_CONN["TcpConnection<br/>AsyncRead/AsyncWrite"]
        TCP_PROXY["proxy_selector<br/>+ chain_group"]
        TCP_COPY["copy_bidirectional"]

        TCP_CHAN --> TCP_CONN
        TCP_CONN --> TCP_PROXY
        TCP_PROXY --> TCP_COPY
    end

    subgraph "Tokio: UDP Manager"
        direction TB
        UDP_CHAN["mpsc channel<br/>raw UDP packets"]
        UDP_MGR["TunUdpManager<br/>LRU session cache"]
        UDP_DEST["destination_task<br/>AsyncMessageStream"]

        UDP_CHAN --> UDP_MGR
        UDP_MGR --> UDP_DEST
    end

    RING_BUF <-->|"wake / unpark"| TCP_CONN
    TCP_STACK -->|"new_conn_tx"| TCP_CHAN
    DEVICE -->|"udp_tx"| UDP_CHAN
    UDP_DEST -->|"response_tx"| UDP_MGR
    UDP_MGR -->|"UdpWriter<br/>build_udp_packet"| TUN_FD
```
