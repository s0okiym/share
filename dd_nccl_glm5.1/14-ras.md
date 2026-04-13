# NCCL RAS 容错可用性服务

RAS (Resilience Availability Service) 是 NCCL 内置的分布式监控子系统，在所有 NCCL 进程间形成监控网格，实现故障检测、peer 生命周期管理和外部状态查询。

---

## 1. RAS 架构总览

```mermaid
flowchart TD
    subgraph "每个 NCCL 进程"
        RT["RAS 线程\n(rasThreadMain)\n每进程一个"]
        NP["通知管道\nRAS_ADD_RANKS\nRAS_TERMINATE"]
        LS["RAS 网络监听\n随机端口"]
        CS["RAS 客户端监听\n端口 28028"]
        PS["动态 peer sockets"]
        CL["动态 client sockets"]
    end

    RT --> NP
    RT --> LS
    RT --> CS
    RT --> PS
    RT --> CL
```

---

## 2. RAS 环形网络拓扑

```mermaid
flowchart LR
    P0["Process 0\nnextLink→P1\nprevLink←P3"] --> P1["Process 1\nnextLink→P2\nprevLink←P0"]
    P1 --> P2["Process 2\nnextLink→P3\nprevLink←P1"]
    P2 --> P3["Process 3\nnextLink→P0\nprevLink←P2"]
    P3 --> P0
```

每个 link (`rasLink`) 包含一个 `rasLinkConn` 链表：
- 第一个条目是主连接
- 额外条目是故障恢复时创建的 fallback 连接
- 主连接恢复后，fallback 被清理 (`rasLinkSanitizeFallbacks`)

Peer 计算逻辑 (`rasLinkCalculatePeer`):
- 沿 `rasPeers` 排序数组，按 link 方向遍历
- 跳过 dead peer
- Fallback-of-fallback 时跳过同一节点的所有 peer

---

## 3. 连接生命周期

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Connecting : rasConnCreate/ncclSocketConnect (async)
    Connecting --> Handshake : ncclSocketAccept (对端接受)
    Handshake --> Ready : CONNINIT/CONNINITACK 交换
    Ready --> Terminating : SHUT_WR (发送端关闭)
    Terminating --> Closed : EOF (接收端关闭)
    Ready --> Closed : 错误/超时
    Connecting --> Closed : 连接失败
    Handshake --> Closed : 握手失败
```

**连接竞态解决**: 地址较小的一方发起连接。

**握手协议**:
- 发送: `RAS_MSG_CONNINIT` (NCCL 版本、监听地址、peers hash)
- 响应: `RAS_MSG_CONNINITACK` (可包含 NACK 拒绝)

---

## 4. 消息协议

### 4.1 消息类型

| 类型 | 方向 | 用途 |
|------|------|------|
| `CONNINIT` | 连接方 → 接受方 | 握手发起 |
| `CONNINITACK` | 接受方 → 连接方 | 握手响应 |
| `KEEPALIVE` | 双向 | 心跳 (peers hash, dead hash, link mask, wallclock) |
| `PEERSUPDATE` | 双向 | peer 列表同步 (delta + full dead peers) |
| `COLLREQ` | 发起方 → 下游 | 集合操作请求 (broadcast/gather) |
| `COLLRESP` | 下游 → 上游 | 集合操作响应 |

### 4.2 消息发送

消息入队到 per-connection `ncclIntruQueue<rasMsgMeta>` 发送队列，包含：
- 入队时间
- 发送进度偏移
- 消息长度

### 4.3 Peer 更新去重

每个连接追踪 4 个 hash 值：

| Hash | 用途 |
|------|------|
| `lastSentPeersHash` | 上次发送的 peer 列表 hash |
| `lastRecvPeersHash` | 上次接收的 peer 列表 hash |
| `lastSentDeadPeersHash` | 上次发送的 dead peer hash |
| `lastRecvDeadPeersHash` | 上次接收的 dead peer hash |

仅当 hash 不同时才发送/处理更新。

---

## 5. 故障检测与恢复

### 5.1 超时升级机制

```mermaid
flowchart TD
    A["正常运行"] --> B["1s: 无数据发送\n→ 发送 KEEPALIVE"]
    B --> C["5s: 无数据接收\n→ experiencingDelays=true\n→ 启动 fallback 连接\n→ 从集合操作中清除该连接"]
    C --> D["20s: 仍无数据\n→ 终止 socket\n→ 开始重连"]
    D --> E["1s 间隔: 重连尝试"]
    E --> F["5s: 重连警告"]
    F --> G["60s: 广播 RAS_BC_DEADPEER\n→ 永久宣告 peer 死亡"]

    D --> H{重连成功?}
    H -->|"是"| I["rasConnResume:\nclear experiencingDelays\n重置超时\n清理 fallback"]
    H -->|"否"| E
```

### 5.2 Dead Peer 广播

```mermaid
flowchart TD
    A["rasPeerDeclareDead(peer)"] --> B["添加到 rasDeadPeers\n排序数组"]
    B --> C["重新计算 dead peers hash"]
    C --> D["终止该 peer 所有连接"]
    D --> E["RAS_BC_DEADPEER\n沿环双向洪泛"]

    E --> F["收到广播的进程:\nrasConnDisconnect\nrasPeerDeclareDead (本地)"]
    F --> G{peer 已知 dead?}
    G -->|"否"| H["本地宣告 + 重广播"]
    G -->|"是"| I["不重广播 (hash 去重)"]

    E --> J["64 条 LRU 历史\nrasCollHistory\n防止重复处理"]
```

---

## 6. 集合操作

### 6.1 Broadcast

```mermaid
flowchart TD
    A["RAS_BC_DEADPEER"] --> B["沿环双向洪泛"]
    B --> C["无需响应"]
    C --> D["64 条 LRU 历史去重\nrasCollHistory"]
```

### 6.2 Gather 集合

| 类型 | 收集内容 |
|------|---------|
| `RAS_COLL_CONNS` | 连接行程时间统计 (min/max/sum/count) |
| `RAS_COLL_COMMS` | 每通信器数据: rank 信息、操作计数、状态标志 |

```mermaid
flowchart TD
    A["Gather 请求\nRAS_COLL_CONNS/COMMS"] --> B["沿两个 link 转发"]
    B --> C["每个进程:\n累积下游 peer 响应\n+ 自己的数据\n发送合并结果上游"]
    C --> D["重复检测:\nhistory + active collectives 列表"]
    D --> E["超时:\n5s 软超时 (检查延迟)\n10s 硬超时 (放弃)"]
```

---

## 7. Peer 管理

### 7.1 Peer 信息结构

```mermaid
classDiagram
    class rasPeerInfo {
        +addr: ncclSocketAddress
        +pid: int
        +cudaDevs: uint64_t (bitmask)
        +nvmlDevs: uint64_t (bitmask)
        +hostHash: uint64_t
        +pidHash: uint64_t
    }
```

### 7.2 Peer 添加

```mermaid
flowchart TD
    A["ncclRasAddRanks(comm, rasRanks, nranks)"] --> B["转换 rasRankInit → rasPeerInfo\n合并多 GPU 进程\n(同 PID 合并为一个 peer)"]
    B --> C["合并到排序 rasPeers 数组"]
    C --> D["rasNetUpdatePeers\n通过两个 link 传播更新"]
```

---

## 8. 外部客户端接口

### 8.1 文本协议

```mermaid
flowchart TD
    A["客户端连接 (端口 28028)"] --> B["CLIENT PROTOCOL <version>\n↔ SERVER PROTOCOL <version>"]
    B --> C["可选: TIMEOUT <seconds>"]
    C --> D["可选: SET FORMAT text|json"]
    D --> E["STATUS / VERBOSE STATUS"]
    D --> F["MONITOR [lifecycle,trace,all]"]
```

### 8.2 状态查询流程

```mermaid
flowchart TD
    A["STATUS 查询"] --> B["RAS_CLIENT_INIT"]
    B --> C["触发 RAS_COLL_CONNS\n收集连接统计"]
    C --> D["RAS_CLIENT_CONNS"]
    D --> E["触发 RAS_COLL_COMMS\n收集通信器数据"]
    E --> F["RAS_CLIENT_COMMS"]
    F --> G["RAS_CLIENT_FINISHED\n返回结果"]
```

### 8.3 监控事件

| 事件组 | 事件 |
|--------|------|
| **lifecycle** | PEER_NEW, PEER_DEAD, PEER_CONNECTING, PEER_CONNECTED, PEER_RECOVERED |
| **trace** | PEER_UNRESPONSIVE, PEER_DISCONNECTED, PEER_SEND_STUCK, PEER_KEEPALIVE_TIMEOUT, PEER_TIMEOUT_DEAD, PEER_RETRY, PEER_INIT_TIMEOUT |
| **all** | 以上所有 |

---

## 9. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/ras/ras.cc` | ~800 | RAS 主线程、事件循环、消息分发 |
| `src/ras/ras_internal.h` | ~600 | 所有 RAS 数据结构和声明 |
| `src/ras/rasnet.cc` | ~800 | 网络连接、keep-alive、故障恢复 |
| `src/ras/peers.cc` | ~400 | Peer 管理、dead peer 广播 |
| `src/ras/collectives.cc` | ~300 | RAS 集合操作 (broadcast/gather) |
| `src/ras/client_support.cc` | ~400 | 外部客户端接口 |
| `src/ras/client.cc` | ~200 | 独立 CLI 客户端 |
