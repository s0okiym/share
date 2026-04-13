# NCCL 传输层架构

传输层是 NCCL 中数据搬运的核心抽象。所有传输实现统一的 `ncclTransport` 接口，使上层算法无需关心底层是 NVLink、PCIe、共享内存还是网络。

---

## 1. 传输接口定义

### 1.1 ncclTransport 结构

```mermaid
classDiagram
    class ncclTransport {
        +name: char[8]
        +canConnect(comm, graph, info1, info2) int
        +send: ncclTransportComm
        +recv: ncclTransportComm
    }

    class ncclTransportComm {
        +setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex)
        +connect(comm, connectInfo, nranks, rank, connector)
        +free(comm, connector)
        +proxySharedInit(connection, proxyState, nChannels)
        +proxySetup(connection, proxyState, req, resp, done)
        +proxyConnect(connection, proxyState, req, resp, done)
        +proxyFree(connection, proxyState)
        +proxyProgress(proxyState, args)
        +proxyRegister(connection, proxyState, req, resp, done)
        +proxyDeregister(connection, proxyState, req, done)
    }

    ncclTransport --> ncclTransportComm : send
    ncclTransport --> ncclTransportComm : recv
```

### 1.2 接口函数分类

| 函数 | 执行线程 | 用途 |
|------|---------|------|
| `canConnect` | 用户线程 | 查询两个 rank 间能否建立连接 |
| `setup` | 用户线程 | 初始化连接、分配缓冲区、与代理通信 |
| `connect` | 用户线程 | 建立实际数据通路、映射远端内存 |
| `free` | 用户线程 | 释放连接器资源 |
| `proxySharedInit` | 代理线程 | 共享初始化 (多通道复用) |
| `proxySetup` | 代理线程 | 代理端初始化 (分配缓冲区、监听) |
| `proxyConnect` | 代理线程 | 代理端连接建立 (网络连接、内存注册) |
| `proxyFree` | 代理线程 | 代理端资源释放 |
| `proxyProgress` | 代理线程 | 数据推进核心循环 |
| `proxyRegister` | 代理线程 | 缓冲区注册 |
| `proxyDeregister` | 代理线程 | 缓冲区注销 |

---

## 2. 五种传输类型

| 传输 | ID | 名称 | 节点范围 | proxyProgress | 典型带宽 |
|------|---|------|---------|---------------|---------|
| P2P | 0 | "P2P" | 节点内 | 仅 CE memcpy 模式 | NVLink: 20-40 GB/s |
| SHM | 1 | "SHM" | 节点内 | 有 | 内存带宽 |
| NET | 2 | "NET" | 跨节点 | 有 (核心) | IB: 12.5-25 GB/s |
| COLLNET | 3 | "CollNet" | 跨节点 | 有 | 取决于 SHARP |
| NVLS | — | "NVLS" | 节点内 | — | NVLink multicast |

---

## 3. P2P 传输

### 3.1 P2P 类型

```mermaid
flowchart TD
    A["p2pGetInfo(comm, myInfo, peerInfo)"] --> B{进程关系?}
    B -->|"同 PID"| C["P2P_DIRECT\n直接指针访问\n同 GPU: 直接 memcpy\n不同 GPU: enablePeerAccess + 映射"]
    B -->|"不同 PID"| D{cuMem 可用?}
    D -->|"是"| E["P2P_CUMEM\n不同进程, cuMem API\n跨进程共享 CUDA 内存"]
    D -->|"否"| F["P2P_IPC\n不同进程, CUDA IPC\nlegacy 方式"]
    A --> G{需要中间 GPU?}
    G -->|"intermediateRank != -1"| H["P2P_INTERMEDIATE\n经中间 GPU 中转\n源→中间GPU→目标"]
    G -->|"否"| I["直接路径"]
```

### 3.2 P2P Send Setup 流程

```mermaid
flowchart TD
    A["p2pSendSetup(comm, graph, myInfo, peerInfo,\nconnectInfo, send, channelId, connIndex)"]
    A --> B["p2pGetInfo — 确定 useRead, intermediateRank, p2pType"]

    B --> C["ncclProxyConnect(comm, TRANSPORT_P2P, 1, info->rank, &send->proxyConn)"]
    C --> D["ncclProxyCallBlocking(ncclProxyMsgSetup, &ncclP2pRequest)"]

    D --> E["代理线程: p2pSendProxySetup"]
    E --> E1["ncclP2pAllocateShareableBuffer\n分配 CUDA 可共享缓冲区"]
    E1 --> E2["返回 ncclP2pBuff\ndirectPtr + IPC 描述符"]

    E2 --> F["用户线程: p2pMap — 导入远端缓冲区"]
    F --> F1{P2P 类型?}
    F1 -->|"DIRECT 同 GPU"| F2["直接指针"]
    F1 -->|"DIRECT 不同 GPU"| F3["enablePeerAccess + 映射"]
    F1 -->|"CUMEM"| F4["ncclP2pImportShareableBuffer"]
    F1 -->|"IPC"| F5["cudaIpcOpenMemHandle"]
```

### 3.3 P2P Connect 流程

```mermaid
flowchart TD
    A["p2pSendConnect(comm, connectInfo, nranks, rank, send)"]
    A --> B["p2pMap — 映射远端内存 → remDevMem"]
    B --> C["设置 send->conn.buffs[]\n从远端内存"]
    C --> D["设置 conn.tail / conn.head / conn.ptrExchange"]
    D --> E{useMemcpy (CE 模式)?}
    E -->|"是"| F["ncclProxyCallBlocking(ncclProxyMsgConnect)\n替换 SIMPLE buff 为 CE 设备缓冲区"]
    E -->|"否"| G["完成"]
    F --> G
```

---

## 4. NET 传输

### 4.1 Send Setup 流程

```mermaid
flowchart TD
    A["sendSetup(comm, graph, myInfo, peerInfo,\nconnectInfo, send, channelId, connIndex)"]
    A --> B["ncclTopoGetNetDev — 确定网络设备"]
    B --> C["ncclTopoCheckGdr — 确定 GDR 模式"]
    C --> D["ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn)"]
    D --> E["ncclProxyCallBlocking(ncclProxyMsgSetup, &setupReq)"]

    E --> F["代理线程: sendProxySetup"]
    F --> F1["分配 sendNetResources"]
    F1 --> F2["获取 NIC 属性: ncclNet->getProperties()"]
    F2 --> F3["*done = 1, 返回"]

    F3 --> G["填充 connectInfo: proxyRank + network handle"]
```

### 4.2 Recv Setup 流程

```mermaid
flowchart TD
    A["recvSetup(comm, graph, myInfo, peerInfo,\nconnectInfo, recv, channelId, connIndex)"]
    A --> B["ncclTopoGetNetDev — 确定网络设备"]
    B --> C["ncclTopoCheckGdr"]
    C --> D["ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn)"]
    D --> E["ncclProxyCallBlocking(ncclProxyMsgSetup, &setupReq, connectInfo)"]

    E --> F["代理线程: recvProxySetup"]
    F --> F1["分配 recvNetResources"]
    F1 --> F2["ncclNet->listen(netContext, netDev, respBuff, &netListenComm)\n获取网络监听 handle"]
    F2 --> F3["handle 写入 respBuff 返回"]

    F3 --> G["connectInfo 包含网络 handle\n跨 rank 交换后供 sendConnect 使用"]
```

### 4.3 NET Connect 流程

```mermaid
flowchart TD
    subgraph "sendConnect"
        S1["ncclProxyCallAsync(ncclProxyMsgConnect)"]
        S1 --> S2["代理: sendProxyConnect"]
        S2 --> S2a["ncclNet->connect(handle)\n建立网络发送连接"]
        S2a --> S2b["分配 + 注册内存区域\nhost + device buffers"]
        S2b --> S2c["ncclNet->regMr / regMrDmaBuf"]
        S2c --> S2d["返回 connectMap"]
        S2d --> S3["ncclPollProxyResponse → 获取 connectMap"]
    end

    subgraph "recvConnect"
        R1["ncclProxyCallAsync(ncclProxyMsgConnect)"]
        R1 --> R2["代理: recvProxyConnect"]
        R2 --> R2a["ncclNet->accept()\n接受网络连接"]
        R2a --> R2b["分配 + 注册内存区域"]
        R2b --> R2c["返回 connectMap"]
        R2c --> R3["ncclPollProxyResponse → 获取 connectMap"]
    end

    S3 --> T["导入远端内存 (map SHM, import IPC)"]
    R3 --> T
    T --> T1["设置 conn.head / tail / buffs / stepSize"]
    T1 --> T2["设置 proxyProgress 回调"]
```

---

## 5. 传输连接建立总体序列

一次完整的跨节点集合操作连接建立序列：

```mermaid
sequenceDiagram
    participant U0 as Rank 0 用户线程
    participant P0 as Rank 0 代理线程
    participant U1 as Rank 1 用户线程
    participant P1 as Rank 1 代理线程

    Note over U0,U1: Setup 阶段
    U0->>P0: ncclProxyMsgSetup (sendSetup)
    P0-->>U0: 响应 (sendProxySetup 完成)
    U1->>P1: ncclProxyMsgSetup (recvSetup)
    P1-->>U1: 响应 (recvProxySetup 完成 + network handle)

    Note over U0,U1: Bootstrap 交换 network handle
    U0->>U1: bootstrapAllGather 交换 connectInfo

    Note over U0,U1: Connect 阶段
    U0->>P0: ncclProxyMsgConnect (sendConnect)
    P0->>P1: ncclNet->connect(handle)
    U1->>P1: ncclProxyMsgConnect (recvConnect)
    P1->>P0: ncclNet->accept()
    P0-->>U0: connectMap 响应
    P1-->>U1: connectMap 响应

    Note over U0,U1: 映射远端内存
    U0->>U0: import 远端内存, 设置 connector
    U1->>U1: import 远端内存, 设置 connector
```

---

## 6. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/include/transport.h` | ~250 | ncclTransport / ncclTransportComm 接口定义 |
| `src/transport/p2p.cc` | ~1300 | P2P 传输实现 |
| `src/transport/shm.cc` | ~800 | SHM 传输实现 |
| `src/transport/net.cc` | ~1900 | NET 传输实现 |
| `src/transport/net_socket.cc` | ~600 | NET Socket 后端 |
| `src/transport/net_ib/` | ~2000 | NET IB 后端 |
| `src/transport/coll_net.cc` | ~1200 | CollNet 传输实现 |
| `src/transport/nvls.cc` | ~800 | NVLS 传输实现 |
