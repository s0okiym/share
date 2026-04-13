# NCCL 代理线程架构

代理线程负责主机端的数据推进，特别是 NET 和 SHM 传输的发送/接收操作。GPU 内核只负责在 GPU 侧读写缓冲区，实际的网络 I/O 由代理线程完成。这种架构的核心原因是：网络操作（如 IB Verbs 的 send/recv）需要在 CPU 端调用，GPU 内核无法直接发起网络 I/O。

---

## 1. 三线程模型

每个 GPU 设备运行三个代理线程，各司其职：

```mermaid
flowchart TD
    subgraph "代理线程架构"
        ST["Service Thread\nncclProxyService\n处理 RPC 消息\nInit/Setup/Connect/Register"]
        PT["Progress Thread\nncclProxyProgress\n执行数据推进\ntransport.proxyProgress"]
        UT["UDS Thread\nncclProxyServiceUDS\ncuMem FD 交换\nGetFd/QueryFd"]
    end

    subgraph "共享内存操作池"
        SM["ncclProxyPool\nper-peer 环形队列\nncclProxyOp 链表"]
    end

    PEER["Peer Sockets\n与用户线程通信"]

    ST -->|"读取消息"| PEER
    ST -->|"创建 asyncOp"| SM
    SM -->|"拉取 ops"| PT
    PT -->|"调用回调"| NET["transport.proxyProgress\n网络 I/O: ncclNet->isend/irecv/test"]
    UT -->|"GetFd/QueryFd"| FD["cuMem FD 交换\n跨进程虚拟内存导入"]
```

| 线程 | 入口函数 | 职责 |
|------|---------|------|
| Service | ncclProxyService | 接受连接、处理 RPC、创建异步操作 |
| Progress | ncclProxyProgress | 拉取操作、执行数据推进 |
| UDS | ncclProxyServiceUDS | Unix Domain Socket 处理 cuMem FD |

三线程分离的关键设计理念：Service 线程处理连接建立等低频控制操作，Progress 线程专注于高频的数据推进，两者互不阻塞。UDS 线程处理 cuMem 文件描述符交换，这是一种特殊的跨进程内存共享机制。

---

## 2. Service Thread 主循环

Service 线程是代理的控制面，通过 poll() 多路复用处理来自所有 peer 的 RPC 请求。

```mermaid
flowchart TD
    A["ncclProxyService 主循环\nwhile stop==RUNNING || npeers>0"] --> B["检查 abortFlag\nif set: stop = PROXY_ABORT"]

    B --> C["poll 所有 peer socket + listenSock\ntimeout: 0 if asyncOps, 500ms otherwise\n注释: never let proxy service\nthread block in poll"]

    C --> D{"listenSock 有事件?"}
    D -->|"是"| E["ncclSocketAccept\n接受新 peer 连接"]
    D -->|"否"| F["遍历 peer sockets"]
    E --> F

    F --> G["推进该 peer 的所有 asyncOps\nproxyProgressAsync"]

    G --> H{"POLLIN 事件?"}
    H -->|"是"| I["读取消息类型"]
    H -->|"否"| J{"POLLHUP?"}
    J -->|"是"| K["关闭连接, npeers--"]
    J -->|"否"| L["继续下一个 peer"]

    I --> M{"消息类型?"}
    M -->|Stop| N["stop = PROXY_STOP"]
    M -->|Close| O["关闭该 peer 连接"]
    M -->|"Init/Setup/Connect\nRegister/Deregister"| P["proxyServiceInitOp\n创建 asyncOp"]

    P --> Q["proxyProgressAsync\n异步推进操作"]
    Q --> R{"操作完成 (done==1)?"}
    R -->|"是"| S["发送 RPC 响应给用户线程\nncclProxyRpcResponseHeader + respBuff"]
    R -->|"否"| T["保留 asyncOp, 下次重试"]

    S --> L
    T --> L
    K --> L
    N --> U["循环后清理:\nncclProxyProgressDestroy\nncclProxyFreeConnections\nncclSocketClose(listenSock)"]
```

Service 线程的关键设计：**永不阻塞**。poll 超时设置为 0（有异步操作时）或 500ms（无操作时），确保 abortFlag 能被及时检查。如果有异步操作正在进行，使用 0 超时实现忙等，避免延迟数据推进。

---

## 3. Progress Thread 主循环

Progress 线程是代理的数据面，负责实际的网络 I/O。

```mermaid
flowchart TD
    A["ncclProxyProgress 主循环"] --> B["progressOps(proxyState, state, state->active, &idle)"]

    B --> C["遍历 active ncclProxyArgs 链表"]
    C --> D["调用 args->progress(proxyState, args)\n即 connection->tcomm->proxyProgress"]
    D --> E{"args->state == ncclProxyOpNone?"}
    E -->|"是"| F["removeOp — 移除已完成操作"]
    E -->|"否"| G["保留在 active 链表"]

    F --> H{"idle 或 appendCounter 阈值?"}
    G --> H
    H -->|"是"| I["ncclProxyGetPostedOps\n从共享内存拉取新 ops"]
    H -->|"否"| J["继续循环"]

    I --> K["ncclProxyGetPostedOps 内部"]
    K --> K1["lock pool->mutex\ntry_lock if active, block if idle\n活跃时非阻塞尝试, 空闲时阻塞等待"]
    K1 --> K2["遍历 ops chain"]
    K2 --> K3["ProxyAppend: 转换\nncclProxyOp → ncclProxyArgs\n设置 progress 回调和 sub 参数"]
    K3 --> K4["链接到 active 链表"]
    K4 --> K5["释放 freed ops 回 peer pool"]
    K5 --> K6["unlock mutex"]

    J --> A
    K6 --> A
```

Progress 线程的锁策略很精巧：当有活跃操作时，使用 `try_lock` 非阻塞获取 mutex，避免阻塞数据推进；当空闲时，阻塞等待 mutex，让 CPU 可以休眠。`appendCounter` 阈值机制确保不会频繁获取 mutex：每隔一定次数的操作推进后才拉取新操作。

---

## 4. 连接状态机

代理连接经历从初始化到就绪的状态转换：

```mermaid
stateDiagram-v2
    [*] --> connUninitialized : 初始状态
    connUninitialized --> connInitialized : ncclProxyMsgInit\nproxyConnInit()
    connUninitialized --> connSharedInitialized : ncclProxyMsgSharedInit\ntcomm->proxySharedInit()
    connInitialized --> connSetupDone : ncclProxyMsgSetup\ntcomm->proxySetup()
    connSharedInitialized --> connSetupDone : ncclProxyMsgSetup
    connSetupDone --> connConnected : ncclProxyMsgConnect\ntcomm->proxyConnect()
    connConnected --> [*] : 关闭
```

- **SharedInit** 路径用于多通道共享同一个连接资源（如 NET 传输中多个通道共用一个网络连接）
- **Init** 路径为每个通道创建独立连接
- 只有到达 `connConnected` 状态后，连接才能用于数据传输

---

## 5. RPC 协议

用户线程和代理线程之间通过 socket 进行 RPC 通信，采用请求-响应模式。

### 5.1 请求格式

| 字段 | 类型 | 说明 |
|------|------|------|
| type | int | 消息类型 (Init/Setup/Connect/Register/...) |
| connection | void* | 代理端连接标识 |
| reqSize | int | 请求体大小 |
| respSize | int | 期望响应体大小 |
| reqBuff | bytes | 请求体数据 |
| opId | void* | 操作标识，用于匹配响应 |

### 5.2 响应格式

| 字段 | 类型 | 说明 |
|------|------|------|
| opId | void* | 匹配请求的操作标识 |
| res | ncclResult_t | 操作结果 |
| respSize | int | 响应体大小 |
| respBuff | bytes | 响应体数据 |

异步操作模式：用户线程通过 `ncclProxyCallAsync` 发送请求后，可以继续做其他工作，稍后通过 `ncclPollProxyResponse` 接收响应。阻塞模式 `ncclProxyCallBlocking` 则循环等待直到收到响应。

---

## 6. 数据推进路径

### 6.1 从内核启动到代理提交

```mermaid
flowchart TD
    A["内核启动后\nncclLaunchKernelAfter"] --> B["hostStreamPlanTask"]
    B --> C["uploadProxyOps(comm, plan)\n将 plan 中的 proxy 操作提交"]
    C --> D["ncclProxySaveOp\n确定哪些 peer 需要 proxy ops\n遍历 proxyOpQueue"]
    D --> E["ncclLocalOpAppend\n追加 ncclProxyOp 到 per-peer 共享内存队列\n设置 sub 参数: base, nsteps, sliceSteps"]
    E --> F["ncclProxyStart\nncclProxyPost → 信号 progress thread"]
    F --> G["唤醒条件变量\nproxyState->cond.notify_one()"]
```

### 6.2 NET Send Proxy Progress

发送端代理推进分为三个阶段，采用流水线设计：

```mermaid
flowchart TD
    A["sendProxyProgress"] --> B{"state == Ready?"}
    B -->|"是"| C["初始化: base, posted=0\ntransmitted=0, done=0"]
    B -->|"否"| D["Progress 阶段"]
    C --> D

    D --> E["Phase 1 POST:\n递增 sub->posted by sliceSteps\n推进 sendHead\n写入 connFifo offset\n通知接收端有数据待发"]

    E --> F["Phase 2 TRANSMIT:\n检查 GPU 数据就绪\nrecvTail > base+transmitted"]
    F --> F1["LL/LL128: 验证 flag 有效性\n确保数据一致性"]
    F1 --> F2["ncclNet->isend(buff, size,\ntag, mhandle, &request)\n提交网络发送"]

    F2 --> G["Phase 3 DONE:\nncclNet->test(request, &done, &size)\n检查发送完成"]
    G --> G1{"done?"}
    G1 -->|"是"| G2["重置 connFifo 槽\nsub->done += sliceSteps"]
    G1 -->|"否"| H["下次重试"]
    G2 --> I{"所有 sub 完成?"}
    I -->|"是"| J["args->state = ncclProxyOpNone\n操作完成"]
    I -->|"否"| H
```

### 6.3 NET Recv Proxy Progress

接收端代理推进分为四个阶段：

```mermaid
flowchart TD
    A["recvProxyProgress"] --> B["Phase 1 POST:\nncclNet->irecv(ptrs, sizes, tags, mhandles)\n发布接收缓冲区\n按 recvComm 分组批量 irecv"]

    B --> C["Phase 2 RECEIVE:\nncclNet->test(request, &done, sizes)\n检查网络接收完成"]
    C --> C1{"done?"}
    C1 -->|"是"| C2["检查是否需要 GDR flush\nGDR 写入需要显式刷新才能被 GPU 看到"]
    C1 -->|"否"| C3["下次重试"]

    C2 --> D["Phase 3 FLUSH:\n执行 flush 操作\nGDRcopy: PCIe 读屏障\n或 ncclNet->iflush()"]
    D --> D1["flush 完成: 更新 recvTail\n通知 GPU 数据已就绪"]

    D1 --> E["Phase 4 DONE:\n等待 GPU 确认\n读 sendMem->head"]
    E --> E1["sub->done += sliceSteps\n推进完成计数"]
    E1 --> F{"所有 sub 完成?"}
    F -->|"是"| G["args->state = ncclProxyOpNone"]
    F -->|"否"| C3
```

接收端的 GDR flush 是性能关键路径。当 NIC 通过 RDMA 直接写入 GPU 内存时，GPU 可能看不到最新数据（因为 PCIe 写入可能被缓存）。flush 操作通过读取 GPU 内存的某个地址来强制刷新 PCIe 缓冲区，确保 GPU 能看到所有数据。

---

## 7. 代理消息类型

| 类型 | 值 | 处理函数 | 用途 |
|------|---|---------|------|
| ncclProxyMsgInit | 1 | proxyConnInit | 创建新连接 |
| ncclProxyMsgSharedInit | 2 | tcomm->proxySharedInit | 多通道共享初始化 |
| ncclProxyMsgSetup | 3 | tcomm->proxySetup | 代理端初始化（分配缓冲区/监听） |
| ncclProxyMsgConnect | 4 | tcomm->proxyConnect | 建立数据通路（网络连接/内存注册） |
| ncclProxyMsgClose | 6 | 关闭连接 | 关闭 peer 连接 |
| ncclProxyMsgAbort | 7 | 中止操作 | 异常终止 |
| ncclProxyMsgStop | 8 | 停止线程 | 终止 Service 线程 |
| ncclProxyMsgGetFd | 9 | proxyGetFd | UDS: 获取 cuMem FD |
| ncclProxyMsgQueryFd | 10 | proxyQueryFd | UDS: 查询 FD |
| ncclProxyMsgRegister | 11 | tcomm->proxyRegister | 缓冲区注册 |
| ncclProxyMsgDeregister | 12 | tcomm->proxyDeregister | 缓冲区注销 |

---

## 8. 关键数据结构

| 结构体 | 文件 | 用途 |
|--------|------|------|
| `ncclProxyState` | proxy.h | 代理全局状态（线程、socket、操作池、条件变量） |
| `ncclProxyConnection` | proxy.h | 每连接代理端状态（transportResources, tcomm, 连接状态） |
| `ncclProxyArgs` | proxy.h | 进度操作结构（subs[], progress 回调, 状态, done 计数） |
| `ncclProxySubArgs` | proxy.h | 子操作（connection, base, posted/received/transmitted/done, requests） |
| `ncclProxyOp` | proxy.h | 共享内存中的操作描述（由用户线程写入，Progress 线程读取） |
| `ncclProxyPool` | proxy.h | per-peer 共享内存操作队列（mutex 保护） |

---

## 9. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/proxy.cc` | 1967 | 代理线程完整实现：Service/Progress/UDS 主循环、RPC 处理、操作调度 |
| `src/include/proxy.h` | ~450 | 代理数据结构和声明 |
