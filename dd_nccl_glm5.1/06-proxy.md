# NCCL 代理线程架构

代理线程负责主机端的数据推进，特别是 NET 和 SHM 传输的发送/接收操作。GPU 内核只负责在 GPU 侧读写缓冲区，实际的网络 I/O 由代理线程完成。

---

## 1. 三线程模型

每个 GPU 设备运行三个代理线程：

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
    PT -->|"调用回调"| NET["transport.proxyProgress\n网络 I/O (ncclNet->isend/irecv/test)"]
    UT -->|"GetFd/QueryFd"| FD["cuMem FD 交换\n跨进程虚拟内存导入"]
```

| 线程 | 入口函数 | 职责 |
|------|---------|------|
| Service | ncclProxyService | 接受连接、处理 RPC、创建异步操作 |
| Progress | ncclProxyProgress | 拉取操作、执行数据推进 |
| UDS | ncclProxyServiceUDS | Unix Domain Socket 处理 cuMem FD |

---

## 2. Service Thread 主循环

```mermaid
flowchart TD
    A["ncclProxyService 主循环\nwhile (stop==RUNNING || npeers>0)"] --> B["检查 abortFlag\nif set: stop = PROXY_ABORT"]

    B --> C["poll() 所有 peer socket + listenSock\ntimeout: 0 if asyncOps, 500ms otherwise"]

    C --> D{listenSock 有事件?}
    D -->|"是"| E["ncclSocketAccept\n接受新 peer 连接\npeers[s], pollfds[s]"]
    D -->|"否"| F["遍历 peer sockets"]
    E --> F

    F --> G["推进该 peer 的所有 asyncOps\nproxyProgressAsync"]

    G --> H{POLLIN 事件?}
    H -->|"是"| I["读取消息类型"]
    H -->|"否"| J{POLLHUP?}
    J -->|"是"| K["关闭连接\nnpeers--"]
    J -->|"否"| L["继续下一个 peer"]

    I --> M{消息类型?}
    M -->|Stop| N["stop = PROXY_STOP"]
    M -->|Close| O["关闭该 peer 连接"]
    M -->|Init/Setup/Connect\nRegister/Deregister| P["proxyServiceInitOp\n创建 asyncOp"]
    P --> Q["proxyProgressAsync\n异步推进操作"]

    Q --> R{操作完成 (done=1)?}
    R -->|"是"| S["发送 RPC 响应给用户线程\nncclProxyRpcResponseHeader + respBuff"]
    R -->|"否"| T["保留 asyncOp\n下次重试"]

    S --> L
    T --> L
    K --> L
    N --> U["循环后清理:\nncclProxyProgressDestroy\nncclProxyFreeConnections\nncclSocketClose(listenSock)"]
```

---

## 3. Progress Thread 主循环

```mermaid
flowchart TD
    A["ncclProxyProgress 主循环"] --> B["progressOps(proxyState, state, state->active, &idle)"]

    B --> C["遍历 active ncclProxyArgs 链表"]
    C --> D["调用 args->progress(proxyState, args)\n(即 connection->tcomm->proxyProgress)"]
    D --> E{args->state == ncclProxyOpNone?}
    E -->|"是"| F["removeOp — 移除已完成操作"]
    E -->|"否"| G["保留在 active 链表"]

    F --> H{idle 或 appendCounter 阈值?}
    G --> H
    H -->|"是"| I["ncclProxyGetPostedOps\n从共享内存拉取新 ops"]
    H -->|"否"| J["继续循环"]

    I --> K["ncclProxyGetPostedOps 内部"]
    K --> K1["lock pool->mutex\n(try_lock if active, block if idle)"]
    K1 --> K2["遍历 ops chain"]
    K2 --> K3["ProxyAppend: 转换\nncclProxyOp → ncclProxyArgs"]
    K3 --> K4["链接到 active 链表"]
    K4 --> K5["释放 freed ops 回 peer pool"]
    K5 --> K6["unlock mutex"]

    J --> A
    K6 --> A
```

---

## 4. 连接状态机

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

---

## 5. RPC 协议

### 5.1 用户线程 → 代理线程 (请求)

```mermaid
flowchart LR
    A["ncclProxyCallAsync"] --> B["发送: type(int)"]
    B --> C["发送: connection(void*)"]
    C --> D["发送: reqSize(int)"]
    D --> E["发送: respSize(int)"]
    E --> F["发送: reqBuff(reqSize bytes)"]
    F --> G["发送: opId(void*)"]
    G --> H["入队期望响应"]
```

### 5.2 代理线程 → 用户线程 (响应)

```mermaid
flowchart LR
    A["proxyProgressAsync\n操作完成"] --> B["发送: ncclProxyRpcResponseHeader\n{opId, res, respSize}"]
    B --> C["发送: respBuff(respSize bytes)"]
    C --> D["出队 asyncOp"]
```

### 5.3 用户线程接收响应

```mermaid
flowchart TD
    A["ncclPollProxyResponse"] --> B["读取: ncclProxyRpcResponseHeader"]
    B --> C["匹配 opId 与期望响应"]
    C --> D["读取: respBuff"]
    D --> E["返回结果给调用者"]
```

---

## 6. 数据推进路径

### 6.1 用户线程到代理线程的操作提交

```mermaid
flowchart TD
    A["内核启动后\nncclLaunchKernelAfter"] --> B["hostStreamPlanTask"]
    B --> C["uploadProxyOps(comm, plan)"]
    C --> D["ncclProxySaveOp\n确定哪些 peer 需要 proxy ops"]
    D --> E["ncclLocalOpAppend\n追加 ncclProxyOp 到 per-peer 共享内存队列"]
    E --> F["ncclProxyStart\nncclProxyPost → 信号 progress thread"]
    F --> G["唤醒条件变量\nproxyState->cond.notify_one()"]
```

### 6.2 NET Send Proxy Progress

```mermaid
flowchart TD
    A["sendProxyProgress"] --> B{state == Ready?}
    B -->|"是"| C["初始化: base, posted=0\ntransmitted=0, done=0\n设置 sendMhandle"]
    B -->|"否"| D["Progress 阶段"]
    C --> D

    D --> E["Phase 1 POST:\n递增 sub->posted by sliceSteps\n设置 sendHead\n写入 connFifo offset"]

    E --> F["Phase 2 TRANSMIT:\n检查 GPU 数据就绪\nrecvTail > base+transmitted"]
    F --> F1["LL/LL128: 验证 flag 有效性"]
    F1 --> F2["ncclNet->isend(buff, size,\nmhandle, &request)"]
    F2 --> F3{request != NULL?}
    F3 -->|"是"| F4["sub->transmitted += sliceSteps"]
    F3 -->|"否"| F5["重试 isend"]

    F4 --> G["Phase 3 DONE:\nncclNet->test(request, &done, &size)"]
    G --> G1{done?}
    G1 -->|"是"| G2["sub->done += sliceSteps\n更新 sendHead"]
    G1 -->|"否"| H["下次重试"]
    G2 --> I{sub->done == nsteps?}
    I -->|"是"| J["args->done++"]
    I -->|"否"| H
    J --> K{args->done == nsubs?}
    K -->|"是"| L["args->state = ncclProxyOpNone"]
    K -->|"否"| H
```

### 6.3 NET Recv Proxy Progress

```mermaid
flowchart TD
    A["recvProxyProgress"] --> B{state == Ready?}
    B -->|"是"| C["初始化 + 按 recvComm 分组\n设置 recvMhandle"]
    B -->|"否"| D["Progress 阶段"]
    C --> D

    D --> E["Phase 1 POST:\nncclNet->irecv(netRecvComm,\nsubCount, ptrs, sizes, tags, mhandles, &request)"]
    E --> E1{request 有效?}
    E1 -->|"是"| E2["sub->posted += sliceSteps"]
    E1 -->|"否"| E3["重试 irecv"]

    E2 --> F["Phase 2 RECEIVE:\nncclNet->test(request, &done, sizes)"]
    F --> F1{done?}
    F1 -->|"是"| F2["sub->received += sliceSteps"]
    F1 -->|"否"| F3["下次重试"]

    F2 --> F4{needFlush?}
    F4 -->|"是"| F5["ncclNet->iflush() 或\nPCI-E 读屏障"]
    F4 -->|"否"| G
    F5 --> G["Phase 3 TRANSMIT:\n确认 flush 完成\n更新 recvTail = base + transmitted"]

    G --> H["Phase 4 DONE:\n等待 GPU 确认\n读 sendMem->head"]
    H --> H1["sub->done += sliceSteps"]
    H1 --> I{所有 sub 完成?}
    I -->|"是"| J["args->state = ncclProxyOpNone"]
    I -->|"否"| F3
```

---

## 7. 代理消息类型

| 类型 | 值 | 处理函数 | 用途 |
|------|---|---------|------|
| ncclProxyMsgInit | 1 | proxyConnInit | 创建新连接 |
| ncclProxyMsgSharedInit | 2 | tcomm->proxySharedInit | 多通道共享初始化 |
| ncclProxyMsgSetup | 3 | tcomm->proxySetup | 代理端初始化 |
| ncclProxyMsgConnect | 4 | tcomm->proxyConnect | 建立数据通路 |
| ncclProxyMsgStart | 5 | (未使用) | — |
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
| `ncclProxyState` | proxy.h | 代理全局状态 (线程、socket、操作池) |
| `ncclProxyConnection` | proxy.h | 每连接代理端状态 (transportResources, tcomm, 连接状态) |
| `ncclProxyArgs` | proxy.h | 进度操作结构 (subs[], progress 回调, 状态) |
| `ncclProxySubArgs` | proxy.h | 子操作 (connection, base, posted/received/transmitted/done, requests[]) |
| `ncclProxyOp` | proxy.h | 共享内存中的操作描述 |
| `ncclProxyPool` | proxy.h | per-peer 共享内存操作队列 |

---

## 9. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/proxy.cc` | 1967 | 代理线程完整实现 |
| `src/include/proxy.h` | ~450 | 代理数据结构和声明 |
