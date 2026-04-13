# NCCL RMA 远程内存访问

RMA (Remote Memory Access) 提供单边通信原语 (Put, Signal, WaitSignal)，支持两个可并行运行的后端：CE (Copy Engine, 节点内 NVLink) 和 Proxy (GIN 网络, 跨节点)。

---

## 1. RMA 架构总览

```mermaid
flowchart TD
    A["RMA 操作入口"] --> B{目标 peer 可达性?}

    B -->|"LSA 可达\n(同节点 NVLink)"| C["CE 后端\ncudaMemcpyAsync\n+ CUDA stream mem op"]
    B -->|"GIN 可达\n(跨节点)"| D["Proxy 后端\nncclGin->iput/iputSignal\n+ CPU 进度线程"]
    B -->|"混合: 部分 LSA\n部分 GIN"| E["CE + Proxy 并行"]

    E --> E1["用户流记录 event"]
    E1 --> E2["CE 流等待 event\n执行 CE 部分"]
    E2 --> E3["用户流执行 Proxy 部分"]
    E3 --> E4["CE event 同步回用户流"]
```

---

## 2. 核心数据结构

### 2.1 顶层结构

```mermaid
classDiagram
    class ncclRmaArgs {
        +ctx: int (RMA context ID)
        +func: int (ncclFuncPutSignal/Signal/WaitSignal)
        +nRmaTasks: int
        +nRmaTasksProxy: int
        +nRmaTasksCe: int
    }

    class ncclRmaState {
        +rmaProxyState: ncclRmaProxyState
        +rmaCeState: ncclRmaCeState
    }

    ncclRmaArgs --> ncclRmaState : references
```

### 2.2 CE 后端结构

```mermaid
classDiagram
    class ncclRmaCeCtx {
        +signalOpSeqs: uint64_t[] (per-rank)
        +signalsWin: ncclDevrWindow*
        +signalsDev: uint64_t*
        +signalsHost: uint64_t[] (expected values)
    }

    class ncclRmaCeState {
        +rmaCeCtxCount: int
        +rmaCeCtxs: ncclRmaCeCtx[]
        +ceStream: cudaStream_t
        +ceEvent: cudaEvent_t
    }

    ncclRmaCeState --> ncclRmaCeCtx : rmaCeCtxs[]
```

### 2.3 Proxy 后端结构

```mermaid
classDiagram
    class ncclRmaProxyDesc {
        +srcOff: size_t
        +srcHandle: void*
        +dstOff: size_t
        +dstHandle: void*
        +size: size_t
        +targetRank: int
        +signal: ncclRmaSignal_t
        +seq: uint64_t
        +rmaDescState: int (Pending/InProgress)
        +request: void*
    }

    class ncclRmaProxyCtx {
        +ginCollComm: void*
        +pendingQueues: ncclRmaProxyDesc[] (per-peer circular buffer)
        +pis: uint64_t[] (producer indices)
        +cis: uint64_t[] (consumer indices)
        +rmaProxyInProgressQueues: node*[] (per-peer linked list)
        +opSeqs/readySeqs/doneSeqs: seq counters
        +signalsDev: uint64_t*
        +signalsHost: uint64_t[]
    }

    class ncclRmaProxyState {
        +ncclGin: ncclGin_t*
        +ginInstance: void*
        +ginComms: void*[] (up to NCCL_GIN_MAX_CONNECTIONS)
        +rmaProxyCtxs: ncclRmaProxyCtx[]
        +needsProxyProgress: int
        +thread: pthread_t
    }

    ncclRmaProxyState --> ncclRmaProxyCtx : rmaProxyCtxs[]
    ncclRmaProxyCtx --> ncclRmaProxyDesc : pendingQueues[]
```

---

## 3. 操作类型

| 操作 | 说明 | CE 路径 | Proxy 路径 |
|------|------|---------|-----------|
| **Put** | 写数据到远端 rank | cudaMemcpyAsync | ncclGin->iput |
| **Put+Signal** | 写数据 + 原子通知 | cudaMemcpyAsync + 写 signal | ncclGin->iputSignal |
| **Signal** | 仅原子通知 (无数据) | 写 signal | ncclGin->iputSignal (data=0) |
| **WaitSignal** | 等待远端通知 | WAIT_VALUE_64 | WAIT_VALUE_64 |

---

## 4. Put 执行流程

### 4.1 任务调度 (scheduleRmaTasksToPlan)

```mermaid
flowchart TD
    A["scheduleRmaTasksToPlan"] --> B["找到第一个非空 RMA context 队列"]
    B --> C["创建 ncclRmaArgs (ctx ID + func type)"]

    C --> D{func 类型?}
    D -->|"WaitSignal"| D1["拆分 peer 为:\nCE-accessible (LSA) + Proxy-accessible\n分别创建 ncclTaskRma"]
    D -->|"Put/Signal"| D2["按 isLsaAccessible 路由到 CE 或 Proxy"]

    D2 --> E{canBatchRmaTasks?\n(同 context + 同 func\n或都是 Put/Signal)}
    E -->|"是"| F["批量合并连续任务"]
    E -->|"否"| G["独立任务"]
```

### 4.2 CE Put 执行

```mermaid
flowchart TD
    A["ncclRmaPutCe(task, comm, stream)"] --> B["对每个任务:\n解析 peer 缓冲区\nncclDevrGetLsaRankPtr (LSA 偏移转换)"]

    B --> C["cudaMemcpyAsync\n(peerBuff, srcBuff, bytes\ndeviceToDevice, stream)"]

    C --> D{Signal 启用?}
    D -->|"是"| E["解析 peer signal 缓冲区槽\noffset = comm->rank"]
    E --> F["递增 per-peer 序列号"]
    F --> G["cudaMemcpyAsync\n写 signal 值 (hostToDevice)\n到 signalsDev[peerRank]"]
    D -->|"否"| H["完成"]
    G --> H
```

### 4.3 Proxy Put 执行

```mermaid
flowchart TD
    A["ncclRmaPutProxy(task, comm, stream)"] --> B["对每个任务:\n构建 ncclRmaProxyDesc\nsrcOff/dstOff/srcHandle/dstHandle/size"]

    B --> C{Signal 启用?}
    C -->|"是"| D["设置 signal descriptor:\nop=NCCL_NET_SIGNAL_OP_ADD\noffset=rank*sizeof(uint64_t)\nvalue=1"]
    C -->|"否"| E["不设置 signal"]

    D --> F["入队到 per-peer 环形缓冲区\npendingQueues[peer][pi]"]
    E --> F
    F --> G["原子递增 pi"]

    G --> H["CUDA stream batch mem ops"]
    H --> H1["1. WRITE: readySeqsDev[peer] = seq\n通知 CPU 进度线程"]
    H1 --> H2["2. WAIT: doneSeqsDev[peer] >= seq\n等待网络操作完成"]
```

---

## 5. Proxy 进度线程

```mermaid
flowchart TD
    A["ncclRmaProxyProgressThread\n(while ginProgress==1)"] --> B["对每个 context:\nncclRmaProxyProgress"]

    B --> C["对每个 peer:"]
    C --> D["1. ncclRmaProxyPollCompletion\n测试 in-progress 队列头部\nncclGin->test(request)\n完成: 写 doneSeqs[peer] = seq\n(RELEASE 顺序)\n移动到下一个"]
    D --> E["2. ncclRmaProxyPollDesc\n检查 pending 队列\nreadySeq >= desc->seq?"]
    E --> F{有就绪描述符?}
    F -->|"是"| G["ncclGin->iput (无 signal)\n或 ncclGin->iputSignal (有 signal)"]
    G --> H["移动到 in-progress 队列"]
    F -->|"否"| I["继续下一个 peer"]
    H --> I
```

---

## 6. WaitSignal 执行

### 6.1 CE WaitSignal

```mermaid
flowchart TD
    A["ncclRmaWaitSignalCe(task, comm, stream)"] --> B["对每个 peer:\n计算 waitValue\n= signalsHost[peerRank] + nsignals[i]"]
    B --> C["更新 signalsHost[peerRank] = waitValue"]
    C --> D["CU_STREAM_MEM_OP_WAIT_VALUE_64\n等待 signalsDev[peerRank] >= waitValue\nGEQ 语义"]
```

### 6.2 Proxy WaitSignal

```mermaid
flowchart TD
    A["ncclRmaWaitSignalProxy(task, comm, stream)"] --> B["对每个 peer:\n计算 expected value"]
    B --> C["CU_STREAM_MEM_OP_WAIT_VALUE_64\n等待 proxyCtx->signalsDev[peerRank]\n>= expected value"]
```

---

## 7. 信号协议

### 7.1 信号缓冲区布局

```
offset [0]:        rank 0 的信号 (uint64_t)
offset [1]:        rank 1 的信号
...
offset [nRanks-1]: rank nRanks-1 的信号
offset [nRanks]:   聚合信号 (保留)
```

### 7.2 Proxy 三计数器同步

```mermaid
flowchart LR
    subgraph "GPU Stream"
        G1["opSeqs: 递增序号\n(描述符序列号)"]
        G2["readySeqs: GPU 写入\n(CUDA batch WRITE)"]
        G3["doneSeqs: GPU 等待\n(CUDA batch WAIT)"]
    end

    subgraph "CPU Progress Thread"
        C1["读 readySeqs\n知道描述符就绪"]
        C2["发出 RDMA 操作\niput/iputSignal"]
        C3["写 doneSeqs\n(RELEASE 顺序)"]
    end

    G1 --> G2 --> C1 --> C2 --> C3 --> G3
```

这个三计数器协议实现了 GPU→CPU→Network→CPU→GPU 的完整同步管道，无需任何内核启动。

---

## 8. CE 初始化

```mermaid
flowchart TD
    A["ncclRmaCeInit(comm)"] --> B["确保对称内存运行时初始化\nncclDevrInitOnce"]
    B --> C["分配 numRmaCtx 个 CE context"]
    C --> D["对每个 context:"]
    D --> D1["分配设备信号缓冲区\n(nRanks+1) * sizeof(uint64_t)"]
    D1 --> D2["注册为集合对称窗口\nncclDevrWindowRegisterInGroup"]
    D2 --> D3["创建 host shadow\nncclShadowPoolToHost"]
    D3 --> D4["分配 host 跟踪缓冲区\n+ per-rank 序列计数器"]
    D4 --> E["创建非阻塞 CUDA stream\n+ disable-timing event\n(用于 CE 并行)"]
```

---

## 9. Proxy 连接建立

```mermaid
flowchart TD
    A["ncclRmaProxyConnectOnce(comm)"] --> B["初始化 GIN 实例\nncclGin->init()"]
    B --> C["发现本地 GIN 设备\nncclTopoGetLocalGinDevs"]
    C --> D["AllGather 最小 comm 数"]
    D --> E["对每个 GIN communicator:\nlisten → AllGather handles\n→ connect → get properties"]
    E --> F["创建 rmaProxyCtxCount 个\n虚拟 RMA proxy context\n轮询映射到物理 GIN communicator"]
    F --> G{需要 CPU 进度?}
    G -->|"是"| H["启动 ncclRmaProxyProgressThread"]
    G -->|"否"| I["完成"]
    H --> I
```

---

## 10. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/rma/rma.cc` | ~300 | RMA 顶层调度、Put/WaitSignal 入口 |
| `src/rma/rma_ce.cc` | ~300 | CE 后端 (NVLink) |
| `src/rma/rma_proxy.cc` | ~600 | Proxy 后端 (GIN 网络)、进度线程 |
| `src/include/rma/rma.h` | ~40 | 顶层数据结构 |
| `src/include/rma/rma_ce.h` | ~50 | CE 数据结构 |
| `src/include/rma/rma_proxy.h` | ~120 | Proxy 数据结构 |
