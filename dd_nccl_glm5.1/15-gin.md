# NCCL GIN GPU 发起网络

GIN (GPU-Initiated Networking) 使 GPU 能够直接发起网络操作 (RDMA put, signal)，无需 CPU 介入。通过插件架构支持两种后端：GDAKI (GPU 直连) 和 Proxy (CPU 辅助)。

---

## 1. 双后端架构

```mermaid
flowchart TD
    A["GIN 插件 (ncclGin_t)"] --> B{后端类型?}

    B -->|"NCCL_GIN_TYPE_GDAKI\n(GPU-Direct Accelerated\nKernel Interface)"| C["硬件加速路径\nGPU 直接编程 NIC"]
    C --> C1["仅 MLX5 IB 设备"]
    C1 --> C2["无需 CPU 进度线程"]
    C2 --> C3["GPU 直接发起 RDMA\n最低延迟"]

    B -->|"NCCL_GIN_TYPE_PROXY\n(CPU 辅助)"| D["CPU 代理路径\nGPU 入队 → CPU 提交"]
    D --> D1["任意支持 GDR 的 IB 设备"]
    D1 --> D2["需要 CPU 进度线程"]
    D2 --> D3["GPU 通过 GFD 队列\n提交操作描述符"]
```

---

## 2. GIN 状态与连接

### 2.1 核心数据结构

```mermaid
classDiagram
    class ncclGinState {
        +ncclGin: ncclGin_t* (插件 vtable)
        +ginType: int (GDAKI/PROXY)
        +ginComms: void*[] (集合通信 handle)
        +ginCtx: void*[] (GIN context handle)
        +ginDevHandles: void*[] (设备 handle)
        +ginCommCount: int
        +ginContextCount: int
        +signalSpace: ncclSpace (信号索引分配)
        +counterSpace: ncclSpace (计数器索引分配)
        +needsProxyProgress: int
        +ginProgress: int (0=paused,1=running,-1=terminate,-2=error)
        +thread: pthread_t
    }

    class ncclGin_t {
        +name: char*
        +init(dev, ginInstance)
        +getProperties(props)
        +listen(ginCtx, dev, handle, listenComm)
        +connect(ginCtx, dev, handle, sendComm)
        +accept(listenComm, recvComm)
        +allGather(ginCtx, sendData, recvData, size)
        +allToAll(ginCtx, sendData, recvData, size)
        +iput(comm, srcBuff, dstBuff, size, dstMhandle, request)
        +iputSignal(comm, srcBuff, dstBuff, size, dstMhandle, signalAddr, signalMhandle, request)
        +test(request, done, size)
        +createContext(ginInstance, dev, ginCtx, ctx)
        +regMrSym(comm, data, size, mhandle)
        +regMrSymDmaBuf(comm, data, size, fd, mhandle)
        +deregMrSym(comm, mhandle)
    }

    ncclGinState --> ncclGin_t : ncclGin
```

### 2.2 连接建立

```mermaid
flowchart TD
    A["ncclGinConnectOnce(comm)"] --> B["加载 GIN 插件\n检测后端类型"]

    B --> C["ncclTopoGetLocalGinDevs\n发现本地 GIN 设备"]
    C --> D["确定连接数\nNCCL_GIN_NCONNECTIONS 或默认设备数"]

    D --> E["创建监听端点\nbootstrapAllGather 交换 handle"]
    E --> F["ginComm->connect\n全互联集合通信"]

    F --> G{后端类型?}
    G -->|"Proxy"| H["ncclGinProxyCreateContext"]
    G -->|"GDAKI"| I["ncclGin->createContext"]
    H --> J{needsProxyProgress?}
    I --> K["完成 (无进度线程)"]
    J -->|"是"| L["启动 ncclGinProxyProgressThread"]
    J -->|"否"| K
```

---

## 3. GIN Proxy 后端

### 3.1 Proxy Context 结构

```mermaid
classDiagram
    class ginProxyCtx {
        +comm: ncclComm*
        +collComm: void* (集合通信 handle)
        +devHandle: void* (设备 handle, needsProxyProgress=1)
        +hostGpuCtx: ginProxyHostGpuCtx[]
        +counters: uint64_t* (CPU-accessible)
        +countersDev: uint64_t* (GDR-accessible)
        +signalsDev: uint64_t* (GPU-side signal memory)
        +hasError: int
    }

    class ginProxyHostGpuCtx {
        +queues: void* (GFD queue, nRanks*queueSize)
        +cis: uint64_t[] (Consumed Indices, GPU-visible)
        +cisShadow: uint64_t[] (Host shadow copy)
        +sis: uint64_t[] (Seen Indices, host-side)
        +states: ginProxyGfdState[]
        +inlines: void* (inline data buffer)
        +contextId: int
        +queueSize: int (power of 2)
    }

    class ginProxyGfdState {
        +op: int (operation type)
        +counterId: int (completion counter)
        +done: int (completion flag)
        +request: void* (GIN plugin request)
    }

    ginProxyCtx --> ginProxyHostGpuCtx : hostGpuCtx[]
    ginProxyHostGpuCtx --> ginProxyGfdState : states[]
```

### 3.2 Proxy 操作流程

```mermaid
flowchart TD
    subgraph "GPU 端 (内核)"
        G1["GPU kernel 写入 GFD 队列\n(GPU Functional Descriptor)\n多 qword 描述符: header(op,size)\n+ srcOff/srcHandle\n+ dstOff/dstHandle\n+ completion(counterId,signalId)\n+ inline data"]
        G1 --> G2["原子递增 PI\n(Producer Index)"]
        G2 --> G3["写 readySeq[peer] = seq\n(CUDA batch mem op WRITE)"]
        G3 --> G4["等待 doneSeq[peer] >= seq\n(CUDA batch mem op WAIT)"]
    end

    subgraph "CPU 进度线程 (ncclGinProxyProgress)"
        C1["proxyGinPollGfd\n读取 GFD 队列\nflag 同步 + 重置队列槽"]
        C1 --> C2["proxyGinProcessGfd\n分发操作"]
        C2 --> C3{操作类型?}
        C3 -->|"ncclGinProxyOpPut"| C4["ginComm->iput()\n简单 RDMA 写"]
        C3 -->|"ncclGinProxyOpPut\n+ Signal"| C5["ginComm->iputSignal()\nRDMA 写 + 原子信号"]
        C3 -->|"ncclGinProxyOpVASignal"| C6["仅信号\n无数据搬运"]
        C4 --> C7["proxyGinPollCompletions\nginComm->test()"]
        C5 --> C7
        C6 --> C7
        C7 --> C8["完成:\n原子递增 CI\n(Consumed Index)\n更新 GPU 端 cis\nGPU 端 doneSeq 解除等待"]
    end

    G1 --> C1
    C8 --> G4
```

### 3.3 内存注册

```mermaid
flowchart TD
    A["ncclGinProxyRegMrSym\n(comm, data, size, mhandle)"] --> B{指针类型?}
    B -->|"CUDA 指针"| C["尝试 DMA-BUF:\ncuMemGetHandleForAddressRange\n→ 获取 FD\n→ regMrSymDmaBuf"]
    C --> C1{DMA-BUF 失败?}
    C1 -->|"是"| C2["回退: regMrSym\n(非 DMA-BUF)"]
    C1 -->|"否"| C3["DMA-BUF 注册成功"]
    B -->|"Host 指针"| D["regMrSym\n直接注册"]

    E["Signal 内存注册"] --> F["regMrSymDmaBuf\n+ NCCL_NET_MR_FLAG_FORCE_SO\n(强制强排序)"]
```

---

## 4. GIN GDAKI 后端

### 4.1 特点

```mermaid
flowchart TD
    A["GDAKI 后端"] --> B["仅 MLX5 IB 设备\n(IB_PROVIDER_MLX5)"]
    B --> C["需要 GDR 支持\nnv_peer_mem 或 DMA-BUF"]
    C --> D["GPU 直接编程 NIC\n无需 CPU 进度线程"]
    D --> E["ncclGinGdakiCreateContext\n直接创建 context"]
```

### 4.2 IB 集成 (gin.cc)

```mermaid
flowchart TD
    A["ncclGinIb — 自动选择后端"] --> B{NCCL_GIN_TYPE 环境变量?}
    B -->|"指定"| C["使用指定后端"]
    B -->|"未指定"| D["ncclGinIbInitType\n自动检测"]
    D --> D1{有 MLX5 设备 + GDR 支持?}
    D1 -->|"是"| D2["优先 GDAKI"]
    D1 -->|"否"| D3["使用 Proxy"]
```

### 4.3 集合通信

每个 GIN 连接使用全互联的 send/recv QP 对：

```mermaid
flowchart TD
    A["ncclGinIbCollComm"] --> B["fullSendComm[rank] — per-rank send QP"]
    A --> C["fullRecvComm[rank] — per-rank recv QP"]
    A --> D["allGather() — 软件集合操作\n基于 P2P QP"]
    A --> E["allToAll() — 软件集合操作"]

    F["连接建立"] --> G["ring-based bootstrapping\nconnect to next\naccept from previous"]
    G --> H["全互联 barrier\n建立所有 P2P QP"]
```

---

## 5. IB 层 Proxy 实现

### 5.1 RDMA Put (ncclGinIbProxyIPut)

```mermaid
flowchart TD
    A["ncclGinIbProxyIPut\n(comm, src, dst, size, dstMhandle, request)"] --> B["构建 IBV work request:\nIBV_WR_RDMA_WRITE\nremote_addr = baseVA + dst\nrkey = remote_key"]
    B --> C["ibv_post_send\n提交 RDMA 写"]
    C --> D["返回 request"]
```

### 5.2 RDMA Put + Signal (ncclGinIbProxyIPutSignal)

```mermaid
flowchart TD
    A["ncclGinIbProxyIPutSignal"] --> B["构建两条 work request 链"]

    B --> C["WR1: IBV_WR_RDMA_WRITE\n数据传输 (unsignaled)"]
    C --> D["WR2: IBV_WR_ATOMIC_FETCH_AND_ADD\n信号通知 (signaled)"]
    D --> D1["INC (加 1) 或 ADD (加任意值)"]
    D1 --> D2["使用 putSignalScratchpad\n作为原子操作的本地内存"]

    D2 --> E["ibv_post_send\n提交链式 WR"]
```

### 5.3 完成 Test (ncclGinIbProxyTest)

```mermaid
flowchart TD
    A["ncclGinIbProxyTest"] --> B["ibv_poll_cq(cq, 1, &wc)"]
    B --> C{有完成事件?}
    C -->|"是"| D{wc.status == IBV_WC_SUCCESS?}
    D -->|"是"| E["*done = 1"]
    D -->|"否"| F["报错"]
    C -->|"否"| G["*done = 0"]
```

---

## 6. 关键环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_GIN_PLUGIN` | GIN 插件库路径 |
| `NCCL_GIN_TYPE` | 强制后端类型 (GDAKI/PROXY) |
| `NCCL_GIN_NCONNECTIONS` | GIN 连接数 |

---

## 7. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/gin/gin_host.cc` | ~500 | GIN 连接管理、进度线程 |
| `src/gin/gin_host_proxy.cc` | ~600 | GIN Proxy 后端 |
| `src/transport/net_ib/gin.cc` | ~800 | IB 层 GIN 实现 (GDAKI/Proxy) |
| `src/include/plugin/nccl_gin.h` | ~100 | GIN 插件接口定义 |
