# NCCL 对称内存与 CE 集合操作

对称内存 (Symmetric Memory) 是 NCCL 的高级内存抽象，为 CE 集合操作和对称内核提供 LSA (Local Symmetric Address) 团队和大 VA 空间支持，使 GPU 内核能够直接计算远端 rank 的缓冲区地址。

---

## 1. 对称内存运行时 (devr)

### 1.1 核心概念

```mermaid
flowchart TD
    subgraph "LSA (Local Symmetric Address) 团队"
        L1["lsaSelf: 本 rank 在团队中的索引"]
        L2["lsaSize: 团队大小"]
        L3["lsaRankList: 团队中的 rank 列表"]
        L4["nLsaTeams: LSA 团队数量"]
        L5["条件: rank 必须连续\n虚拟地址布局必须相同\n否则退化为 singleton"]
    end

    subgraph "Big VA Space (128GB per rank)"
        B1["bigSpace: 128GB 虚拟地址空间"]
        B2["bigSize: 每个 rank 的 VA 空间大小"]
        B3["lsaFlatBase: 所有 LSA rank 的 VA 拼接\nrank0 的 bigSpace + rank1 的 + ..."]
    end

    L3 --> B3

    subgraph "远端指针计算"
        P1["ncclDevrGetLsaRankPtr:\npeerPtr = lsaFlatBase\n+ lsaPeer * bigSize\n+ offset"]
        P2["ncclDevrGetLsaTeamPtrMC:\nmulticastPtr (NVLS)"]
    end

    B3 --> P1
    B3 --> P2
```

### 1.2 窗口注册

```mermaid
flowchart TD
    A["ncclDevrWindowRegister\n(userPtr, size, winFlags)"] --> B["在 ncclDevrState 中注册\n计算 bigOffset\n创建 localRegHandle"]
    B --> C["添加到 winSorted[]\n(按 bigOffset 排序)"]

    D["ncclDevrFindWindow\n(userPtr)"] --> E["在 winSorted[] 中\n二分查找包含 userPtr 的窗口"]
    E --> F["返回 ncclDevrWindow*\n或 NULL"]
```

### 1.3 窗口注册类型

| 类型 | 值 | 含义 |
|------|---|------|
| `ncclSymSendNonregRecvNonreg` | 0 | 发送/接收都未注册 |
| `ncclSymSendNonregRecvReg` | 1 | 仅接收端注册 (CE 可用) |
| `ncclSymSendRegRecvNonreg` | 2 | 仅发送端注册 |
| `ncclSymSendRegRecvReg` | 3 | 双端注册 (对称内核可用) |

---

## 2. CE 集合操作

### 2.1 支持条件

| 条件 | 要求 |
|------|------|
| CUDA 版本 | >= 12.5 |
| 节点范围 | 仅单节点 |
| 对称内存 | 必须启用 (comm->symmetricSupport) |
| 缓冲区注册 | 发送端和接收端必须通过对称窗口注册 |
| 支持的集合 | AllGather, AlltoAll, Scatter, Gather |
| 不支持 | 规约操作 (Reduce, AllReduce 等) |

### 2.2 同步协议

```mermaid
flowchart TD
    subgraph "CE 初始化 (ncclCeInit)"
        I1["分配对称内存:\nalignUp(nRanks*sizeof(uint32_t), 16) * 2"]
        I2["前半: Ready 数组\n每 rank 一个 uint32_t"]
        I3["后半: Complete 数组\n每 rank 一个 uint32_t"]
        I4["注册为 NCCL_WIN_COLL_SYMMETRIC\n(所有 rank 可远端寻址)"]
    end

    subgraph "同步路径"
        S1{NVLS 可用?}
        S1 -->|"是"| S2["MC 同步 (ncclPrepMCSync)\n写多播地址 → 传播到所有 rank\n1 次多播写 + nRanks-1 次 wait\n共 nRanks 次 mem op"]
        S1 -->|"否"| S3["UC 同步 (ncclPrepUCSync)\n逐个写每个远端 rank 的槽位\n2*(nRanks-1) 次 mem op"]
    end

    subgraph "序列号 (ceSeqNum)"
        N1["单调递增序列号"]
        N2["CUDA Graph 捕获时:\n使用常量 GRAPH_SYNC_VALUE=1\n完成后重置 flag 为 0"]
    end
```

### 2.3 CE AllGather 流程

```mermaid
flowchart TD
    A["CE AllGather\n(ncclCeAllGather)"] --> B["Pre-sync: ncclMemOpSync\n确保所有 rank 就绪"]

    B --> C["构建 batch ops\nncclCeBatchOpsParams"]
    C --> C1["对每个 peer rank:\nsrc = sendBuff + rank*chunkBytes\ndst = peerRecvBuff + rank*chunkBytes\n(通过 ncclDevrGetLsaRankPtr 计算)"]

    C1 --> D["ncclCeLaunchBatchOps\n执行所有拷贝"]
    D --> D1{CUDA 版本 + 场景?}
    D1 -->|"12.8+ 常规"| D2["cudaMemcpyBatchAsync\n+ PreferOverlapWithCompute"]
    D1 -->|"Graph 捕获 / 旧版"| D3["逐个 cudaMemcpyAsync"]
    D1 -->|"intraBatchSync 启用"| D4["分轮提交\n每轮 intraBatchSyncMsgThreshold/numOps"]

    D2 --> E["Post-sync: ncclMemOpSync\n确保所有传输完成"]
    D3 --> E
    D4 --> E
```

### 2.4 其他 CE 集合

| 集合 | 数据流 |
|------|--------|
| **AlltoAll** | src[dstRank] → peerDst[myRank] |
| **Scatter** | 仅 root: rootSend[peer] → peerRecv |
| **Gather** | 仅 root: peerSend → rootRecv[peer] |

### 2.5 Intra-Batch 同步

大规模场景下 (numOps > intraBatchSyncFreq=8, 总量 > 512MB)，启用批内同步：

```mermaid
flowchart TD
    A["大传输 (nOps > 8, bytes > 512MB)"] --> B["分轮提交"]
    B --> C["每轮: intraBatchSyncMsgThreshold / numOps 字节"]
    C --> D["提交一轮 batch memcpy"]
    D --> E["同步: 确保前面的操作完成"]
    E --> F{还有数据?}
    F -->|"是"| C
    F -->|"否"| G["完成"]
```

---

## 3. 对称内核

### 3.1 内核 ID 与选择

| 集合操作 | 内核 ID | 说明 |
|---------|---------|------|
| AllReduce | AGxLL_R, AGxLLMC_R | AllGather(LL) + Reduce(LL/MC) |
| AllReduce | RSxLD_AGxST, RSxLDMC_AGxSTMC | ReduceScatter(LD) + AllGather(ST/MC) |
| AllGather | LL, LLMC, ST, STMC | 按协议和是否多播分类 |
| AllGather | RailRing_LsaSTMC | 多轨 Ring + LSA ST MC |
| ReduceScatter | LL, LD, LDMC | 按协议和是否多播分类 |
| ReduceScatter | RailA2A_LsaLD, RailA2A_LsaLDMC | 多轨 AlltoAll + LSA |

`_MC` 后缀 = 多播 (NVLS) 变体。`RailRing`/`RailA2A` = GIN 多轨算法。

### 3.2 任务调度

```mermaid
flowchart TD
    A["ncclMakeSymmetricTaskList"] --> B["对每个集合任务:\n检查 ncclSymkAvailable"]
    B --> C["查找 send/recv 对称窗口\nncclDevrFindWindow"]
    C --> D["确定 winRegType"]
    D --> E["按 (func, redOp, datatype, winRegType) 分组"]
    E --> F["累积任务到 workArgsBytes 预算"]
    F --> G["ncclSymkPickKernel\n选择最优内核"]
    G --> H{回退条件?}

    H -->|"LL 内核 + 缓冲区未注册\n+ 多 GPU 进程"| I["回退到传统内核\n(对称内核不适用)"]
    H -->|"代价模型选非 LL 协议"| I
    H -->|"否"| J["使用对称内核"]
```

### 3.3 工作分配

```mermaid
flowchart TD
    A["ncclSymmetricTaskScheduler"] --> B["计算总 cell 数\ncellSize = NCCL_SYM_KERNEL_CELL_SIZE = 1024 bytes"]
    B --> C["将 cells 分配到 nMaxChannels 个通道\nround-robin 方式"]
    C --> D["使用 16-bit 定点 fracHi\n追踪小数边界"]
    D --> E["支持工作融合:\n相邻任务在同通道可共享 block"]
    E --> F["构建 ncclSymkDevWorkArgs\n包含 device comm + 通道工作范围"]
    F --> G["存储为 plan->kernelSymArgs"]
```

### 3.4 设备端原语

| 原语 | 说明 |
|------|------|
| `ncclSymPtr<T>` | 对称指针: window + offset → localPtr() / multimemPtr() |
| `ncclLsaPointerGetter<T>` | 计算 per-LSA-rank 指针: lsaFlatBase + lsaPeer * stride4G |
| `bcastMultimem` | 多播广播: multimem_st_global 指令, 128B chunk 展开循环 |
| `ncclSymkSmemPartition` | 动态共享内存分区 |

---

## 4. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/ce_coll.cc` | ~700 | CE 集合操作实现 |
| `src/include/ce_coll.h` | ~60 | CE 数据结构 |
| `src/scheduler/symmetric_sched.cc` | ~200 | 对称内核任务调度 |
| `src/include/dev_runtime.h` | ~100 | 对称内存运行时 (ncclDevrState) |
| `src/include/sym_kernels.h` | ~120 | 对称内核状态 (ncclSymkState) |
| `src/device/symmetric/primitives.cuh` | ~200 | 对称内核设备端原语 |
