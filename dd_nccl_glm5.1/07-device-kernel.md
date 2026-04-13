# NCCL GPU 设备端内核架构

设备端内核是 NCCL 在 GPU 上执行的实际工作单元。从内核入口到协议原语再到算法实现，形成四层调度架构。

---

## 1. 四层调度架构

```mermaid
flowchart TD
    L1["第1层: 内核入口\nncclDevKernel_xxx(args4K)"]
    L2["第2层: 内核主循环\nncclKernelMain\n初始化 shmem → 调度循环"]
    L3["第3层: 工作调度\nRunWorkBatch → RunWorkColl\n按 funcId 分发"]
    L4["第4层: 算法实现\nrunRing / runTreeSplit / ...\n调用 Primitives 原语"]

    L1 --> L2 --> L3 --> L4
```

---

## 2. 内核入口与主循环

### 2.1 内核定义

特化内核 (由 `generate.py` 生成):
```c
DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId)
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K args4K) {
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args);
  }
```

通用内核:
```c
__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}
```

### 2.2 ncclKernelMain 流程

```mermaid
flowchart TD
    A["ncclKernelMain<specializedFnId, Fn>(&args)"] --> B["协作拷贝 args 到共享内存\nncclShmem.args = args"]

    B --> C["blockIdx → channelId 映射\n使用 channelMask 的 popcount"]

    C --> D["warp0: 加载 ncclKernelComm\n到共享内存"]
    D --> E["warp1: 加载 ncclDevChannel\n到共享内存"]
    E --> F["其余 warp: loadWorkBatchToShmem\n加载工作批次"]

    F --> G["工作批次调度循环\nwhile (!aborted)"]

    G --> H{funcId == SpecializedFnId?}
    H -->|"是"| I["直接执行\nFn::run() 即 RunWorkBatch::run()"]
    H -->|"否 (SpecializedFnId==-1)"| J["查函数表\ndevFuncTable[funcId]()"]

    I --> K["RunWorkBatch::run(tid, nthreads, work)"]
    J --> K

    K --> K1["遍历工作批次中的所有 work"]
    K1 --> K2["对每个 ncclDevWorkColl:\nRunWorkColl<...>::run(tid, nthreads, work)"]
    K2 --> K3{连续 work 有不同 nWarps?}
    K3 -->|"是"| K4["__syncthreads()"]
    K3 -->|"否"| K5["继续"]

    K4 --> L{nextJump 存在?}
    K5 --> L
    L -->|"是"| M["加载下一批工作\nloadWorkBatchToShmem"]
    L -->|"否"| N["内核完成"]
    M --> G
```

---

## 3. 工作批次加载

`loadWorkBatchToShmem` 从内核参数或工作 FIFO 加载工作描述：

```mermaid
flowchart TD
    A["loadWorkBatchToShmem"] --> B["解码 64-bit offsetBitset\n确定哪些 work 属于当前通道"]
    B --> C["从 kernel args (参数空间)\n或 work FIFO (全局内存)\n拷贝工作描述到共享内存"]
    C --> D["以 16 字节 pack 为单位加载"]
    D --> E["设置 ncclShmem.work\n指向已加载的工作描述"]
```

---

## 4. 协议原语层

### 4.1 Primitives 模板

```mermaid
classDiagram
    class Primitives_T_RedOp_Fan_Direct_Proto_P2p_ {
        +send(data, count)
        +recv(data, count)
        +recvReduceSend(data)
        +recvReduceCopy(data)
        +copySend(data)
        +recvCopySend(data)
        +recvReduceCopySend(data)
    }

    class FanSymmetric_N_ {
        +MAX_RECV: N
        +MAX_SEND: N
    }

    class FanAsymmetric_MaxRecv_MaxSend_ {
        +MAX_RECV: MaxRecv
        +MAX_SEND: MaxSend
    }

    Primitives_T_RedOp_Fan_Direct_Proto_P2p_ --> FanSymmetric_N_ : Fan=FanSymmetric (Ring)
    Primitives_T_RedOp_Fan_Direct_Proto_P2p_ --> FanAsymmetric_MaxRecv_MaxSend_ : Fan=FanAsymmetric (Tree)
```

**模板参数**:
- **T**: 数据类型 (int, float, half, ...)
- **RedOp**: 规约操作 (Sum, Prod, Min, Max, ...)
- **Fan**: FanSymmetric\<N\> (Ring) 或 FanAsymmetric\<MaxRecv, MaxSend\> (Tree)
- **Direct**: 是否启用直连 (P2P/NVLS 读写)
- **Proto**: ProtoLL / ProtoLL128 / ProtoSimple
- **P2p**: 是否 SendRecv 模式

### 4.2 三种协议对比

```mermaid
flowchart TD
    subgraph "ProtoLL — 低延迟"
        LL1["16字节 FIFO 行:\n8字节数据 + 2×4字节flag"]
        LL2["50% 带宽开销 (flag 占比)"]
        LL3["每元素 flag 检查 → 极低延迟"]
        LL4["MaxGroupWidth = 1"]
        LL5["适用: 小消息 (<8KB)"]
    end

    subgraph "ProtoLL128 — 中延迟"
        L1["128字节 cache line 同步"]
        L2["每16元素1个flag字 (1/8线程)"]
        L3["93.75% 数据效率"]
        L4["warp 级 load128/store128"]
        L5["适用: 中等消息 (8KB-256KB)"]
    end

    subgraph "ProtoSimple — 高带宽"
        S1["代理辅助通信\n大缓冲区, 无逐元素 flag"]
        S2["线程角色分工:\nWaitRecv/WaitSend/Worker/PostSend/PostRecv"]
        S3["支持 Direct Read/Write"]
        S4["支持 Scatter/Gather"]
        S5["适用: 大消息 (>256KB)"]
    end
```

### 4.3 LL 协议操作流程

```mermaid
flowchart TD
    A["LLGenericOp<RECV, SEND, SrcBuf, DstBuf>"] --> B["waitSend — 自旋等待\nsendConnHeadPtr 有空位\n(最多 NCCL_STEPS 未完成)"]

    B --> C["数据循环 (per-thread, per-element)"]
    C --> D["DataLoader 加载本地数据\n(处理 sub-4-byte 类型非对齐)"]
    D --> E["readLL — 从 LL FIFO 读取远端数据\n自旋等待 flag 匹配\ndata1|flag1|data2|flag2"]

    E --> F["applyPreOp — 本地预处理 (按需)"]
    F --> G["applyReduce — 规约 (按需)\n本地数据 + 远端数据 → 结果"]
    G --> H["applyPostOp — 后处理 (按需)"]

    H --> I["storeLL — 写入发送 FIFO\n设置 flag 标记数据有效"]
    I --> J["存储结果到用户输出缓冲区"]

    J --> K{还有数据?}
    K -->|"是"| C
    K -->|"否"| L["步进推进:\n递增 recv/send step"]
    L --> M["更新 head/tail 指针\n(通知对端)"]
```

### 4.4 LL128 协议操作流程

```mermaid
flowchart TD
    A["recvReduceSendCopy\n(LL128 GenericOp)"] --> B["Wait first recv:\n所有线程协作 load128\nflag 线程 (tid%8==7) 检查 flag"]

    B --> C["__any_sync warp 投票\n任何 flag 线程不匹配 → 重载"]
    C --> D["Finish register load:\nflag 线程 shuffle 数据到正确位置"]

    D --> E["Recv from remaining peers:\n迭代额外 recv 连接\napplyReduce 规约"]

    E --> F["Send:\n写入所有 send 连接\n替换 flag 线程位置的 flag 值"]

    F --> G["Store to destination:\n写入用户输出缓冲区\n使用共享内存暂存非对齐写入"]
```

### 4.5 Simple 协议操作流程

```mermaid
flowchart TD
    A["genericOp (Simple)"] --> B["外层: chunk 循环"]
    B --> C["内层: slice 循环 (SlicePerChunk)"]

    C --> D["线程角色分工"]
    D --> D1["WaitRecv 线程:\n自旋等待 connStepPtr\n设置 srcs/dsts 到 FIFO 或 direct buffer"]
    D --> D2["WaitSend 线程:\n自旋等待发送缓冲区空位\n设置 srcs/dsts"]
    D --> D3["Worker 线程:\n(总数 = nthreads - (有send? WARP_SIZE : 0))"]

    D1 --> E["subBarrier\n同步 worker 与 wait 线程"]
    D2 --> E
    D3 --> E

    E --> F["Worker: reduceCopy<Unroll>\n从 srcs[] 读取 + 规约\n写入 dsts[]"]

    F --> G["barrier\n同步 worker 与 post 线程"]

    G --> H["PostSend 线程:\nst_relaxed_sys_global\n推进发送步进计数"]
    H --> I["PostRecv 线程:\n推进接收步进计数"]

    I --> J{更多 slice?}
    J -->|"是"| C
    J -->|"否"| K{更多 chunk?}
    K -->|"是"| B
    K -->|"否"| L["完成"]
```

---

## 5. 算法实现

### 5.1 Ring AllReduce

```mermaid
flowchart TD
    A["runRing<T, RedOp, Proto>"] --> B["创建 Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>\nring->prev, ring->next"]

    B --> C["Phase 1: Reduce-Scatter\n(nRanks-1 步)"]
    C --> C1["step 0..nRanks-2:\nrecv(from prev) + reduce + send(to next)"]
    C1 --> C2["最后一步:\ndirectRecvReduceCopyDirectSend\n+ postOp (最终规约)"]

    C2 --> D["Phase 2: All-Gather\n(nRanks-1 步)"]
    D --> D1["step 0..nRanks-2:\nrecv(from prev) + copySend(to next)"]
    D1 --> D2["最后一个 chunk:\ndirectRecv (直接接收)"]
```

### 5.2 Tree AllReduce

```mermaid
flowchart TD
    A["runTreeSplit"] --> B["将线程分为两组"]

    B --> C["组1: Reduce-Up (叶→根)"]
    C --> C1["叶节点: send up\nPrimitives<FanAsymmetric<1,3>>::send"]
    C1 --> C2["中间节点: recv from children + reduce + send to parent"]
    C2 --> C3["根节点: recv + reduce"]

    B --> D["组2: Broadcast-Down (根→叶)"]
    D --> D1["根节点: send down\nPrimitives<FanAsymmetric<3,1>>::send"]
    D1 --> D2["中间节点: recv from parent + forward to children"]
    D2 --> D3["叶节点: recv"]

    C3 --> D1
```

### 5.3 NVLS AllReduce

```mermaid
flowchart TD
    A["runNvls AllReduce"] --> B["线程分区:\nscatter/gather/reduce+broadcast warps"]

    B --> C{单节点?}
    C -->|"是"| D["NVLS multicast 直接规约\nMultimemSrcs/MultimemDsts"]

    C -->|"否"| E["多节点路径"]
    E --> E1["scatter: 分发本地数据到 NVLink peers"]
    E1 --> E2["reduce: 累积 NVLink peers 数据\n→ 发送到网络"]
    E2 --> E3["network: 接收规约结果"]

    D --> F["NVLS gather + broadcast"]
    E3 --> F
    F --> F1["gather: 收集 NVLink peers 数据"]
    F1 --> F2["broadcast: 从网络接收\n→ 分发到 NVLink peers"]
```

### 5.4 CollNet Direct AllReduce

```mermaid
flowchart TD
    A["runTreeUpDown\n(CollNet Direct)"] --> B["线程分区:\nscatter/reduce/gather/broadcast"]

    B --> C["scatter: 分发本地数据到 NVLink peers"]
    C --> D["reduce: 累积 NVLink peers 数据\n→ 发送到 CollNet 网络"]
    D --> E["network: 接收网络规约结果"]
    E --> F["gather: 收集 NVLink peers 数据"]
    F --> G["broadcast: 从网络接收\n→ 分发到 NVLink peers"]
```

---

## 6. 内核代码生成 (generate.py)

### 6.1 生成流程

```mermaid
flowchart TD
    A["generate.py"] --> B["枚举所有有效组合\n(collective, redop, type, algo, proto)"]
    B --> B1["尊重算法约束:\nAllReduce: TREE/RING/COLLNET/NVLS\nBroadcast/Reduce: 仅 RING"]

    B1 --> C["equivalent_primary\n合并等价组合:\nsigned int ≡ unsigned int\n(规约内核相同)"]

    C --> D["best_kernel — 确定特化内核"]
    D --> D1["Nop → Generic 内核"]
    D1 --> D2["SendRecv → 独立内核"]
    D2 --> D3["非规约集合 → RING+LL 特化"]
    D3 --> D4["规约集合 → (Sum, 同类型, RING/TREE, LL)"]

    B --> E["生成输出文件"]
    E --> E1["device_table.cu\ndeclare ncclDevFunc_* + ncclDevFuncTable[]"]
    E --> E2["host_table.cc\nncclDevFuncRowToId[], ncclDevKernelList[]\nncclDevKernelForFunc[], ncclDevKernelForFuncIsSpecialized[]"]
    E --> E3["per-collective .cu 文件\nDEFINE_ncclDevKernel (特化)\nDEFINE_ncclDevFunc (函数表)"]
```

### 6.2 调度路径

```mermaid
flowchart TD
    A["内核启动\ncuLaunchKernelEx(ncclDevKernel_xxx, args4K)"] --> B{specializedFnId == funcId?}

    B -->|"是: 快路径"| C["直接执行 RunWorkBatch<coll,ty,redop,algo,proto>::run()\n编译时完全特化, 零间接开销"]

    B -->|"否: 慢路径"| D["ncclDevFuncTable[funcId]()\n运行时查表, 间接调用\n但仍有模板特化"]
```

### 6.3 ONLY_FUNCS 过滤

`ONLY_FUNCS` 环境变量允许开发时只编译指定函数，减少二进制大小。

---

## 7. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/device/common.h` | ~500 | 内核入口定义、ncclKernelMain、RunWorkBatch/Coll |
| `src/device/common.cu` | ~50 | Generic 内核定义 |
| `src/device/primitives.h` | ~200 | Primitives 模板声明 |
| `src/device/prims_ll.h` | ~300 | LL 协议原语 |
| `src/device/prims_ll128.h` | ~300 | LL128 协议原语 |
| `src/device/prims_simple.h` | ~350 | Simple 协议原语 |
| `src/device/all_reduce.h` | ~500 | AllReduce 算法 (Ring/Tree/NVLS/CollNet) |
| `src/device/all_gather.h` | ~300 | AllGather 算法 |
| `src/device/reduce_scatter.h` | ~300 | ReduceScatter 算法 |
| `src/device/broadcast.h` | ~200 | Broadcast 算法 |
| `src/device/reduce.h` | ~200 | Reduce 算法 |
| `src/device/sendrecv.h` | ~200 | Send/Recv 实现 |
| `src/device/generate.py` | ~600 | 内核变体自动生成 |
