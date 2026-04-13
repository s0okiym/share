# NCCL GPU 设备端内核架构

设备端内核是 NCCL 在 GPU 上执行的实际工作单元。从内核入口到协议原语再到算法实现，形成四层调度架构。每一层都有明确的职责边界：内核入口管理执行上下文，主循环负责工作调度，工作批次执行具体集合操作，算法实现通过协议原语完成数据传输。

---

## 1. 四层调度架构

```mermaid
flowchart TD
    L1["第1层: 内核入口\nncclDevKernel_xxx(args4K)\n设置共享内存, 加载通道数据"]
    L2["第2层: 内核主循环\nncclKernelMain\n调度工作批次, 查函数表"]
    L3["第3层: 工作调度\nRunWorkBatch → RunWorkColl\n按 funcId 分发到具体算法"]
    L4["第4层: 算法实现\nrunRing / runTreeSplit / ...\n调用 Primitives 原语完成数据传输"]

    L1 --> L2 --> L3 --> L4
```

---

## 2. 内核入口与主循环

### 2.1 内核定义

NCCL 使用模板特化生成大量内核变体，每个变体对应一种 (集合操作, 数据类型, 规约操作, 算法, 协议) 组合：

```c
// 特化内核 (由 generate.py 生成)
DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId)
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K args4K) {
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args);
  }

// 通用内核 (处理非特化的 funcId)
__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}
```

特化内核的优势：编译时完全确定所有模板参数，编译器可以进行激进的内联和优化，消除虚函数调用开销。通用内核通过函数表间接调用，灵活性更高但性能略低。

### 2.2 ncclKernelMain 流程

```mermaid
flowchart TD
    A["ncclKernelMain(specializedFnId, Fn, &args)"] --> B["协作拷贝 args 到共享内存\n所有线程协作拷贝, 利用带宽"]

    B --> C["blockIdx → channelId 映射\n使用 channelMask 的 popcount\n跳过未使用的通道"]

    C --> D["warp0: 加载 ncclKernelComm\n到共享内存"]
    D --> E["warp1: 加载 ncclDevChannel\n到共享内存"]
    E --> F["其余 warp: loadWorkBatchToShmem\n加载工作批次描述"]

    F --> G["工作批次调度循环\nwhile !aborted"]

    G --> H{"funcId == SpecializedFnId?"}
    H -->|"是: 快路径"| I["直接执行\nFn::run() 即 RunWorkBatch::run()\n零间接开销"]
    H -->|"否: SpecializedFnId==-1"| J["查函数表\ndevFuncTable[funcId]()\n运行时分发"]

    I --> K["RunWorkBatch::run(tid, nthreads, work)"]
    J --> K

    K --> K1["遍历工作批次中的所有 work"]
    K1 --> K2["对每个 ncclDevWorkColl:\nRunWorkColl::run(tid, nthreads, work)"]
    K2 --> K3{"连续 work 有不同 nWarps?"}
    K3 -->|"是"| K4["__syncthreads() 同步\n防止共享内存竞争"]
    K3 -->|"否"| K5["继续"]

    K4 --> L{"nextJump 存在?\n有下一批工作?"}
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
    A["loadWorkBatchToShmem"] --> B["解码 64-bit offsetBitset\n确定哪些 work 属于当前通道\n每个 bit 对应一个 work item"]
    B --> C{"workStorageType?"}
    C -->|"Args"| D["从内核参数空间加载\n工作数据嵌入在 kernelArgs 中"]
    C -->|"Fifo"| E["从 Work FIFO 加载\n全局内存循环缓冲区\n需要等待 FIFO 数据就绪"]
    C -->|"Persistent"| F["从持久化缓冲区加载\nCUDA Graph 固定位置"]
    D --> G["以 16 字节 pack 为单位\n协作拷贝到共享内存"]
    E --> G
    F --> G
    G --> H["设置 ncclShmem.work\n指向已加载的工作描述"]
```

工作批次加载使用共享内存暂存工作描述，这比直接从全局内存读取快得多。所有线程协作拷贝（每个线程拷贝一部分），充分利用全局内存带宽。

---

## 4. 协议原语层

Primitives 是 NCCL 设备端代码的核心抽象。每个算法（Ring、Tree、NVLS 等）通过调用 Primitives 的 send/recv/reduce 等原语来完成数据传输，而不需要关心底层协议细节。

### 4.1 三种协议对比

| 特性 | LL (Low Latency) | LL128 | Simple |
|------|-----------------|-------|--------|
| 同步机制 | 每 8B 数据 + 4B flag | 每 16 元素 1 个 flag 字 | 无逐元素 flag，依赖步进计数 |
| 数据效率 | 50% | 93.75% | ~100% |
| 延迟 | 最低 | 中等 | 最高 |
| 适用场景 | 小消息 (<8KB) | 中等消息 (8KB-256KB) | 大消息 (>256KB) |
| 传输方式 | GPU 直连 P2P | GPU 直连 P2P | 代理辅助 (NET/SHM) |

### 4.2 LL 协议操作流程

LL 协议使用内联 flag 实现低延迟同步。每个 FIFO 行为 16 字节：8 字节数据 + 两个 4 字节 flag。

```mermaid
flowchart TD
    A["LLGenericOp"] --> B["waitSend — 自旋等待\nsendConnHeadPtr 有空位\n最多 NCCL_STEPS 步未完成"]

    B --> C["数据循环: per-thread, per-element"]
    C --> D["DataLoader 加载本地数据\n处理 sub-4-byte 类型非对齐"]
    D --> E["readLL — 从 LL FIFO 读取远端数据\n自旋等待 flag 匹配\ndata1|flag1|data2|flag2"]

    E --> F["applyPreOp — 本地预处理"]
    F --> G["applyReduce — 规约\n本地数据 + 远端数据 → 结果"]
    G --> H["applyPostOp — 后处理"]

    H --> I["storeLL — 写入发送 FIFO\n设置 flag 标记数据有效"]
    I --> J["存储结果到用户输出缓冲区"]

    J --> K{"还有数据?"}
    K -->|"是"| C
    K -->|"否"| L["步进推进:\n递增 recv/send step\n更新 head/tail 指针"]
```

LL 的 flag 检查是每元素级别，开销很大（50% 带宽浪费），但延迟极低——数据一到达就能被消费，无需等待整个 slice 完成。

### 4.3 LL128 协议操作流程

LL128 使用 128 字节 cache line 同步，每 16 个元素只有 1 个 flag 字（1/8 线程负责检查 flag）。

```mermaid
flowchart TD
    A["LL128 GenericOp"] --> B["Wait first recv:\n所有线程协作 load128\nflag 线程 tid%8==7 检查 flag"]

    B --> C["__any_sync warp 投票\n任何 flag 线程不匹配 → 重载"]
    C --> D["Finish register load:\nflag 线程 shuffle 数据到正确位置"]

    D --> E["Recv from remaining peers:\n迭代额外 recv 连接\napplyReduce 规约"]

    E --> F["Send:\n写入所有 send 连接\n替换 flag 线程位置的 flag 值"]

    F --> G["Store to destination:\n写入用户输出缓冲区\n使用共享内存暂存非对齐写入"]
```

LL128 相比 LL 的关键改进：flag 开销从 50% 降到 6.25%，同时保持较低的延迟。利用了 GPU 的 128 字节 cache line 特性，一次 load128 指令加载整行数据。

### 4.4 Simple 协议操作流程

Simple 协议是大消息的主力协议，采用线程角色分工模式：

```mermaid
flowchart TD
    A["genericOp (Simple)"] --> B["外层: chunk 循环"]
    B --> C["内层: slice 循环"]

    C --> D["线程角色分工"]
    D --> D1["WaitRecv 线程:\n自旋等待 connStepPtr\n设置 srcs/dsts"]
    D --> D2["WaitSend 线程:\n自旋等待发送缓冲区空位\n设置 srcs/dsts"]
    D --> D3["Worker 线程:\n执行数据搬运和规约\n总数 = nthreads - 有send时减WARP_SIZE"]

    D1 --> E["subBarrier\n同步 worker 与 wait 线程"]
    D2 --> E
    D3 --> E

    E --> F["Worker: reduceCopy\n从 srcs[] 读取 + 规约\n写入 dsts[]"]

    F --> G["barrier\n同步 worker 与 post 线程"]

    G --> H["PostSend 线程:\nst_relaxed_sys_global\n推进发送步进计数\n通知接收端数据已就绪"]
    H --> I["PostRecv 线程:\n推进接收步进计数"]

    I --> J{"更多 slice?"}
    J -->|"是"| C
    J -->|"否"| K{"更多 chunk?"}
    K -->|"是"| B
    K -->|"否"| L["完成"]
```

Simple 协议的关键设计：线程分为 Wait/Worker/Post 三种角色。Wait 线程自旋等待数据就绪（不消耗计算资源），Worker 线程执行实际的数据搬运和规约，Post 线程在 Worker 完成后推进步进计数。这种分工让每种角色都能高效执行，避免条件分支的浪费。

---

## 5. 算法实现

### 5.1 Ring AllReduce

Ring AllReduce 分为 Reduce-Scatter 和 All-Gather 两个阶段：

```mermaid
flowchart TD
    A["runRing with Proto"] --> B["创建 Primitives\nFanSymmetric: 1 recv + 1 send\nring->prev, ring->next"]

    B --> C["Phase 1: Reduce-Scatter\nnRanks-1 步"]
    C --> C1["每步: recv from prev\n+ reduce + send to next\n数据沿 ring 流动并逐步规约"]
    C1 --> C2["最后一步:\ndirectRecvReduceCopyDirectSend\n+ postOp 执行最终规约"]

    C2 --> D["Phase 2: All-Gather\nnRanks-1 步"]
    D --> D1["每步: recv from prev\n+ copySend to next\n已规约的数据沿 ring 传播"]
    D1 --> D2["最终每个 rank 拥有完整结果"]
```

Ring 算法在 nRanks-1 步中完成每个阶段，总数据传输量 = nBytes * (nRanks-1) / nRanks（每步只传输 1/nRanks 的数据）。在 NVLink 互连下，Ring 可以充分利用双向带宽。

### 5.2 Tree AllReduce

Tree AllReduce 使用双二叉树，将线程分为两组并行执行 reduce 和 broadcast：

```mermaid
flowchart TD
    A["runTreeSplit"] --> B["将线程分为两组"]

    B --> C["组1: Reduce-Up 叶→根"]
    C --> C1["叶节点: send up\nFanAsymmetric: 1 recv, 3 send"]
    C1 --> C2["中间节点: recv from children\n+ reduce + send to parent"]
    C2 --> C3["根节点: recv + reduce"]

    B --> D["组2: Broadcast-Down 根→叶"]
    D --> D1["根节点: send down\nFanAsymmetric: 3 recv, 1 send"]
    D1 --> D2["中间节点: recv from parent\n+ forward to children"]
    D2 --> D3["叶节点: recv"]

    C3 --> D1
```

双树的关键优势：reduce 和 broadcast 可以流水线执行——当一个树的 reduce 阶段完成一部分时，另一个树就可以开始 broadcast，减少总体延迟。

### 5.3 NVLS AllReduce

NVLS 利用 NVSwitch 的硬件多播/归约能力：

```mermaid
flowchart TD
    A["runNvls AllReduce"] --> B["线程分区:\nscatter/gather/reduce+broadcast warps"]

    B --> C{"单节点?"}
    C -->|"是"| D["NVLS multicast 直接规约\nMultimemSrcs/MultimemDsts\n硬件自动归约"]

    C -->|"否"| E["多节点路径"]
    E --> E1["scatter: 分发本地数据到 NVLink peers"]
    E1 --> E2["reduce: 累积 NVLink peers 数据\n→ 发送到网络"]
    E2 --> E3["network: 接收规约结果"]

    D --> F["NVLS gather + broadcast"]
    E3 --> F
    F --> F1["gather: 收集 NVLink peers 数据"]
    F1 --> F2["broadcast: 从网络接收\n→ 分发到 NVLink peers"]
```

NVLS 的核心优势：NVSwitch 硬件可以在多播写入时自动执行归约操作，减少了 GPU 侧的计算量。

---

## 6. 内核代码生成 (generate.py)

NCCL 使用 Python 脚本自动生成内核变体，避免手动编写大量模板实例化代码。

### 6.1 生成策略

generate.py 枚举所有有效的 (集合操作, 数据类型, 规约操作, 算法, 协议) 组合，为每个组合生成特化内核。等价组合（如 signed/unsigned int 使用相同的规约内核）会被合并以减少二进制大小。

### 6.2 调度路径

```mermaid
flowchart TD
    A["内核启动\ncuLaunchKernelEx"] --> B{"specializedFnId == funcId?"}

    B -->|"是: 快路径"| C["直接执行 RunWorkBatch::run()\n编译时完全特化\n零间接开销"]
    B -->|"否: 慢路径"| D["ncclDevFuncTable[funcId]()\n运行时查表, 间接调用\n但仍有模板特化"]
```

### 6.3 生成输出

| 输出文件 | 内容 |
|---------|------|
| `device_table.cu` | ncclDevFunc_* 函数定义 + ncclDevFuncTable[] 函数表 |
| `host_table.cc` | ncclDevFuncRowToId[], ncclDevKernelList[] 等主机端查找表 |
| `per-collective .cu` | DEFINE_ncclDevKernel (特化内核) + DEFINE_ncclDevFunc (函数表条目) |

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
