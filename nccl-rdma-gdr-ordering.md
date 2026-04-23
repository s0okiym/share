# NCCL RDMA 操作与 GPUDirect RDMA 数据有序性分析

## 一、GPUDirect RDMA (GDR) 数据乱序问题

### 1.1 乱序的本质

从 ConnectX-7 (CX7) 网卡通过 GPUDirect RDMA 将数据写入 GPU HBM，**数据确实可能出现乱序**。这里的"乱序"有两种层面：

**层面一：PCIe 事务层面的写入乱序（Relaxed Ordering）**

当 MR (Memory Region) 注册时启用了 `IBV_ACCESS_RELAXED_ORDERING` 标志，PCIe switch 对经过它的写入事务可以重排。一个连续的 RDMA Write 的不同 cache line 可能以不同于发起顺序到达 GPU HBM。

**层面二：GPU 可见性乱序（GDR Write 可见性问题）**

这是更根本的问题。当 NIC 通过 PCIe 将数据直接写入 GPU HBM 时，GPU 的 L2 cache 和 SM 的视角存在差异：
- NIC 的写入绕过了 GPU 的 L2 cache，直接落入 HBM
- GPU SM 可能先看到旧数据（L2 cache 中），而后才能看到 NIC 写入的新数据
- 多个 RDMA Write 之间，GPU 端的可见性顺序不保证与 NIC 发送顺序一致

### 1.2 乱序发生的场景

| 场景 | 乱序类型 | 原因 |
|------|---------|------|
| Ampere 及更早 GPU (sm_80 以下) | GPU 可见性乱序 | GPU L2 cache 与 HBM 之间没有一致性保证，NIC 写入 HBM 的数据对 GPU SM 不可见，需要显式 flush |
| C2C (Cache Coherent Interconnect) 平台 + NIC 经 PCIe switch | 路径不一致导致乱序 | 数据经 PCIe switch 到 GPU，而完成标志/flag 经 C2C 路径，两条路径到达顺序不一致 |
| 启用 IBV_ACCESS_RELAXED_ORDERING | PCIe 事务乱序 | PCIe switch 可以重排写入事务，导致同一 RDMA Write 的不同部分乱序到达 |
| 多 QP 并发传输 | 逻辑层面乱序 | 数据被分片到多个 QP，不同 QP 的数据到达顺序不确定 |
| Adaptive Routing (AR) | 网络层面乱序 | AR 会让不同包走不同路径，到达顺序与发送顺序不一致 |

### 1.3 NCCL 的解决方案

NCCL 在多个层面解决 GDR 数据乱序/可见性问题：

#### 方案一：RDMA READ Flush（核心机制）

**代码位置**: `src/transport/net_ib/p2p.cc:504-543` (`ncclIbIflush`)

这是解决 GPU 可见性乱序的关键机制。当接收端收到通过 GDR 写入 GPU HBM 的数据后，NCCL 会发起一次 RDMA READ 操作，从同一块 GPU 内存中读回 1 字节数据到主机内存。

```cpp
// p2p.cc:504-543 简化逻辑
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;  // 不需要flush则跳过

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;

  // 在所有设备上发起 RDMA READ
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    wr.opcode = IBV_WR_RDMA_READ;          // 使用 RDMA READ
    wr.wr.rdma.remote_addr = (uint64_t)data[last];  // 从目标GPU内存读取
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;  // 读到主机端 dummy buffer
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
  }
}
```

**原理**：RDMA READ 是一种请求-响应型操作，NIC 必须等数据从 GPU HBM 返回后才能完成 CQE。这个过程中，PCIe 的 read response 会强制将之前所有 pending 的 write 数据"推"到 GPU 可见。这利用了 PCIe 的排序规则：read completion 必须在之前的 write 完成之后才能返回。

**触发条件** (`connect.cc:1101-1102`):
```cpp
rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                          && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;
```
- GDR 或 DMA-BUF 可用 **且** `NCCL_GDR_FLUSH_DISABLE=0`（默认不禁用）
- 可通过 `NCCL_GDR_FLUSH_DISABLE=1` 关闭

**是否需要 flush 的拓扑判断** (`graph/paths.cc:515-535`):
```cpp
ncclResult_t ncclTopoNeedFlush(...) {
  *flush = 1;
  if (props.forceFlush == 1 || ncclParamNetForceFlush()) return ncclSuccess;
  // Hopper (sm_90) 及以上不需要 flush
  if (gpu->gpu.cudaCompCap >= 90) *flush = 0;
  // C2C 平台上数据经 PCIe 但完成标志经 C2C，必须 flush
  if (gpu->paths[NET][n].type <= PATH_PXB && gpu->paths[CPU][c].type == PATH_C2C) {
    *flush = 1;
  }
}
```

关键结论：
- **Ampere 及更早**：必须 flush，因为 L2 cache 一致性问题
- **Hopper (sm_90) 及以上**：默认不需要 flush，GPU 硬件保证了 GDR 写入的可见性
- **C2C 平台**：即使 Hopper 也需要 flush，因为数据路径和完成标志路径不一致

#### 方案二：Relaxed Ordering + 32 字节对齐

**代码位置**: `src/transport/net_ib/common.h:392-399`, `reg.cc:26-27`

当启用 PCIe Relaxed Ordering 时，NCCL 通过强制 CTS FIFO 结构体 32 字节对齐，确保单条 CTS 消息不会被拆分到多个 PCIe 事务中：

```cpp
// common.h:392-399
// CTS FIFO must not get split and written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0);
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0);
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0);
```

MR 注册时条件性启用 Relaxed Ordering (`reg.cc:26-27`):
```cpp
bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
```

Relaxed Ordering 可以提升 PCIe 带宽利用率，但需要额外的对齐约束来避免控制消息乱序。

#### 方案三：软件内存屏障

**代码位置**: `src/transport/net_ib/p2p.cc:282`

在接收端通知发送端之前，使用 C++ 顺序一致性内存屏障，确保 CTS FIFO 中的 tag/rkey/addr 加载不会被重排到 nreqs 加载之前：

```cpp
std::atomic_thread_fence(std::memory_order_seq_cst);
// order the nreqsPtr load against tag/rkey/addr loads below
```

#### 方案四：RDMA WRITE WITH IMM 保证完成通知

**代码位置**: `src/transport/net_ib/p2p.cc:154`

数据传输使用 RDMA WRITE（无信号完成），最后一个 WR 使用 `IBV_WR_RDMA_WRITE_WITH_IMM`（带信号完成和立即数）。这确保：
- 数据写入和完成通知的顺序：所有 RDMA WRITE 在 RDMA WRITE WITH IMM 之前完成
- 接收端通过 `IBV_WC_RECV_RDMA_WITH_IMM` CQE 确认传输完成

```cpp
lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
lastWr->send_flags = IBV_SEND_SIGNALED;   // 只有最后一个 WR 产生 CQE
```

在多 QP 分片场景下，数据先通过 RDMA WRITE 发送（可能乱序到达），最后通过 RDMA WRITE WITH IMM 发送完成信号（包含请求大小/ID），接收端据此知道数据已全部到达。

#### 方案五：GPU 端 fence 指令

**代码位置**: `src/device/common.h`

NCCL 的 CUDA kernel 使用 PTX fence 指令确保 GPU 端内存可见性：

- `fence.acquire.cta` — CTA (thread block) 内的获取屏障
- `fence.release.gpu` — GPU 级释放屏障
- `fence.acq_rel.sys` — 系统级获取-释放屏障（跨 GPU/PCIe）

这些 fence 确保在 GPU kernel 访问 RDMA 写入的数据之前，所有之前的内存操作对当前线程可见。

#### 方案六：GIN IPut + Atomic Signal

**代码位置**: `src/transport/net_ib/gin.cc:527-607`

GIN (GPU-Initiated Network) 模式下，数据 PUT 和信号通知使用链式 WR 发送：

```cpp
// 数据 PUT
wr[0].opcode = IBV_WR_RDMA_WRITE;
wr[0].next = &wr[1];  // 链接到信号 WR

// 信号通知 — 保证在数据写入完成后执行
wr[1].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
wr[1].send_flags = IBV_SEND_SIGNALED;
```

IB Verbs 保证链式 WR 按顺序执行，所以 atomic signal 一定在数据写入完成之后，接收端通过检查 signal 值即可确认数据就绪。

---

## 二、NCCL 中的 RDMA 操作类型

NCCL 的 IB 传输层使用了以下 RDMA 操作：

### 2.1 RDMA WRITE (`IBV_WR_RDMA_WRITE`)

**用途**：批量数据传输和 CTS 消息发送

**场景一：数据传输** (`p2p.cc:111`)
- 发送端将数据从本地 buffer 写到接收端的 GPU HBM buffer
- 通常不设置 `IBV_SEND_SIGNALED`，避免 CQ 风暴
- 数据按 QP 分片，每个 QP 负责一部分数据

```cpp
wr->opcode = IBV_WR_RDMA_WRITE;
wr->send_flags = 0;  // 不产生 CQE，减少开销
wr->wr.rdma.remote_addr = slots[r].addr;  // 接收端 GPU buffer 地址
```

**场景二：CTS 消息** (`p2p.cc:371`)
- 接收端通过 RDMA WRITE 将 Clear-to-Send 消息写入发送端的 FIFO
- 使用 `IBV_SEND_INLINE`（数据在 WR 中内联发送，无需注册 MR）
- 周期性设置 `IBV_SEND_SIGNALED` 以防止发送队列溢出

```cpp
wr.opcode = IBV_WR_RDMA_WRITE;
wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE
```

**场景三：GIN IPut** (`gin.cc:506, 569`)
- GPU 发起的 PUT 操作，将数据从本地 GPU 写到远端 GPU

### 2.2 RDMA WRITE WITH IMM (`IBV_WR_RDMA_WRITE_WITH_IMM`)

**用途**：数据传输的完成通知

**代码位置**: `p2p.cc:154`

每次 send 操作的最后一个 WR 使用此操作码：
- 携带 4 字节立即数（imm_data），传递请求 ID 或发送大小
- 必须设置 `IBV_SEND_SIGNALED`，确保产生 CQE
- 接收端收到 `IBV_WC_RECV_RDMA_WITH_IMM` 完成事件，确认数据已到达

```cpp
lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
lastWr->imm_data = htobe32(immData);  // 请求ID或发送大小
lastWr->send_flags = IBV_SEND_SIGNALED;
```

**Adaptive Routing 场景**：当启用 AR 时，先发 RDMA WRITE 传输数据（数据可能走不同路径乱序到达），再发 RDMA WRITE WITH IMM 通知完成。最后一条消息保证在之前所有数据之后到达。

### 2.3 RDMA READ (`IBV_WR_RDMA_READ`)

NCCL 中 RDMA READ 有两个用途：

**用途一：GDR Flush** (`p2p.cc:527`)

如上节所述，从 GPU HBM 读回 1 字节到主机内存，利用 PCIe 排序规则确保之前的写入对 GPU 可见：
```cpp
wr.opcode = IBV_WR_RDMA_READ;
wr.wr.rdma.remote_addr = (uint64_t)data[last];  // GPU 内存地址
wr.sg_list = &comm->devs[i].gpuFlush.sge;       // 主机端 dummy buffer
```

**用途二：Resiliency Probe** (`p2p_resiliency.cc:416`)

当检测到发送失败时，通过 RDMA READ 从远端读取完成记录，判断接收端是否已经收到了数据，避免不必要的重传：
```cpp
probeWr.opcode = IBV_WR_RDMA_READ;
probeWr.wr.rdma.remote_addr = remoteAddr;  // 远端完成记录地址
```

### 2.4 RDMA Atomic Fetch-and-Add (`IBV_WR_ATOMIC_FETCH_AND_ADD`)

**用途**：GIN 模式下的信号通知

**代码位置**: `gin.cc:587`

在 GIN IPut+Signal 操作中，原子操作作为链式 WR 的最后一个，保证在数据写入完成后执行：
```cpp
wr[1].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
wr[1].wr.atomic.remote_addr = (uint64_t)signalPtr;
wr[1].wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
wr[1].wr.atomic.rkey = signalRkey;
```

支持两种信号操作：
- `NCCL_NET_SIGNAL_OP_INC`：信号值加 1（用于计数）
- `NCCL_NET_SIGNAL_OP_ADD`：信号值加指定值

### 2.5 IB Send/Recv（非 RDMA）

**用途**：连接建立阶段的控制消息交换

在 `connect.cc` 中，QP 建立和元数据交换通过 TCP socket 完成（不是 IB Send/Recv）。而 QP 建立后，数据传输完全走 RDMA，不再使用 IB Send/Recv。

---

## 三、GDR 支持检测与内存注册

### 3.1 GDR 支持检测

NCCL 支持两种 GDR 内核模块：

**传统 peermem 模式** (`gdr.cc:16-30`):
- 检测 `/sys/kernel/mm/memory_peers/nv_mem/version` (nv_peer_mem)
- 检测 `/sys/module/nvidia_peermem/version` (nvidia_peermem)
- 通过 `ncclIbGdrSupport()` 判断

**DMA-BUF 模式** (`gdr.cc:46-84`):
- 内核 5.12+ 的标准 DMA-BUF 接口
- 通过 `ibv_reg_dmabuf_mr` 系统调用探测
- 通过 `ncclIbDmaBufSupport(dev)` 判断

### 3.2 内存注册

**代码位置**: `src/transport/net_ib/reg.cc`

NCCL 使用 MR Cache 避免重复注册：
- 按 IB 设备维护 LRU 缓存，按页对齐合并
- DMA-BUF 路径：`ibv_reg_dmabuf_mr` / `mlx5dv_reg_dmabuf_mr`
- 传统路径：`ibv_reg_mr` / `ibv_reg_mr_iova2`（支持 Relaxed Ordering）
- 注册权限：`IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC`

### 3.3 GDR 模式选择

**代码位置**: `src/graph/paths.cc:422-489`

NCCL 根据拓扑自动选择 GDR 模式：

| 模式 | 值 | 含义 |
|------|---|------|
| `ncclTopoGdrModeDisable` | 0 | 禁用 GDR，走主机 bounce buffer |
| `ncclTopoGdrModeDefault` | 1 | 标准 GDR，NIC 直接访问 GPU 内存 |
| `ncclTopoGdrModePci` | 2 | 强制 PCIe 映射（C2C 平台上避免走 C2C 路径） |

选择逻辑：
1. NIC 和 GPU 都必须支持 GDR
2. GPU 到 NIC 的拓扑距离必须在阈值内（默认 `PATH_PXB`，C2C 平台放宽到 `PATH_P2C`）
3. C2C 平台上如果 GPU-CPU 走 C2C 但 GPU-NIC 走 PCIe，强制使用 PCI 映射模式

---

## 四、数据流全景

### 4.1 标准 P2P 传输流程（GDR）

```
Sender GPU                    Sender CPU Proxy           Network            Receiver CPU Proxy          Receiver GPU
    |                              |                         |                      |                         |
    |--- data in GPU buffer ------>|                         |                      |                         |
    |                              |--- RDMA WRITE --------->|--- NIC DMA to ------>|--- RDMA WRITE --------->|
    |                              |   (to remote GPU HBM)   |   remote GPU HBM    |   (CTS to sender FIFO)  |
    |                              |                         |                      |                         |
    |                              |--- RDMA WRITE WITH IMM->|                      |<-- CQE (IBV_WC_RECV_    |
    |                              |   (completion signal)   |                      |    RDMA_WITH_IMM)       |
    |                              |                         |                      |                         |
    |                              |                         |                      |--- RDMA READ ---------->|
    |                              |                         |                      |   (FLUSH: read 1B from  |
    |                              |                         |                      |    GPU HBM to host)     |
    |                              |                         |                      |                         |
    |                              |                         |                      |<-- CQE (flush complete) |
    |                              |                         |                      |                         |
    |<-- kernel reads data --------|                         |                      |--- notify GPU ---------->|
    |   (data now visible to GPU)  |                         |                      |                         |
```

### 4.2 关键环境变量

| 环境变量 | 默认值 | 作用 |
|---------|--------|------|
| `NCCL_GDR_FLUSH_DISABLE` | 0 | 设为 1 禁用 GDR flush（仅在 Hopper+ 上安全） |
| `NCCL_NET_FORCE_FLUSH` | 0 | 设为 1 强制启用 flush（即使 Hopper） |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | 控制 PCIe Relaxed Ordering |
| `NCCL_NET_GDR_LEVEL` | - | 自定义 GDR 启用的拓扑距离阈值 |
| `NCCL_NET_GDR_C2C` | 1 | C2C 平台是否使用 GDR |
| `NCCL_GDRCOPY_ENABLE` | 0 | 是否使用 GDRCOPY 库 |
| `NCCL_GDRCOPY_SYNC_ENABLE` | 1 | 使用 GDRCOPY 同步 |
| `NCCL_IB_GDR_FLUSH_DISABLE` | 0 | IB 层 GDR flush 禁用 |

---

## 五、总结

**GDR 数据乱序是真实存在的问题**，主要发生在：
1. Ampere 及更早 GPU 上的 L2 cache 可见性问题
2. C2C 平台上的异构路径问题
3. PCIe Relaxed Ordering 带来的事务重排

**NCCL 的核心解决方案是 RDMA READ Flush**：通过发起一次 RDMA READ，利用 PCIe 排序规则（read completion 必须在之前的 write 完成后返回），确保所有 GDR 写入对 GPU 可见。Hopper (sm_90) 及以上架构因硬件保证可见性，默认不需要此 flush。

**NCCL 使用的 RDMA 操作共 4 种**：RDMA WRITE（数据传输 + CTS）、RDMA WRITE WITH IMM（完成通知）、RDMA READ（flush + resiliency probe）、Atomic Fetch-and-Add（GIN 信号）。加上连接阶段的 TCP socket 控制消息交换，构成了完整的通信协议。
