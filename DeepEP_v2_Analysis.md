# DeepEP v2 架构与优化深度分析

## 概述

DeepEP v2 是一次近乎重写的版本（104 个文件变更，+14299/-4131 行代码），在 v1 的基础上进行了根本性的架构升级，带来了显著的性能提升和 SM 利用率的大幅降低。本文从代码层面深入分析 v2 相对 v1 的每一项核心优化。

---

## 一、架构级变革

### 1.1 JIT 编译取代 AOT 编译

**v1 方案**：所有 CUDA kernel 在 `setup.py build` 时通过 PyTorch `cpp_extension` 预编译（AOT）。C++ 模板参数（如 `kNumRanks`, `kHidden`, `kNumTopk` 等）在编译期固定，导致必须为不同参数组合预编译多个 kernel 变体，或者采用运行时分支（牺牲优化机会）。

**v2 方案**：引入完整的 JIT（Just-In-Time）编译系统（`csrc/jit/`），在运行时根据实际参数生成特化 kernel 代码，再通过 NVCC 编译为 CUBIN 并缓存。

核心流程（`csrc/jit/compiler.hpp`）：
```cpp
// 1. 根据运行时参数生成特化的 kernel 实例化代码
std::string code = CombineRuntime::generate(args);
// generate_impl() 拼接出类似：
//   #include <deep_ep/impls/combine.cuh>
//   static void __instantiate_kernel() {
//       auto ptr = reinterpret_cast<void*>(&combine_impl<true, false, true, 2, 8, 256, ...>);
//   }

// 2. 通过 NVCC 编译为 CUBIN
const auto runtime = jit::compiler->build("combine", code);

// 3. 加载并启动
CombineRuntime::launch(runtime, args, stream);
```

JIT 编译的关键优势：
- **全量编译期常量传播**：所有模板参数（`kNumSMs`, `kNumScaleoutRanks`, `kNumScaleupRanks`, `kHidden`, `kNumMaxTokensPerRank`, `kNumExperts`, `kNumTopk` 等）在编译期完全确定，编译器可以执行完整的常量折叠、循环展开和死代码消除
- **无需预编译变体**：不同集群配置（如 4节点8卡 vs 2节点16卡）无需重新安装，运行时自动适配
- **缓存机制**（`csrc/jit/cache.hpp`）：基于内容哈希（`get_hex_digest(kernel_signature)`）的缓存目录（`~/.deep_ep/cache/`），二次运行零编译开销
- **分布式文件系统安全**：通过临时目录 + 原子重命名（`std::filesystem::rename`）+ `fsync` 确保多进程并发编译的安全性

### 1.2 NCCL GIN 取代 NVSHMEM

**v1 方案**：使用 NVSHMEM 作为 RDMA 通信后端，依赖 `nvshmem_put`/`nvshmem_get`/`nvshmem_barrier_all` 等原语。需要独立安装 NVSHMEM 库、打补丁、环境配置复杂。

**v2 方案**：完全迁移到 NCCL 的 GIN（GPU InfiniBand Networking）API（`csrc/kernels/backend/nccl.cu`）。

NCCL GIN 的核心抽象（`deep_ep/include/deep_ep/common/handle.cuh`）：
```cpp
struct NCCLGin {
    const ncclDevComm_t& nccl_dev_comm;
    const ncclWindow_t& nccl_window;
    ncclGin gin;
    ncclTeam team_world, team_lsa, team_rail;  // 三种通信域

    // RDMA 写入（put）
    template <typename team_t>
    void put(recv_sym_ptr, send_sym_ptr, num_bytes, dst_rank_idx, ...);

    // 带远程原子操作的 RDMA 写入（red_add_rel）
    template <typename team_t>
    void red_add_rel(sym_ptr, value, dst_rank_idx, ...);

    // RDMA 读取（get）
    template <typename team_t>
    void get(src_ptr, dst_ptr, num_bytes, src_rank_idx, ...);

    // NVLink 对称指针访问（get_sym_ptr）
    template <typename team_t>
    dtype_t* get_sym_ptr(ptr, dst_rank_idx);
};
```

三种 Team 的语义：
- `ncclTeamTagLsa`：NVLink 域内通信（同节点 GPU 间），使用对称指针直接访问
- `ncclTeamTagRail`：RDMA 域内通信（跨节点），使用 GIN put/get
- `ncclTeamTagWorld`：全局通信，自动判断走 NVLink 还是 RDMA

关键优势：
- **统一通信后端**：不再需要 NVSHMEM，NCCL 是 PyTorch 标配，零额外安装
- **更灵活的 QP（Queue Pair）管理**：支持 QP 共享模式（`NCCL_GIN_RESOURCE_SHARING_CTA`/`GPU`），允许多个 SM 或 warp 共享同一个 QP
- **对称内存（Symmetric Memory）**：通过 `ncclCommWindowRegister` + `ncclGetLsaPointer` 实现 NVLink 域内零拷贝直接访问
- **原子远程操作**：`ncclGin_VASignalAdd` 支持远程原子加，用于通知机制和屏障

### 1.3 Hybrid Kernel 统一跨域通信

**v1 方案**：intranode（NVLink）和 internode（RDMA+NVLink 转发）是完全独立的 kernel 实现（`intranode.cu` vs `internode.cu`），代码逻辑不共享，且 internode 的数据流是先 RDMA 发到目标节点的某个 GPU，再由该 GPU 通过 NVLink 转发到同节点其他 GPU。

**v2 方案**：引入 "hybrid" kernel（`hybrid_dispatch.cuh` / `hybrid_combine.cuh`），将 scaleout（跨节点 RDMA）和 scaleup（同节点 NVLink）融合在一个 kernel 中，通过 warp 分工实现流水线式数据传输。

---

## 二、核心性能优化

### 2.1 Warp 级角色特化——SM 利用率降低的关键

v2 最核心的 SM 利用率优化来自 kernel 内部的 warp 角色特化设计。

**Hybrid dispatch kernel 的 warp 分工**（`hybrid_dispatch.cuh`）：

```
每个 SM 的线程布局：
┌─────────────────┬──────────────────┬──────────────────┐
│  Notify Warps   │ Scaleout Warps   │  Forward Warps   │
│  (4 warps)      │ (N warps)        │  (N warps)       │
│  统计+通知      │ RDMA 数据发送    │  NVLink 转发     │
└─────────────────┴──────────────────┴──────────────────┘
```

- **Notify warps**（`kNumNotifyWarps = 4`）：遍历所有 token 的 top-k 索引，在 shared memory 中统计每个 rank 和每个 expert 将接收的 token 数量，然后通过 GIN 将计数结果发送给所有对端。这部分只做元数据操作，不搬运数据。
- **Scaleout warps**：将需要跨节点传输的 token 通过 TMA 加载到 shared memory，再通过 GIN put 发送到目标节点的接收缓冲区。每个 warp 对应一个 "channel"，实现多通道并行传输。
- **Forward warps**：从 RDMA 接收缓冲区中读取跨节点收到的 token，判断其目标 scaleup rank，再通过 TMA store + GIN put 转发到目标 NVLink 对端。

**对比 v1 的 SM 使用方式**：

v1 的 internode kernel 使用所有 SM 执行同一套逻辑：先 RDMA 发送，再 NVLink 转发，每个 SM 都参与所有阶段的工作。这导致大量 SM 在等待 RDMA 完成时空转。

v2 中，不同角色的 warp 可以并行执行：notify warps 统计计数的同时，scaleout warps 已经在发送数据，forward warps 可以在收到部分数据后立即开始 NVLink 转发。这大幅减少了 SM 空转时间。

**SM 数量自适应**：

v2 引入了 `prefer_overlap_with_compute` 参数（`csrc/elastic/buffer.hpp`）：
- 当 `prefer_overlap_with_compute=True`（默认）：使用更少的 SM 运行通信 kernel，为计算 kernel 留出更多 SM
- 当 `prefer_overlap_with_compute=False`：使用更多 SM 加速通信

在 `csrc/kernels/elastic/dispatch.hpp` 中：
```cpp
// 纯 NVLink 模式：限制每个 SM 的 warp 数，总线程数不超过 512
num_dispatch_warps = std::min<int>(std::min<int>(
    (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps),
    math::ceil_div(512, num_sms));
```

### 2.2 TMA（Tensor Memory Accelerator）全面应用

**v1** 中数据搬运主要依赖传统的 `memcpy` 风格操作（`int4` 向量化加载/存储），以及少量 TMA 用法。

**v2** 中 TMA 被用于几乎所有数据搬运路径（`deep_ep/include/deep_ep/common/ptx.cuh`）：

```cpp
// TMA 异步加载：从全局内存到 shared memory
ptx::tma_load_1d(dst_ptr, src_ptr, mbarrier_ptr, num_bytes);

// TMA 异步存储：从 shared memory 到全局内存
ptx::tma_store_1d(dst_ptr, src_ptr, num_bytes);

// cp.async：用于 SF（scaling factor）的非对齐加载
ptx::cp_async_ca(gmem_src, smem_dst);
```

TMA 的关键优势：
- **零 SM 计算资源消耗**：TMA 操作由 GPU 的 TMA 硬件单元执行，不占用 CUDA core，SM 上的线程只需发起请求和等待完成
- **与 mbarrier 的天然协作**：TMA 操作通过 mbarrier（`mbarrier_arrive_and_set_tx`）自动同步，不需要额外的 barrier 指令
- **L2 缓存提示**：v2 使用 `kEvictFirst`（加载时快速逐出）和 `kEvictNormal`（存储时正常缓存）的缓存提示，优化 L2 缓存利用率

**TMA 预取**（prefetch）优化：在 hybrid dispatch 中，scaleout warps 在发送当前 token 的同时，预取下一个 token 的数据：
```cpp
// 预取下一个 token（与 IBGDA 请求重叠）
preload_next_token(token_idx + kNumChannels);
```

### 2.3 Programmatic Dependent Launch（PDL）

v2 利用了 CUDA 的 PDL 机制，将 dispatch 操作拆分为两个协作 kernel：

1. **主 dispatch kernel**（`dispatch_impl` / `hybrid_dispatch_impl`）：完成数据传输
2. **Copy epilogue kernel**（`dispatch_copy_epilogue_impl`）：将接收到的数据从内部缓冲区拷贝到用户输出张量

PDL 机制：
```cpp
// 主 kernel 末尾触发 PDL 完成信号
cudaTriggerProgrammaticLaunchCompletion();

// Epilogue kernel 开头等待依赖
cudaGridDependencySynchronize();
```

**性能意义**：
- 主 kernel 完成数据传输后立即通过 PDL 通知 epilogue kernel 启动，无需 GPU 全局同步
- Epilogue kernel 在等待期间不消耗 SM 资源（由 CUDA runtime 管理）
- 相比 v1 的同步方式（CPU 轮询 `moe_recv_counter`），PDL 实现了真正的 kernel-to-kernel 低延迟衔接

### 2.4 Channel 化多通道并行传输

v2 引入了多 channel 架构，每个 SM 拥有多个 channel（每个 scaleout/forward warp 对应一个 channel）：

```cpp
constexpr int kNumChannelsPerSM = kNumScaleoutWarps;  // 每个 SM 的 channel 数
constexpr int kNumChannels = kNumScaleoutWarps * kNumSMs;  // 总 channel 数
constexpr int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels);
```

数据分布策略：
- Token 按 channel 交错分配：`token_idx = channel_idx, channel_idx + kNumChannels, ...`
- 每个 channel 有独立的接收缓冲区和 tail 指针
- Forward warps 以 chunk 为单位处理数据，实现细粒度的流水线

**Channel tail 信令机制**：
```cpp
// Scaleout warp 定期更新 channel tail（每 3 个 token 更新一次）
const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
    if (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag) {
        gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
    }
};
```

这个 `kScaleoutUpdateInterval = 3` 的设计非常巧妙：不是每发送一个 token 就通知一次（信令开销大），也不是等所有 token 发完才通知（forward warp 空等），而是每 3 个 token 批量通知一次，平衡了信令开销和转发延迟。

### 2.5 CPU-GPU 同步优化

**v1 方案**：dispatch 完成后，GPU 将接收计数写入 CPU 可见的 mapped memory（`moe_recv_counter_mapped`），CPU 轮询等待计数到达预期值后才继续执行。这意味着每次 dispatch/combine 都有一次 CPU-GPU 同步。

**v2 方案**：引入 `do_cpu_sync` 参数：
- `do_cpu_sync=True`：与 v1 相同，CPU 等待 GPU 计数（向后兼容）
- `do_cpu_sync=False`（默认推荐）：GPU 端直接从 GPU tensor 读取计数，CPU 不等待

```cpp
// combine kernel 中：不需要 CPU 同步时，从 GPU tensor 读取实际接收 token 数
if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
    num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);
```

这消除了 dispatch→combine 之间的 CPU-GPU 同步点，实现真正的 GPU 端到 GPU 端流水线。

### 2.6 确定性模式（Deterministic Mode）与 Slot 重用

v2 引入了两种优化 dispatch 路径的模式：

**Cached Mode**（`cached_mode=True`）：
- 跳过 notify 阶段（`num_notify_warps = 0`），直接使用前一次 dispatch 缓存的 slot 索引
- 适用于 token 路由模式不变的推理场景（如同 batch 内所有 token 走相同的 expert）
- 节省了 notify 的 4 个 warp 和所有计数统计开销

**Deterministic Mode**（`deterministic=True`）：
- 使用单独的 `dispatch_deterministic_prologue_impl` kernel（`dispatch_deterministic_prologue.cuh`）
- 在 shared memory 中完成全局 prefix sum，为每个 top-k 选择预计算确定性的目标 slot 索引
- 保证相同输入产生相同的输出排列顺序

```cpp
// Cached mode 跳过 notify
const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;
const bool reuse_slot_indices = cached_mode or deterministic;
```

### 2.7 多 Reduction 模式与 Expanded Layout

**v1**：combine 阶段每个 token 的多个 top-k 选择共享同一个 slot，在 combine 时需要额外的 reduction 步骤来合并多个来源的数据。

**v2** 引入 `allow_multiple_reduction` 和 `use_expanded_layout` 两个选项：

- **Expanded layout**（`do_expand=True`）：dispatch 时为每个 top-k 选择分配独立的 slot，combine 时无需 reduction，直接写回
- **Multiple reduction**（`allow_multiple_reduction=True`）：在 combine epilogue 中执行本地 reduction（多个 top-k 来源在同一 rank 上的数据），而不是在通信过程中 reduction

```cpp
// combine_utils.cuh 中的本地 reduction 优化
if (enable_hadd_bypass) {
    // 只有两个 top-k 来源时，使用 BF16 直接相加，避免 FP32 转换开销
    bf162_view_0[j * ... + l] += bf162_view_1[j * ... + l];
} else {
    // 多来源时才升级到 FP32 累加
    ptx::accumulate(reduced[j], bf162_view[j]);  // add.rn.f32.bf16
}
```

`add.rn.f32.bf16` 指令（SM100+）是 v2 在 PTX 层面的又一优化：将 BF16 加法融合为单条 FP32 累加指令，避免了 BF16→FP32 转换的额外开销。

### 2.8 Linked List 数据结构

在 hybrid mode 的 dispatch 中，v2 使用 linked list 来跟踪 token 的转发位置：

```cpp
// dispatch_copy_epilogue.cuh
// 维护 linked list 的尾部
if constexpr (kDoCreateLinkedList) {
    if (ptx::elect_one_sync())
        channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
}

// linked list 末尾标记为 -1
channel_linked_list[*workspace_layout.get_channel_scaleup_tail_ptr(i, k)] = -1;
```

Combine kernel 通过遍历 linked list 来定位需要获取的 token，避免了固定大小缓冲区的对齐浪费。

---

## 三、Barrier 与同步优化

### 3.1 分层 Barrier 设计

v1 使用 `nvshmem_barrier_all`（全局屏障）或 `barrier_block`（NVLink 屏障），粒度粗、开销大。

v2 实现了精细的分层 barrier（`deep_ep/include/deep_ep/common/comm.cuh`）：

```cpp
template <bool kIsScaleupNVLink, int kNumScaleoutRanks, int kNumScaleupRanks, ...>
void gpu_barrier(gin, workspace, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx,
                 bool do_scaleout = true, bool do_scaleup = true) {
    if (do_scaleup and do_scaleout) {
        // 并行执行 scaleup 和 scaleout barrier！
        if (sm_idx == 0) {
            // SM 0 做 NVLink barrier
            scaleup_barrier_wo_local_sync<...>(gin, workspace, ...);
        } else {
            // 其余 SM 做 RDMA barrier
            scaleout_barrier_wo_local_sync<...>(gin, ...);
        }
    }
}
```

关键优化：**scaleup barrier 和 scaleout barrier 并行执行**——SM 0 负责 NVLink 域的 barrier，其余 SM 负责 RDMA 域的 barrier，两种 barrier 重叠执行，总 barrier 时间从 `max(NVLink, RDMA)` 降低到接近 `max(NVLink, RDMA)`（而非串行的 `NVLink + RDMA`）。

### 3.2 NVLink Barrier 的 Phase 复用

```cpp
void nvlink_barrier_wo_local_sync(...) {
    const int status = static_cast<int>((*workspace.get_nvl_barrier_counter_ptr()) & 3);
    const int phase = status & 1, sign = status >> 1;
    // 使用 phase 位交替 +1/-1，避免计数器溢出和清零
    ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
}
```

通过 phase 位实现屏障的自动翻转，无需在每次 barrier 后清零信号量，减少了同步开销。

### 3.3 GIN Barrier 的 QP Flush 优化

```cpp
void gin_barrier_wo_local_sync(...) {
    // 先 flush 所有 QP 确保写入可见（release 语义）
    for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {
        ncclGin(..., i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
    }
    // 然后只用 QP 0 做 barrier 信号
    gin.signal(team, i, ncclGin_SignalInc{...});
}
```

所有 SM 的所有 warp 参与 flush，但只有 SM 0 参与 barrier 信号交换。flush 确保之前所有 GIN 写入对其他 rank 可见，barrier 信号本身只使用一个 QP，减少了信号量开销。

---

## 四、新功能

### 4.1 Engram（远程 KV Cache 获取）

```cpp
// 写入 KV cache 到对称内存
void engram_write(const torch::Tensor& storage);
// 异步获取远程 KV cache
std::function<torch::Tensor()> engram_fetch(const torch::Tensor& indices, int num_qps);
```

Engram 利用 NCCL GIN 的 RDMA get 操作直接从远程 GPU 的对称内存中读取 KV cache，无需通过 CPU 中转。返回一个 lambda，调用者可以在需要时同步等待结果，实现通信与计算的流水线。

### 4.2 Pipeline Parallel Send/Recv

```cpp
void pp_set_config(int64_t num_max_tensor_bytes, int num_max_inflight_tensors);
void pp_send(const torch::Tensor& x, int dst_rank_idx, int num_sms);
void pp_recv(const torch::Tensor& x, int src_rank_idx, int num_sms);
```

支持 NVLink 域内的 pipeline parallel 通信，支持多 inflight tensor（滑动窗口协议），利用 GIN put 实现低延迟传输。

### 4.3 All-Gather Reduce-Scatter（AGRS）

```cpp
void agrs_start_session(int num_max_sends, int num_sms);
void agrs_send(const torch::Tensor& x, int dst_rank_idx);
void agrs_recv(torch::Tensor& x, int src_rank_idx);
void agrs_end_session();
```

Session 化的 AGRS API，支持在一次 session 内执行多轮 all-gather/reduce-scatter，减少 barrier 开销。

---

## 五、PTX 层面优化

v2 在 `ptx.cuh` 中封装了大量底层 PTX 内联汇编指令，这些是 v2 性能优化的微观基础：

| 指令 | 作用 | 性能意义 |
|------|------|----------|
| `elect.sync` | 随机选举一个 lane 执行单次操作 | 替代 `lane_idx == 0`，减少 warp 分歧 |
| `mbarrier.init/arrive/wait` | 异步屏障同步 | TMA 的天然同步机制 |
| `tma_load_1d`/`tma_store_1d` | TMA 异步内存搬运 | 零 SM 计算资源消耗 |
| `cp.async.ca` | 异步拷贝（缓存对齐） | SF 数据的非对齐加载 |
| `red.add.rel.sys` | 远程原子加（release 语义） | 通知机制的核心，无需额外 fence |
| `ld.acquire.sys` | 获取语义加载 | 读取远程写入的数据，保证可见性 |
| `st.relaxed.sys`/`st.release.sys` | 不同语义的存储 | 精细控制内存序，减少不必要的 fence |
| `setmaxnreg.inc/dec` | 寄存器数量动态调整 | 在 warp group 间重新分配寄存器资源 |
| `add.rn.f32.bf16` | BF16→FP32 融合加法（SM100+） | 消除 BF16 combine 时的转换开销 |
| `ld.L1::no_allocate` | L1 缓存不分配策略 | 一次性读取的数据不污染 L1 缓存 |
| `match`/`deduplicate` | warp 内值匹配和去重 | 高效的 rank 去重操作 |

特别是 `L1::no_allocate` 缓存提示和 `ld.acquire.sys`/`st.release.sys` 的精确内存序控制，这些在 v1 中都不存在，是 v2 降低 SM 资源占用和提升带宽利用率的关键微观优化。

---

## 六、内存布局优化

### 6.1 Token Layout 统一抽象

v2 定义了统一的 `TokenLayout` 结构（`layout.cuh`），描述了一个 token 在缓冲区中的完整布局：

```
Token Layout:
┌──────────────────┬──────────────┬──────────────────────────────────────┬──────────┐
│  Hidden Bytes    │  SF Packs    │  Metadata                            │ MBarrier │
│  (对齐到32B)     │  (对齐到32B) │  [topk_idx | topk_weights | src_idx | linked] │ (对齐32B)│
└──────────────────┴──────────────┴──────────────────────────────────────┴──────────┘
```

v1 中 token 数据和元数据是分开存储和传输的，v2 将它们打包在一起，通过 TMA 一次性搬运。这减少了内存访问次数和同步开销。

### 6.2 BufferLayout 的 Rank/Channel 分层

```
Buffer Layout (Hybrid Dispatch):
┌─────────────────────────────────────────────────────────────────┐
│  Scaleup Buffer                                                 │
│  [scaleup_rank_0 | scaleup_rank_1 | ... | scaleup_rank_N]      │
├─────────────────────────────────────────────────────────────────┤
│  Scaleout Send Buffer (1 rank)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Scaleout Recv Buffer                                           │
│  [scaleout_rank_0: [channel_0 | channel_1 | ... | channel_M]]  │
│  [scaleout_rank_1: [channel_0 | channel_1 | ... | channel_M]]  │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

每个 scaleout rank 的接收缓冲区被分割为多个 channel，允许 forward warps 独立地从不同 channel 读取数据并转发，实现细粒度并行。

### 6.3 Workspace 统一管理

v2 的 `WorkspaceLayout` 将所有通信元数据（barrier 信号、notify 计数、channel tail、AGRS 信号等）统一放在一块 workspace 内存中，通过固定的偏移量访问：

```cpp
struct WorkspaceLayout {
    static constexpr int kNumMaxRanks = 1024;
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;
    // 固定大小的布局，所有配置复用同一块内存
    static int64_t get_num_bytes();
};
```

这消除了 v1 中各种临时 buffer 的动态分配，workspace 在 buffer 创建时一次性分配并清零。

---

## 七、v1 与 v2 架构对比总结

| 维度 | v1 | v2 |
|------|----|----|
| **编译模型** | AOT（setup.py 预编译） | JIT（运行时编译 + 缓存） |
| **RDMA 后端** | NVSHMEM | NCCL GIN |
| **内存模型** | IPC 共享内存 + NVSHMEM 对称内存 | NCCL 对称内存（ncclWindow） |
| **Kernel 架构** | intranode/internode 分离 | hybrid 统一 + 纯 NVLink 路径保留 |
| **Warp 分工** | 所有 warp 执行相同逻辑 | notify/scaleout/forward 角色特化 |
| **数据搬运** | int4 向量化 load/store | TMA 异步 load/store + cp.async |
| **同步机制** | NVSHMEM barrier / CPU 轮询 | GPU barrier（分层 + 并行）+ PDL |
| **CPU 同步** | 每次 dispatch/combine 必须同步 | 可选跳过 CPU 同步（GPU 端读取计数） |
| **Channel 架构** | 固定 channel 数 | 自适应 channel 数（每 SM 多 channel） |
| **SM 使用量** | 使用所有可用 SM | 自适应，默认更少 SM（`prefer_overlap_with_compute`）|
| **Combine 模式** | 单一模式 | expanded layout / multiple reduction / BF16 直加 |
| **确定性** | 不支持 | 支持 deterministic mode |
| **Cached mode** | 不支持 | 支持（跳过 notify，重用 slot 索引） |
| **新功能** | dispatch/combine only | +Engram, +PP, +AGRS |
| **SM100+ 优化** | 不支持 | `add.rn.f32.bf16`, `longlong4_t` 256-bit LD/ST |

---

## 八、SM 利用率大幅降低的根因分析

v2 实现 SM 利用率大幅降低的核心机制，按贡献从大到小排列：

1. **TMA 替代 SM 数据搬运**（最大贡献）：v1 中每个 token 的数据搬运由 SM 上的 CUDA core 执行 load/store 指令完成；v2 中 TMA 硬件单元负责搬运，SM 线程只需发起 TMA 请求和等待 mbarrier，实际计算资源占用极低。

2. **更少的 SM 数量**：v2 通过 `prefer_overlap_with_compute` 和自适应 SM 选择，默认使用更少的 SM 运行通信 kernel。例如 dispatch kernel 限制总线程数不超过 512/num_sms，将更多 SM 留给计算 kernel。

3. **Warp 角色特化减少空转**：v1 中所有 SM 执行相同的全流程逻辑，大量时间花在等待上；v2 中 notify/scaleout/forward 三种角色并行工作，每种角色都有持续的 work 可做，减少了等待空转。

4. **Channel 流水线化**：多 channel 允许 forward warps 在 scaleout warps 还在发送数据时就开始转发已到达的数据，实现了发送与转发的重叠，减少了整体 kernel 执行时间。

5. **CPU-GPU 同步消除**：`do_cpu_sync=False` 模式下，整个 dispatch→combine 流水线无需 CPU 介入，GPU 端到端完成。v1 中每次通信都需要 CPU 轮询等待，CPU-GPU 同步的延迟（约 10-50μs）被完全消除。

6. **PDL 实现 kernel 无缝衔接**：dispatch 主 kernel 与 copy epilogue kernel 之间通过 PDL 衔接，无需 GPU 全局同步或 CPU 介入，减少了 kernel 间的空闲时间。

7. **Cached mode 跳过 notify**：在推理场景下，跳过 notify 阶段（节省 4 个 warp = 128 线程/SM）直接使用缓存的 slot 索引，进一步降低了 SM 占用。

---

## 九、带宽性能提升的根因分析

1. **多通道并行 RDMA**：v2 的 channel 架构允许同一 SM 内的多个 warp 同时发起 RDMA 请求，多个 QP 的并行利用（最多 65/129 个 QP）大幅提升了 RDMA 带宽利用率。

2. **GIN 的 QP 共享与聚合**：`ncclGinOptFlagsAggregateRequests` 标志允许 GIN 将多个小消息聚合为更大的 RDMA 传输，减少了 per-message 开销。

3. **TMA 预取与流水线**：scaleout warps 在等待当前 token 的 RDMA 传输完成时，预取下一个 token 的数据到 shared memory，隐藏了内存访问延迟。

4. **NVLink 直接写入**：v2 中 NVLink 对端的数据通过 TMA store 直接写入远程 GPU 的缓冲区（对称指针），无需先写入本地 send buffer 再转发。v1 的 internode 实现需要先将 RDMA 数据写入本地 GPU buffer，再通过 NVLink 转发到目标 GPU。

5. **Scaleout/Scaleup Barrier 并行**：分层 barrier 中 NVLink barrier 和 RDMA barrier 并行执行，减少了同步开销对带宽的影响。

6. **L2 缓存优化**：TMA 操作的缓存提示（`kEvictFirst` 用于加载、`kEvictNormal` 用于存储）和 `L1::no_allocate` 策略减少了缓存污染，提升了有效带宽。
