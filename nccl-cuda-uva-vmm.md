# NCCL 中 CUDA Unified Addressing 与 Virtual Memory Management API 使用分析

## 一、背景：CUDA 内存寻址模型演进

### 1.1 统一虚拟寻址 (UVA)

CUDA 4.0 引入 UVA：CPU 和所有 GPU 共享单一虚拟地址空间。任意指针在进程内唯一，可通过 `cudaMemcpyDefault` 自动推断传输方向。UVA 是 `cudaIpcGetMemHandle` 等跨进程共享机制的基础。

### 1.2 虚拟内存管理 (VMM)

CUDA 10.2 (Driver API 11.3+) 引入 VMM API，将 GPU 内存管理拆分为三层：

```
┌────────────────────────────────────────────┐
│  虚拟地址 (VA) — cuMemAddressReserve/Free  │  ← 对进程可见的地址空间
├────────────────────────────────────────────┤
│  物理分配 (PA) — cuMemCreate/Release       │  ← 实际的 HBM 物理页
├────────────────────────────────────────────┤
│  映射 (Map)  — cuMemMap/Unmap + cuMemSetAccess │ ← VA → PA 映射 + 访问权限
└────────────────────────────────────────────┘
```

VMM 将传统 `cudaMalloc` 的"分配即绑定"拆解为可独立控制的步骤。

### 1.3 NCCL 面临的内存管理挑战

| 挑战 | 说明 |
|------|------|
| 跨进程共享 | MPI 多进程场景下，不同 rank 需要访问彼此的 GPU buffer |
| 跨节点 RDMA | 网卡需要直接访问 GPU 内存 (GDR)，需要 DMA-BUF fd 或 peermem |
| Suspend/Resume | 动态离线/上线 GPU，需要保存/恢复内存内容 |
| 多 GPU 可见性 | 同一物理内存需被多个 GPU（通过 NVLink/P2P）同时访问 |
| 用户 buffer 注册 | 用户调用 `ncclCommRegister` 注册自己的 buffer 用于通信 |
| MNNVL (Multi-Node NVLink) | 跨节点 NVLink fabric 需要特殊的内存共享机制 |

---

## 二、NCCL 使用的 CUDA 内存 API 详解

### 2.1 VMM 分配：`cuMemCreate` + `cuMemAddressReserve` + `cuMemMap`

**核心代码**: `src/allocator.cc:15-91` (`ncclMemAlloc`), `src/include/alloc.h:255-290` (`ncclCuMemAlloc`)

这是 NCCL 的主要 GPU 内存分配方式（当 VMM 可用时），替代 `cudaMalloc`：

```cpp
// allocator.cc:36-91 — ncclMemAlloc 简化流程
CUmemAllocationProp memprop = {};
memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;         // 固定内存（不可换出）
memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 位于 GPU 上
memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // 可导出 fd
if (CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED)
    memprop.allocFlags.gpuDirectRDMACapable = 1;      // 支持 GDR

// Step 1: 创建物理分配
cuMemCreate(&handle, handleSize, &memprop, 0);

// Step 2: 预留虚拟地址
cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0);

// Step 3: 映射 VA → PA
cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0);

// Step 4: 设置多 GPU 访问权限
for (int i = 0; i < dcnt; ++i) {
    if (i == cudaDev || cudaDeviceCanAccessPeer(&p2p, i, cudaDev)) {
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1);
    }
}
```

**为什么用 VMM 而非 `cudaMalloc`？**

| `cudaMalloc` | VMM (`cuMemCreate` + ...) |
|---|---|
| 分配即绑定，无法单独控制物理/虚拟 | 物理/虚拟/映射三层解耦 |
| 无法导出共享 handle | `requestedHandleTypes` 支持导出 POSIX fd 或 FABRIC handle |
| 无法设置 GDR capable 标志 | `gpuDirectRDMACapable=1` 让 NIC 可以直接 DMA |
| 访问权限隐式限制在本设备 | `cuMemSetAccess` 可精确控制哪些 GPU 可访问 |
| 不支持 suspend/resume | 解映射后可重新映射到不同的物理页 |

### 2.2 Handle 导出/导入：跨进程共享

**核心代码**: `src/transport/p2p.cc:216-254` (`ncclP2pAllocateShareableBuffer`), `p2p.cc:260-319` (`ncclP2pImportShareableBuffer`)

#### 导出端（发送方分配 buffer 后导出 handle）

```cpp
// p2p.cc:228-233
if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // POSIX fd 路径：handle 通过 Unix Domain Socket 传 fd
    memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle));
} else {
    // FABRIC handle 路径：直接序列化 handle
    cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0);
}
```

#### 导入端（接收方映射远端 buffer 到本地 VA）

```cpp
// p2p.cc:278-300
if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // 通过 UDS 从远端获取 fd
    ncclProxyClientGetFdBlocking(comm, peer, &cuDesc->data, &fd);
    cuMemImportFromShareableHandle(&handle, (void*)(uintptr_t)fd, type);
    close(fd);
} else {
    cuMemImportFromShareableHandle(&handle, cuDesc, type);
}
// 在本地预留 VA 并映射
cuMemAddressReserve(&dptr, size, 0, 0, 0);
cuMemMap(dptr, size, 0, handle, 0);
// 设置本地 GPU 可读写
cuMemSetAccess(dptr, size, &accessDesc, 1);
```

**关键点**：导入端获得的是同一块物理内存的另一个虚拟地址映射。两个进程通过各自的 VA 访问同一物理 HBM，NVLink/P2P 硬件保证一致性。

### 2.3 Handle 类型选择

**代码**: `src/misc/cudawrap.cc:19`, `allocator.cc:38-43`

```cpp
CUmemAllocationHandleType ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
// 运行时可能升级为 FABRIC handle:
if (cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED))
    requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
```

| Handle 类型 | 传输方式 | 适用场景 |
|---|---|---|
| `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` | Unix Domain Socket 传 fd | 单节点内进程间共享 |
| `CU_MEM_HANDLE_TYPE_FABRIC` | 直接序列化 handle 结构 | MNNVL 跨节点 NVLink fabric |

### 2.4 DMA-BUF FD：RDMA 注册

**代码**: `src/transport/net.cc:961`, `alloc.h:51-93` (`ncclCuMemHostAlloc`)

对于 GDR 场景，NCCL 使用 `cuMemGetHandleForAddressRange` 获取 DMA-BUF fd，传给 `ibv_reg_dmabuf_mr`：

```cpp
// net.cc:961
CUCHECK(cuMemGetHandleForAddressRange(
    (void*)&dmabuf_fd,
    (CUdeviceptr)resources->buffers[p],   // GPU buffer 地址
    resources->buffSizes[p],              // 大小
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,  // DMA-BUF fd 类型
    getHandleForAddressRangeFlags(resources->useGdr)  // PCI 模式标志
));
NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(
    resources->netSendComm, resources->buffers[p], resources->buffSizes[p],
    NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &resources->mhandles[p]
));
close(dmabuf_fd);
```

C2C (Cache Coherent) 平台上，`getHandleForAddressRangeFlags` 会强制 PCIe 映射：
```cpp
// net.cc:284-290
if (useGdr == ncclTopoGdrModePci)
    flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
```

**为什么需要 DMA-BUF 而非传统 peermem？**

| 传统 peermem (nv_peer_mem) | DMA-BUF |
|---|---|
| 需要加载额外内核模块 | 内核 5.12+ 原生支持 |
| 依赖 nvidia 驱动的 peermem 接口 | 标准 Linux DMA-BUF 框架 |
| 不支持 `IBV_ACCESS_RELAXED_ORDERING` | 配合 `ibv_reg_dmabuf_mr` 支持 |
| 与 GDR Copy 交互受限 | 更好的生态兼容性 |

### 2.5 Host 端 VMM 分配：CPU/GPU 双向可访问

**代码**: `src/include/alloc.h:51-93` (`ncclCuMemHostAlloc`)

CUDA 12.2+ 支持在 HOST_NUMA 位置创建物理分配，并映射到 GPU VA：

```cpp
// alloc.h:66-88
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;  // 物理内存位于 CPU NUMA 节点
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
cuMemCreate(&handle, size, &prop, 0);
cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0);
cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0);
// GPU 可读写
accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1);
// CPU 可读写
accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1);
```

**效果**：同一块物理内存（CPU 侧 pinned），CPU 和 GPU 都可通过各自的地址直接访问。避免了 `cudaMallocHost` 只能 CPU 侧访问、GPU 需要 `cudaMemcpy` 的限制。

### 2.6 地址范围查询：`cuMemGetAddressRange` + `cuPointerGetAttribute`

**代码**: `src/transport/p2p.cc:893`, `p2p.cc:992-993`, `allocator.cc:120`

NCCL 在用户 buffer 注册场景中，需要从用户指针反查底层分配信息：

```cpp
// p2p.cc:893 — 遍历用户 buffer 覆盖的所有 VMM 段
cuMemGetAddressRange(&tmpBase, &tmpBaseSize, mappedPtrEnd);

// p2p.cc:992-993 — 判断用户 buffer 是否支持 legacy IPC
cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff);
cuPointerGetAttribute(&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)baseAddr);

// allocator.cc:120 — ncclMemFree 中确定指针所属设备
cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr);
```

**作用**：用户传入的 buffer 可能跨多个 VMM 段（例如 `cudaMalloc` 分配的大 buffer 在 VMM 中可能被分成多段）。NCCL 需要逐段导出 handle，每段单独做 IPC 映射。

### 2.7 Handle 保留：`cuMemRetainAllocationHandle`

**代码**: `src/transport/p2p.cc:895`, `p2p.cc:236`

```cpp
// p2p.cc:895 — 从已有 VA 反查物理分配 handle
cuMemRetainAllocationHandle(&segmentHandles[segment], (void*)tmpBase);

// p2p.cc:236 — 增加引用计数，防止对端进程 abort 释放后本端访问失效
for (int r = 0; r < refcount; ++r)
    cuMemRetainAllocationHandle(&handle, *ptr);
```

**用途**：
1. 用户 buffer 注册：从 `cudaMalloc` 分配的指针反查底层 handle，用于 IPC 导出
2. 引用计数管理：同进程多 rank 共享 buffer 时，增加引用计数防止提前释放

### 2.8 Legacy CUDA IPC：`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`

**代码**: `src/transport/p2p.cc:244`, `p2p.cc:313`

VMM 不可用时的回退方案：

```cpp
// 导出
cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr);

// 导入
cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess);
```

**与 VMM 方案的对比**：

| Legacy CUDA IPC | VMM (cuMem) |
|---|---|
| 一个 IPC handle 对应一个 `cudaMalloc` 分配 | 一个 handle 对应一个 `cuMemCreate` 物理分配 |
| 不支持 GDR capable 标志 | `gpuDirectRDMACapable` 可选 |
| `cudaIpcOpenMemHandle` 原子操作 | `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess` 三步走 |
| 不支持 FABRIC handle（仅单节点） | 支持 FABRIC handle（MNNVL 跨节点） |
| 访问权限隐式（Lazy P2P） | 显式 `cuMemSetAccess` 精确控制 |
| 不支持 suspend/resume | 解映射后可重映射 |

### 2.9 P2P 访问：`cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess`

**代码**: `src/allocator.cc:83`, `src/transport/p2p.cc:343`, `src/transport/net.cc:456`

```cpp
// allocator.cc:83 — 决定哪些 GPU 可以访问新分配的内存
if (i == cudaDev || (cudaDeviceCanAccessPeer(&p2p, i, cudaDev) && p2p)) {
    // 给 GPU i 设置读写权限
    accessDesc.location.id = i;
    cuMemSetAccess(..., &accessDesc, 1);
}

// p2p.cc:343 — 同进程不同 GPU 间启用 P2P
cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
```

**VMM 模式下的关键区别**：传统模式下 `cudaDeviceEnablePeerAccess` 是必须的，VMM 模式下通过 `cuMemSetAccess` 显式授权，但仍需底层 P2P 硬件支持（`cudaDeviceCanAccessPeer` 返回 true）。

---

## 三、NCCL P2P 传输的四种模式

**代码**: `src/transport/p2p.cc:20`

```cpp
enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };
```

| 模式 | 场景 | 内存共享方式 | 代码路径 |
|------|------|------------|---------|
| `P2P_DIRECT` | 同进程 + P2P 可达 | 直接指针（`cudaDeviceEnablePeerAccess`） | `p2p.cc:405` |
| `P2P_CUMEM` | 跨进程 + VMM 可用 | `cuMemExport/Import` + `cuMemMap` | `p2p.cc:411` |
| `P2P_IPC` | 跨进程 + VMM 不可用 | `cudaIpcGet/OpenMemHandle` | `p2p.cc:417` |
| `P2P_INTERMEDIATE` | P2P 不可达 | 通过中间 GPU 转发 | `p2p.cc:424` |

**选择逻辑** (`p2p.cc:402-424`):
```
同进程 + P2P 可达 + 未禁用 → P2P_DIRECT
同进程/跨进程 + VMM 可用   → P2P_CUMEM
同进程/跨进程 + VMM 不可用 → P2P_IPC
P2P 不可达                 → P2P_INTERMEDIATE
```

### 3.1 P2P_DIRECT：直接指针

```
GPU0 (进程A)                    GPU1 (进程A)
  ┌──────────┐                   ┌──────────┐
  │ VA: 0x100│─── NVLink ────→  │ VA: 0x100│
  │ (直接访问) │                   │ (本设备)  │
  └──────────┘                   └──────────┘
```

- 同进程内，通过 `cudaDeviceEnablePeerAccess` 后直接使用对方 buffer 的 VA
- 无需 IPC handle 交换
- 性能最高（零拷贝，无映射开销）

### 3.2 P2P_CUMEM：VMM 跨进程映射

```
进程A (GPU0)                    进程B (GPU1)
  ┌──────────┐                   ┌──────────┐
  │ VA_A: 0x200│                  │ VA_B: 0x300│
  │    ↕ 映射  │                   │    ↕ 映射  │
  │ 物理分配 H │←──共享 handle──→│ 物理分配 H │
  └──────────┘ (同一物理内存)    └──────────┘
```

- 进程 A 通过 `cuMemExportToShareableHandle` 导出 handle
- 进程 B 通过 `cuMemImportFromShareableHandle` 导入 handle
- 进程 B 在本地 `cuMemAddressReserve` + `cuMemMap` 创建新的 VA 映射
- `cuMemSetAccess` 授权本地 GPU 访问
- NVLink/P2P 硬件保证缓存一致性

### 3.3 P2P_IPC：Legacy IPC 跨进程

与 P2P_CUMEM 类似，但使用 `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`。不支持 GDR、FABRIC handle、suspend/resume。

### 3.4 P2P_INTERMEDIATE：中间 GPU 转发

当两个 GPU 之间没有 P2P 通路（如拓扑上不直连），通过一个可达的中间 GPU 做两跳传输。

---

## 四、Suspend/Resume 与 VMM

**代码**: `src/mem_manager.cc`

NCCL 支持 GPU 动态离线（suspend）和上线（resume），VMM 是实现此功能的基础：

```
Resume 流程:
1. cuMemCreate — 在新 GPU 上创建物理分配
2. cudaMemcpy — 从 CPU 备份 buffer 恢复数据
3. cuMemUnmap — 解除旧的 VA→PA 映射
4. cuMemMap   — 映射 VA→新 PA（同一 VA，不同的物理内存）
5. cuMemSetAccess — 重新设置访问权限
6. cuMemExportToShareableHandle — 重新导出 handle 给对端
```

**关键**：VMM 的 VA/PA 分离使得 NCCL 可以在 **不改变虚拟地址** 的前提下，将物理内存从一个 GPU 迁移到另一个 GPU。所有持有旧 VA 的 kernel 代码无需修改。

---

## 五、用户 Buffer 注册流程

**代码**: `src/transport/p2p.cc:1125` (`ncclIpcLocalRegisterBuffer`)

当用户调用 `ncclCommRegister` 注册自己的 buffer 时：

```
1. ncclCuMemGetAddressRange  — 查找用户 buffer 覆盖的 VMM 段
2. cuMemRetainAllocationHandle — 从每段 VA 反查物理 handle
3. cuMemExportToShareableHandle — 导出 handle（或传 fd）
4. 对端: cuMemImportFromShareableHandle → cuMemAddressReserve → cuMemMap → cuMemSetAccess
```

**用户 buffer 可能由 `cudaMalloc` 分配**（非 VMM），此时 NCCL 仍能通过 `cuMemRetainAllocationHandle` 获取底层 handle 并导出。这是 VMM API 的一个重要特性——它可以操作任何 CUDA 分配的内存，无论原始分配方式。

---

## 六、方案对比：为什么 NCCL 选择 VMM

### 6.1 VMM vs `cudaMalloc` + `cudaIpcGetMemHandle`

| 维度 | `cudaMalloc` + Legacy IPC | VMM (`cuMemCreate` + ...) |
|------|--------------------------|--------------------------|
| 跨进程共享 | `cudaIpcGetMemHandle` | `cuMemExportToShareableHandle` |
| 跨节点共享 | 不支持 | FABRIC handle (MNNVL) |
| GDR 支持 | 依赖 peermem 模块 | `gpuDirectRDMACapable` + DMA-BUF fd |
| Suspend/Resume | 不支持（VA 绑定 PA） | 支持（VA 可重映射） |
| 访问控制 | 隐式（Lazy P2P） | 显式 `cuMemSetAccess` |
| 用户 buffer 注册 | 需要探测 IPC capability | `cuMemRetainAllocationHandle` 通用 |
| 多段 buffer | 一个 IPC handle = 一个分配 | 逐段导出/映射 |
| Host/GPU 共享内存 | `cudaMallocHost`（仅 CPU 侧） | VMM host alloc（CPU+GPU 双向可访问） |
| 引用计数 | 无 | `cuMemRetainAllocationHandle` |

### 6.2 VMM vs 手动 mmap + RDMA 注册

| 维度 | 手动 mmap | VMM |
|------|----------|-----|
| GPU 内存共享 | 需要自己管理 pin/unpin | CUDA 驱动自动管理 |
| 页表同步 | 需要手动 invalidate | 驱动自动处理 |
| GDR path | 依赖 peermem 注册页 | DMA-BUF fd 原生支持 |
| 跨 GPU 可见性 | 手动 `cudaDeviceEnablePeerAccess` | `cuMemSetAccess` 批量设置 |

### 6.3 VMM 的代价

- **粒度对齐**：VMM 分配必须按 `cuMemGetAllocationGranularity` 对齐（通常 2MB），小分配浪费空间
- **API 复杂度**：4 步分配 vs `cudaMalloc` 一步到位
- **驱动版本要求**：CUDA 11.3+ (Driver 470+)，早期驱动不可用
- **容器限制**：某些容器运行时禁用 NUMA，导致 Host VMM 分配失败

NCCL 通过 `ncclCuMemEnable()` 自动检测 VMM 可用性，不可用时回退到 `cudaMalloc` + legacy IPC。

---

## 七、DMA-BUF FD 在 RDMA 注册中的角色

### 7.1 两种 GDR 内存注册路径

```
路径1: 传统 peermem
  GPU buffer → nv_peer_mem 内核模块 → ibv_reg_mr → NIC 可 DMA

路径2: DMA-BUF
  GPU buffer → cuMemGetHandleForAddressRange → dmabuf_fd → ibv_reg_dmabuf_mr → NIC 可 DMA
```

### 7.2 DMA-BUF 路径的优势

1. **内核原生支持**：不需要 `nv_peer_mem` 模块
2. **Relaxed Ordering**：`ibv_reg_dmabuf_mr` 支持 `IBV_ACCESS_RELAXED_ORDERING`
3. **PCIe 映射控制**：`CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE` 强制 PCIe 路径（C2C 平台需要）
4. **与 VMM 分配配合**：`cuMemCreate` 时设置 `gpuDirectRDMACapable=1`，确保物理页对 NIC 可见

### 7.3 NCCL 的选择逻辑

```
if (useGdr && dmaBufSupport && ptrSupport & NCCL_PTR_DMABUF) {
    // DMA-BUF 路径
    cuMemGetHandleForAddressRange → ibv_reg_dmabuf_mr
} else {
    // peermem 回退路径
    ibv_reg_mr (nv_peer_mem 自动处理 GPU 页)
}
```

---

## 八、总结

NCCL 大量使用 CUDA VMM API 而非 `cudaMalloc`，核心动机是：

1. **跨进程共享**：VMM handle 可通过 fd 或 FABRIC handle 传递，支持单节点和跨节点
2. **GDR 支持**：`gpuDirectRDMACapable` 标志 + DMA-BUF fd 让 NIC 直接 DMA 到 GPU HBM
3. **显式访问控制**：`cuMemSetAccess` 精确控制哪些 GPU/NUMA 节点可访问
4. **Suspend/Resume**：VA/PA 分离使得动态迁移成为可能
5. **用户 buffer 注册**：`cuMemRetainAllocationHandle` 可操作任意 CUDA 分配
6. **Host/GPU 共享内存**：VMM host alloc 同时映射到 CPU 和 GPU 地址空间

VMM 的代价（粒度对齐、API 复杂度、驱动版本要求）通过自动检测和回退机制缓解。整体架构确保了 NCCL 在从单 GPU 到 MNNVL 跨节点 NVLink 的全场景下都能高效地进行内存共享和通信。
