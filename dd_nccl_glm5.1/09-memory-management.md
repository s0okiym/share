# NCCL 内存管理系统

NCCL 内存管理分三层：底层 CuMem 虚拟内存管理分配器、中层内存管理器（挂起/恢复）、上层类型安全分配接口。

---

## 1. 分配架构总览

```mermaid
flowchart TD
    A["NCCL 内部分配调用"] --> B["ncclCudaMalloc / ncclCudaCalloc\n(alloc.h 模板封装)"]
    B --> C{ncclCuMemEnable()?}
    C -->|"是 (CUDA >= 11.3)"| D["ncclCuMemAlloc\nCUDA VMM 路径"]
    C -->|"否"| E["cudaMalloc\n标准路径"]

    D --> D1["cuMemGetAllocationGranularity — 对齐"]
    D1 --> D2["cuMemCreate — 分配物理内存"]
    D2 --> D3["cuMemAddressReserve — 预留虚拟地址"]
    D3 --> D4["cuMemMap — 映射 VA 到物理分配"]
    D4 --> D5["cuMemSetAccess — 授予本设备+P2P peers 读写"]
    D5 --> D6["ncclMemTrack — 注册到内存管理器"]

    E --> F["返回 cudaMalloc 指针\n(不经管理器追踪)"]
```

### 1.1 CuMem Handle 类型

| Handle 类型 | CUDA 版本 | 用途 |
|------------|----------|------|
| `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` | >= 11.3 | POSIX FD，用于跨进程共享 |
| `CU_MEM_HANDLE_TYPE_FABRIC` | >= 12.3 | Fabric handle，更高效，优先选择 |

### 1.2 ncclMemAlloc / ncclMemFree (公共 API)

```mermaid
flowchart TD
    A["ncclMemAlloc(ptr, size)"] --> B{CuMem 启用?}
    B -->|"是"| C["ncclCuMemAlloc\nVMM 序列"]
    B -->|"否"| D["cudaMalloc"]
    C --> E["对齐到分配粒度\n优先 FABRIC handle\n回退 POSIX FD"]
    D --> F["返回指针"]
    E --> F

    G["ncclMemFree(ptr)"] --> H{CuMem 启用?}
    H -->|"是"| I["ncclCuMemFree\nVMM 释放序列"]
    H -->|"否"| J["cudaFree"]
    I --> K["恢复原始 CUDA 设备"]
    J --> K
```

---

## 2. 内存类型与内存管理器

### 2.1 三种内存类型

| 类型 | 值 | 行为 | 追踪方式 |
|------|---|------|---------|
| ncclMemPersist | 0 | 永不释放 | 仅原子计数，不创建链表条目 |
| ncclMemScratch | 1 | 挂起时直接释放 (不保存内容) | 链表条目 |
| ncclMemOffload | 2 | 挂起前拷贝到 CPU，恢复时还原 | 链表条目 |

### 2.2 内存管理器结构

```mermaid
classDiagram
    class ncclMemManager {
        +entries: ncclDynMemEntry* (linked list)
        +numEntries: int
        +lock: mutex
        +released: int (1=suspended)
        +initialized: int (atomic)
        +refCount: int (shared communicators)
        +totalPersist: size_t
        +totalPersistImported: size_t
        +totalScratch: size_t
        +totalScratchImported: size_t
        +totalOffload: size_t
        +totalOffloadImported: size_t
        +cpuBackupUsage: size_t
        +commCudaDev: int
    }

    class ncclDynMemEntry {
        +ptr: void*
        +size: size_t
        +handle: CUmemGenericAllocationHandle
        +handleType: int
        +memType: ncclMemType_t
        +state: int (Active/Released)
        +cudaDev: int
        +cpuBackup: void*
        +isImportedFromPeer: int
        +desc: union (local/imported)
        +next: ncclDynMemEntry*
    }

    ncclMemManager --> ncclDynMemEntry : entries (linked list)
```

### 2.3 追踪流程

```mermaid
flowchart TD
    A["ncclMemTrack(manager, ptr, size, memType)"] --> B{memType?}
    B -->|"Persist"| C["原子递增 totalPersist\n不创建链表条目"]
    B -->|"Scratch/Offload"| D["创建 ncclDynMemEntry\nmutex 保护\nprepend 到链表"]

    E["ncclMemTrackImportFromPeer\n(manager, ptr, size, memType, ownerRank, ownerDev, ownerVA)"]
    E --> F["同 Scratch/Offload\n但标记 isImportedFromPeer\n记录 owner 信息"]

    G["ncclMemUntrack(manager, ptr)"] --> H["搜索链表 (by ptr)"]
    H --> I{找到?}
    I -->|"是"| J["移除条目\n递减计数器"]
    I -->|"否"| K["递减 totalPersist\n(Persist 类型无链表条目)"]
```

---

## 3. 挂起与恢复

### 3.1 挂起流程 (ncclCommMemSuspend)

```mermaid
flowchart TD
    A["ncclCommMemSuspend"] --> A1["检查 refCount <= 1\n(共享通信器不可挂起)"]
    A1 --> B["cudaDeviceSynchronize\n+ bootstrap barrier\n确保所有 rank 就绪"]

    B --> C["第一遍: 释放 peer-imported 缓冲区"]
    C --> C1["cuMemUnmap(VA)\ncuMemRelease(physical handle)"]
    C1 --> C2["state = Released"]

    C2 --> D["第二遍: 释放本地缓冲区"]
    D --> D1{memType?}
    D1 -->|"Offload"| D2["分配 cpuBackup\ncudaMemcpy D2H\n保存数据"]
    D1 -->|"Scratch"| D3["不保存数据"]
    D1 -->|"Persist"| D4["不应到这里 (Persist 不追踪)"]

    D2 --> D5["关闭 POSIX FD / FABRIC handle"]
    D3 --> D5
    D5 --> D6["cuMemUnmap(VA) — 保留 VA 预留\ncuMemRelease(physical handle)"]
    D6 --> D7["state = Released"]

    D7 --> E["manager->released = 1"]
```

### 3.2 恢复流程 (ncclCommMemResume)

```mermaid
flowchart TD
    A["ncclCommMemResume"] --> B["第一遍: 恢复本地缓冲区"]

    B --> B1["cuMemCreate — 新物理分配"]
    B1 --> B2["ncclCuMemMapAndSetAccess\n映射到同一 VA"]
    B2 --> B3["恢复 peer 访问权限\n(所有 exportedPeerRanks)"]

    B3 --> B4{memType?}
    B4 -->|"Offload"| B5["cudaMemcpy H2D\n还原数据\n释放 cpuBackup"]
    B4 -->|"Scratch"| B6["数据已丢失, 无需还原"]

    B5 --> B7["FABRIC: cuMemExportToShareableHandle"]
    B6 --> B7
    B7 --> B8["state = Active"]

    B8 --> C["Bootstrap barrier\n所有 rank 本地内存已恢复"]

    C --> D["Handle 交换:\nAllGather P2P handle 计数\nsend/recv ncclDynMemP2pHandleInfo"]

    D --> E["第二遍: 恢复 peer-imported 缓冲区"]
    E --> E1["查找匹配 handle info"]
    E1 --> E2{handle 类型?}
    E2 -->|"POSIX FD"| E3["ncclProxyClientGetFdBlocking\n获取 FD → cuMemImportFromShareableHandle"]
    E2 -->|"FABRIC"| E4["直接 import fabric handle"]
    E3 --> E5["ncclCuMemMapAndSetAccess\n映射到同一 VA"]
    E4 --> E5
    E5 --> E6["state = Active"]

    E6 --> F["Final barrier\n所有 rank 完成"]
```

### 3.3 公共 API

| API | 说明 |
|-----|------|
| `ncclCommSuspend(comm, flags)` | 挂起通信器内存。需 `NCCL_SUSPEND_MEM` flag。拒绝 refCount>1 |
| `ncclCommResume(comm)` | 恢复通信器内存 |
| `ncclCommMemStats(comm, stats)` | 查询内存统计: total/persist/suspend/suspended |

---

## 4. 子分配器

### 4.1 ncclSpace — 偏移量子分配器

```mermaid
flowchart LR
    subgraph "ncclSpace 切割视图"
        E1["empty\n0-100"] --> F1["full\n100-300"] --> E2["empty\n300-500"] --> F2["full\n500-700"] --> E3["empty\n700-∞"]
    end
```

**数据结构**:
```c
struct ncclSpace {
    int count;      // 切割点数量
    int capacity;   // 分配容量
    int64_t* cuts;  // 升序排列的边界值
};
```

**不变量**: 段 `i` 为 "full" 当 `i % 2 != count % 2`，否则为 "empty"。最后一段始终为 empty。

**操作**:
- `ncclSpaceAlloc`: 线性扫描空段，首次适配。快速路径移动边界；慢速路径 insertSegment
- `ncclSpaceFree`: 线性扫描满段。快速路径收缩；慢速路径 insertSegment
- `insertSegment`: 插入两个切割点，然后压缩相邻零大小空段

### 4.2 ncclShadowPool — 设备/主机影射对象池

```mermaid
flowchart TD
    A["ncclShadowPool"] --> B["哈希表: devicePtr → hostShadowPtr\n动态增长, 2:1 对象/桶比"]
    A --> C["CUDA 内存池 (cudaMemPool)"]
    A --> D["页面链表"]

    C --> E{对象大小?}
    E -->|"小对象 (64KB/size >= 3)"| F["页式分配:\nncclShadowPage (最多64个slot)\nfreeMask 位图管理"]
    E -->|"大对象"| G["直接从 CUDA 内存池分配\ncudaMallocFromPoolAsync"]

    F --> F1["popFirstOneBit(&page->freeMask)\n获取空闲 slot"]
    F1 --> F2["Host shadow 内存内联在\nncclShadowObject 之后\n对齐到 max_align_t"]
```

---

## 5. 分配接口 (alloc.h)

### 5.1 核心 API

| 函数 | 用途 |
|------|------|
| `ncclCudaMalloc(ptr, count, manager, memType)` | GPU 内存分配，CuMem 或 cudaMalloc |
| `ncclCudaCalloc(ptr, count, manager, memType)` | 同上 + 零初始化 (side stream) |
| `ncclCudaCallocAsync(ptr, count, stream, manager, memType)` | 同上 + 在指定 stream 上零初始化 |
| `ncclCudaFree(ptr, numSegments)` | GPU 内存释放 |
| `ncclCudaHostCalloc(ptr, count)` | 固定主机内存 (cudaHostAllocMapped) |
| `ncclCalloc(ptr, count)` | 普通主机内存 (malloc + memset) |
| `ncclRealloc(ptr, oldCount, newCount)` | 增长主机分配 |
| `ncclIbMalloc(ptr, size)` | 页对齐分配 (posix_memalign, 用于 IB 注册) |

### 5.2 智能指针

| 类型 | 用途 |
|------|------|
| `ncclUniquePtr<T>` | RAII 封装，std::unique_ptr<T, decltype(&std::free)> |
| `ncclUniqueArrayPtr<T>` | 数组版本 RAII |

---

## 6. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/allocator.cc` | ~600 | CuMem 分配器、ncclSpace、ncclShadowPool |
| `src/mem_manager.cc` | ~1000 | 内存管理器、挂起/恢复 |
| `src/include/alloc.h` | ~500 | 分配接口模板封装 |
| `src/include/mem_manager.h` | ~150 | 内存管理器数据结构 |
| `src/include/allocator.h` | ~60 | ncclSpace/ShadowPool 声明 |
