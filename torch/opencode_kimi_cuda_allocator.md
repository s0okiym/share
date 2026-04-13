# PyTorch CUDA Caching Allocator深度分析

## 目录
1. [架构概览](#1-架构概览)
2. [核心数据结构](#2-核心数据结构)
3. [Block分配策略](#3-block分配策略)
4. [分配流程详解](#4-分配流程详解)
5. [释放流程详解](#5-释放流程详解)
6. [Block分割与合并](#6-block分割与合并)
7. [OOM处理机制](#7-oom处理机制)
8. [垃圾回收](#8-垃圾回收)
9. [Expandable Segments](#9-expandable-segments)
10. [CUDA Graph支持](#10-cuda-graph支持)

---

## 1. 架构概览

### 1.1 核心文件位置

| 文件 | 路径 | 描述 |
|------|------|------|
| 主实现 | c10/cuda/CUDACachingAllocator.cpp | 核心分配器逻辑 (~4969行) |
| 头文件 | c10/cuda/CUDACachingAllocator.h | 公共API和类接口 |
| 通用接口 | c10/core/CachingDeviceAllocator.h | 基类和统计结构 |
| CUDA配置 | c10/cuda/CUDAAllocatorConfig.h | CUDA特定配置 |
| 基础配置 | c10/core/AllocatorConfig.h | 通用分配器配置 |

### 1.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code                                │
│              at::empty(), tensor.clone(), etc.              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDACachingAllocator                     │
│    - malloc: 分配内存                                       │
│    - free: 释放内存                                         │
│    - emptyCache: 清理缓存                                   │
│    - memory_stats: 统计信息                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BlockPool                                │
│    ┌─────────────────┐    ┌─────────────────┐              │
│    │   small_blocks  │    │   large_blocks  │              │
│    │   (<= 1MB)      │    │   (> 1MB)       │              │
│    │   std::set      │    │   std::set      │              │
│    │   (按大小排序)   │    │   (按大小排序)   │              │
│    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Block                                    │
│    - ptr: 内存地址                                          │
│    - size: 块大小                                           │
│    - allocated: 是否已分配                                  │
│    - prev/next: 分割块链表                                  │
│    - pool: 所属池                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Driver                              │
│    - cudaMalloc, cudaFree                                   │
│    - cuMemAddressReserve (expandable)                       │
│    - cuMemMap, cuMemUnmap                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心数据结构

### 2.1 Block结构

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第196-265行)
struct Block {
  c10::DeviceIndex device;      // GPU设备索引
  cudaStream_t stream;          // 分配流
  stream_set stream_uses;       // 使用过该块的流
  int32_t registration_counter{-1};
  size_t size;                  // 块大小（字节）
  size_t requested_size;        // 原始请求大小
  BlockPool* pool{nullptr};     // 所属内存池
  void* ptr{nullptr};           // 内存地址
  bool allocated{false};        // 是否已分配
  bool mapped{true};            // 物理内存映射状态
  Block* prev{nullptr};         // 前一个分割块
  Block* next{nullptr};         // 后一个分割块
  int event_count{0};           // 未完成CUDA事件数
  int64_t gc_count_base{0};     // 插入时的GC计数器
  std::shared_ptr<GatheredContext> context_when_allocated;
  std::shared_ptr<GatheredContext> context_when_segment_allocated;
  ExpandableSegment* expandable_segment_{nullptr};
  
  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};
```

### 2.2 BlockPool结构

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第171-192行)
struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  std::set<Block*, Comparison> blocks;      // 空闲块，按大小排序
  std::set<Block*, Comparison> unmapped;    // 未映射块（expandable segments）
  const bool is_small;
  PrivatePool* owner_PrivatePool;
  int64_t get_free_blocks_call_count{0};
};
```

### 2.3 大小阈值

```cpp
// 来自c10/core/AllocatorConfig.h (第15-24行)
constexpr size_t kSmallBuffer = 2097152;     // 2 MiB
constexpr size_t kMinBlockSize = 512;        // 最小分配大小
constexpr size_t kSmallSize = 1048576;       // 1 MiB - 小/大块分界
constexpr size_t kMinLargeAlloc = 10485760;  // 10 MiB
constexpr size_t kRoundLarge = 2097152;      // 大块舍入粒度（2 MiB）
```

---

## 3. Block分配策略

### 3.1 池选择

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第3321-3346行)
BlockPool& get_pool(size_t size, cudaStream_t stream) {
  // 检查CUDA图捕获池
  if (C10_UNLIKELY(!captures_underway.empty())) {
    for (auto it = captures_underway.rbegin(); it != captures_underway.rend(); ++it) {
      if (it->second(stream)) {
        auto it1 = graph_pools.find(it->first);
        if (size <= kSmallSize) {
          return it1->second->small_blocks;
        } else {
          return it1->second->large_blocks;
        }
      }
    }
  }
  
  // 默认池基于大小选择
  if (size <= kSmallSize) {
    return small_blocks;
  } else {
    return large_blocks;
  }
}
```

### 3.2 大小舍入

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第2802-2814行)
static size_t round_size(size_t size) {
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  } else {
    auto divisions = AcceleratorAllocatorConfig::roundup_power2_divisions(size);
    if (divisions > 1 && size > (kMinBlockSize * divisions)) {
      return roundup_power2_next_division(size, divisions);
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }
}
```

### 3.3 分配大小计算

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第3376-3384行)
static size_t get_allocation_size(size_t size) {
  if (size <= kSmallSize) {
    return kSmallBuffer;  // 2 MiB for small
  } else if (size < kMinLargeAlloc) {
    return AcceleratorAllocatorConfig::large_segment_size();  // 20 MiB default
  } else {
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }
}
```

---

## 4. 分配流程详解

### 4.1 分配流程图

```mermaid
flowchart TD
    A[malloc size] --> B[Round size to alignment]
    B --> C[Select Pool]
    C --> D[Process Events]
    D --> E[Search Free Block]
    
    E -->|Found| F[Check Split]
    E -->|Not Found| G[Try GC]
    
    G -->|Success| E
    G -->|Fail| H[Allocate New Block]
    
    H -->|Success| F
    H -->|Fail| I[OOM Retry Chain]
    
    I -->|Success| F
    I -->|Fail| J[Throw OOM Error]
    
    F -->|Should Split| K[Split Block]
    F -->|No Split| L[Use Entire Block]
    
    K --> M[alloc_found_block]
    L --> M
    
    M --> N[Update Stats]
    N --> O[Add to active_blocks]
    O --> P[Return Block.ptr]
```

### 4.2 核心分配代码

```cpp
// 分配主流程
void* CUDACachingAllocator::malloc(int device, size_t size, cudaStream_t stream) {
  // 1. 舍入大小
  size = round_size(size);
  
  // 2. 选择池
  BlockPool& pool = get_pool(size, stream);
  
  // 3. 处理事件（释放已完成的块）
  process_events();
  
  // 4. 查找空闲块
  Block* block = get_free_block(size, stream, &pool);
  
  if (block != nullptr) {
    // 5. 检查是否需要分割
    if (should_split(block, size)) {
      block = split_block(block, size);
    }
    return alloc_found_block(block, size, stream);
  }
  
  // 6. 尝试垃圾回收
  garbage_collect_cached_blocks();
  block = get_free_block(size, stream, &pool);
  if (block != nullptr) {
    return alloc_found_block(block, size, stream);
  }
  
  // 7. 分配新块
  block = alloc_block(size, stream, &pool);
  if (block != nullptr) {
    return alloc_found_block(block, size, stream);
  }
  
  // 8. OOM处理
  return oom_allocation(device, size, stream);
}
```

### 4.3 查找空闲块

```cpp
Block* get_free_block(size_t size, cudaStream_t stream, BlockPool* pool) {
  // 使用best-fit策略在有序集合中查找
  auto it = pool->blocks.lower_bound(&key);
  
  // 遍历找到合适的块
  for (; it != pool->blocks.end(); ++it) {
    Block* block = *it;
    if (!block->allocated && block->size >= size) {
      // 检查流兼容性
      if (block->stream == stream || block->stream_uses.empty()) {
        return block;
      }
    }
  }
  return nullptr;
}
```

---

## 5. 释放流程详解

### 5.1 释放流程图

```mermaid
flowchart TD
    A[free ptr] --> B[Find Block]
    B --> C[Update Statistics]
    C --> D[Check Stream Uses]
    
    D -->|Multiple Streams| E[Check Capture Status]
    D -->|Single Stream| F[Direct free_block]
    
    E -->|In Capture| G[Deferred Free]
    E -->|Not Capturing| H[insert_events]
    
    H --> I[Add to cuda_events queue]
    
    F --> J[Remove from active_blocks]
    J --> K[Check Merge Candidates]
    
    K --> L[try_merge_blocks]
    L -->|Merge with prev| M[Merge prev block]
    L -->|Merge with next| N[Merge next block]
    
    M --> O[Return to pool]
    N --> O
    
    G --> P[record_free_markers]
    P --> Q[Add to deferred_blocks]
```

### 5.2 释放核心代码

```cpp
void CUDACachingAllocator::free(Block* block) {
  // 1. 更新统计
  stats.increaseAllocated(-block->size);
  
  // 2. 检查是否在多个流上使用
  if (!block->stream_uses.empty()) {
    // 插入事件进行延迟释放
    insert_events(block);
    return;
  }
  
  // 3. 直接释放
  free_block(block);
}

void free_block(Block* block) {
  // 从active_blocks中移除
  active_blocks.erase(block->ptr);
  
  // 尝试合并
  Block* merged = try_merge_blocks(block->prev, block, pool);
  if (merged != nullptr) {
    block = merged;
  }
  merged = try_merge_blocks(block, block->next, pool);
  if (merged != nullptr) {
    block = merged;
  }
  
  // 返回池中
  pool->blocks.insert(block);
}
```

---

## 6. Block分割与合并

### 6.1 分割决策

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第3356-3374行)
bool should_split(const Block* block, size_t size, bool is_expandable_segments_active) {
  // 检查池是否标记为不可分割
  if (no_split_pools.find(block->pool->owner_MempoolId()) != no_split_pools.end()) {
    return false;
  }
  
  size_t remaining = block->size - size;
  if (block->pool->is_small || is_expandable_segments_active) {
    return remaining >= kMinBlockSize;
  } else {
    return (size < AcceleratorAllocatorConfig::max_split_size()) &&
           (remaining > kSmallSize);
  }
}
```

### 6.2 合并块

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第3288-3319行)
size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
  // 无法合并的条件
  if (!src || src->allocated || src->event_count > 0 ||
      !src->stream_uses.empty() || dst->mapped != src->mapped) {
    return 0;
  }

  AT_ASSERT(dst->is_split() && src->is_split());

  if (dst->prev == src) {  // [src dst]
    dst->ptr = src->ptr;
    dst->prev = src->prev;
    if (dst->prev) {
      dst->prev->next = dst;
    }
  } else {  // [dst src]
    dst->next = src->next;
    if (dst->next) {
      dst->next->prev = dst;
    }
  }
  
  const size_t subsumed_size = src->size;
  dst->size += subsumed_size;
  
  auto erased = src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
  delete src;
  return subsumed_size;
}
```

### 6.3 分割与合并流程图

```mermaid
flowchart TD
    subgraph Splitting
        A[Found free block] --> B[Size = 20MB]
        B --> C[Need = 8MB]
        C --> D[should_split?]
        D -->|Check no_split_pool| E[Check remaining >= kMinBlockSize]
        E -->|Yes| F[alloc_found_block with split]
        F --> G[Create new Block for allocation]
        G --> H[Update remaining block]
        H --> I[Update statistics]
    end
    
    subgraph Coalescing
        J[free_block] --> K[Remove from active_blocks]
        K --> L[Check merge candidates]
        L --> M[try_merge_blocks]
        M -->|Conditions met| N[Merge [prev block][this block]]
        N --> O[prev->size += this->size]
        O --> P[Delete this block]
        P --> Q[Update statistics]
    end
```

---

## 7. OOM处理机制

### 7.1 OOM处理流程图

```mermaid
flowchart TD
    A[alloc_block fails] --> B[cudaErrorMemoryAllocation]
    B --> C[Check oom_rejection_info.rejected]
    
    C -->|Yes| D[Return nullptr]
    C -->|No| E[Normal Retry Chain]
    
    E --> F[try_mempool_fallback]
    F -->|Success| G[Use block from pool]
    F -->|Fail| H[release_available_cached_blocks]
    
    H -->|Success| I[Retry alloc]
    H -->|Fail| J[release_cached_blocks]
    
    J -->|Success| I
    J -->|Fail| K[All retries failed]
    
    K --> L[OOM Error Reporting]
    L --> M[Get free/total from cudaMemGetInfo]
    M --> N[Calculate stats]
    N --> O[Call oom_observers]
    O --> P[Format error message]
    P --> Q[Throw OutOfMemoryError]
```

### 7.2 OOM错误报告

```cpp
// OOM错误格式
"CUDA out of memory. Tried to allocate X MiB. "
"GPU Y has Z MiB total capacity. "
"Allocated A MiB, reserved B MiB. "
"CUDACachingAllocator stats: ..."
```

---

## 8. 垃圾回收

### 8.1 GC机制

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第3455-3512行)
void garbage_collect_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
  // 计算阈值
  size_t gc_threshold = static_cast<size_t>(
      AcceleratorAllocatorConfig::garbage_collection_threshold() *
      static_cast<double>(allowed_memory_maximum.value()));
  
  if (total_allocated_memory <= gc_threshold) {
    return;  // 无需GC
  }
  
  const auto target_size = total_allocated_memory - gc_threshold;
  
  // 计算可释放块的平均年龄
  size_t total_age = 0.0;
  int freeable_block_count = 0;
  for (auto& b : large_blocks.blocks) {
    if (!b->is_split()) {
      total_age += b->gc_count();
      ++freeable_block_count;
    }
  }
  
  double age_threshold = static_cast<double>(total_age) / freeable_block_count;
  
  // 释放超过平均年龄的块
  for (auto it = large_blocks.blocks.begin(); it != large_blocks.blocks.end();) {
    Block* block = *it;
    ++it;
    if (!block->is_split() && !block->expandable_segment_ &&
        static_cast<double>(block->gc_count()) >= age_threshold) {
      release_block(block, context);
    }
  }
}
```

---

## 9. Expandable Segments

### 9.1 概述

Expandable Segments允许分配器预留大的虚拟地址空间，按需映射物理页面，减少碎片。

### 9.2 ExpandableSegment结构

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第376-966行)
struct ExpandableSegment {
  // 映射虚拟地址范围到物理内存
  SegmentRange map(SegmentRange range);
  
  // 取消映射物理页面（归还内存给系统）
  SegmentRange unmap(SegmentRange range);
  
  // 创建可共享的句柄用于IPC
  SegmentRange share(SegmentRange range, std::ostream& buf);
  
  // 从共享句柄导入
  static std::unique_ptr<ExpandableSegment> fromShared(...);
  
  // 使用CUDA驱动的cuMemAddressReserve, cuMemCreate, cuMemMap
};
```

### 9.3 分配流程

```cpp
Block* try_allocate_expandable_block(
    c10::DeviceIndex device,
    cudaStream_t stream,
    BlockPool* pool,
    size_t size,
    const std::shared_ptr<GatheredContext>& ctx) {
  
  // 查找或创建expandable segment
  Block* candidate = find_expandable_block(device, stream, pool, size);
  
  // 如需要则映射物理内存
  if (!candidate->mapped &&
      !map_block(candidate, std::min(candidate->size, size), ctx)) {
    return nullptr;
  }
  
  // 通过映射更多内存来扩展
  while (candidate->size < size) {
    auto remaining = size - candidate->size;
    auto new_candidate = candidate->next;
    if (!map_block(new_candidate, std::min(remaining, candidate->next->size), ctx)) {
      return nullptr;
    }
    candidate = new_candidate;
  }
  
  return candidate;
}
```

---

## 10. CUDA Graph支持

### 10.1 PrivatePool

```cpp
// 来自c10/cuda/CUDACachingAllocator.cpp (第1119-1162行)
struct PrivatePool {
  MempoolId_t id;
  int use_count{1};           // 使用该池的图数量
  int cudaMalloc_count{0};    // 未释放的cudaMalloc数量
  std::shared_ptr<CUDAAllocator> allocator_;
  BlockPool large_blocks;
  BlockPool small_blocks;
};
```

### 10.2 捕获通知

```cpp
void beginAllocateToPool(MempoolId_t mempool_id, std::function<bool(cudaStream_t)> filter);
void endAllocateToPool(MempoolId_t mempool_id);
void releasePool(MempoolId_t mempool_id);  // 图销毁时调用
```

### 10.3 延迟释放

```cpp
// 捕获期间，cudaEventQuery是非法的，使用图节点标记延迟释放
void record_free_markers(Block* block);
void add_to_deferred_blocks(Block* block);
// 捕获结束后释放
```

---

## 11. 配置选项

### 11.1 关键设置

| 选项 | 默认值 | 描述 |
|------|--------|------|
| max_split_size_mb | 无限制 | 可分割的最大块大小 |
| max_non_split_rounding_mb | 0 | 超大块的舍入容差 |
| garbage_collection_threshold | 0 | GC触发阈值（0-1） |
| expandable_segments | false | 启用可扩展段 |
| roundup_power2_divisions | [] | 大小舍入除数 |
| per_process_memory_fraction | 1.0 | 内存使用限制 |
| throw_on_cudamalloc_oom | false | 预抢占OOM拒绝 |
| graph_capture_record_stream_reuse | false | 启用流重用优化 |

---

## 12. 内存统计

### 12.1 DeviceStats

```cpp
struct DeviceStats {
  StatArray allocation;          // COUNT: 分配请求
  StatArray segment;             // COUNT: cudaMalloc段
  StatArray active;              // COUNT: 活跃内存块
  StatArray inactive_split;      // COUNT: 非活跃分割块
  StatArray allocated_bytes;     // SUM: 已分配字节
  StatArray reserved_bytes;      // SUM: 保留字节
  StatArray active_bytes;        // SUM: 活跃字节
  StatArray inactive_split_bytes;// SUM: 非活跃分割字节
  StatArray requested_bytes;     // SUM: 客户端请求字节
  
  int64_t num_alloc_retries = 0;
  int64_t num_ooms = 0;
  int64_t num_sync_all_streams = 0;
};
```

---

## 13. 总结

PyTorch的CUDA Caching Allocator是一个精密的GPU内存管理系统：

1. **块池管理**：小/大块池分离，快速查找合适块

2. **内存分配策略**：Best-fit分配，避免过多碎片

3. **分割合并**：动态分割大块，释放时自动合并相邻块

4. **OOM处理**：多层重试机制，包括GC、释放缓存块

5. **Expandable Segments**：虚拟内存管理，减少碎片

6. **CUDA Graph支持**：PrivatePool管理图内存，延迟释放

7. **统计和调试**：详细的内存统计和追踪功能

该分配器成功平衡了：
- **性能**：快速分配，避免频繁cudaMalloc
- **内存效率**：块重用，合并分割
- **稳定性**：OOM优雅处理
- **灵活性**：支持多种使用场景（标准训练、CUDA图、多流）
