# NCCL CUDA Graph 持久化集合

"持久化" (Persistent) 在 NCCL 中与 CUDA Graph 捕获同义。被捕获到 CUDA Graph 中的 Plan 称为持久化 Plan，因为 Graph 可能被反复重放，Plan 数据必须在 Graph 生命周期内保持有效。

---

## 1. 持久化 vs 非持久化对比

| 方面 | 非持久化 Plan | 持久化 Plan |
|------|-------------|------------|
| **触发条件** | 正常执行 (非 Graph 捕获) | CUDA Graph 捕获中 |
| **工作存储** | FIFO (预算 = FIFO 大小 / 2) | Persistent buffer (预算 = 1<<30 bytes) |
| **回收时机** | 内核启动后立即回收 | Graph 销毁时回收 |
| **引用计数** | 无 | persistentRefs++ / localPersistentRefs++ |
| **Host stream** | 检查 serial event 后决定 | 始终使用 host stream |
| **缓冲区注册** | ncclCommRegister | ncclCommGraphRegister |
| **序列号** | 动态递增 | 使用常量 GRAPH_SYNC_VALUE=1 |

---

## 2. 持久化 Plan 生命周期

### 2.1 创建与捕获

```mermaid
flowchart TD
    A["ncclLaunchPrepare"] --> B{ncclCudaGraphValid?\n(planner->capturingGraph)"}
    B -->|"否"| C["非持久化 Plan\nworkStorageType = Fifo"]
    B -->|"是"| D["持久化 Plan\nworkStorageType = Persistent"]

    D --> D1["分配大预算工作缓冲区\nworkBufPersistent\n(1<<30 bytes)"]
    D1 --> D2["persistent = true"]
    D2 --> D3["缓冲区注册使用\nncclCommGraphRegister\n(而非 ncclCommRegister)"]
```

### 2.2 启动与保持

```mermaid
flowchart TD
    A["ncclLaunchKernel (持久化 Plan)"] --> B["cuLaunchKernelEx\n内核被捕获到 CUDA Graph"]

    B --> C["不立即回收 Plan!\n(非持久化 Plan 才立即回收\nvia callbackQueue)"]

    C --> D["递增引用计数:\nsharedRes->persistentRefs++\ncomm->localPersistentRefs++"]

    D --> E["注册 persistentDestructor\n到 CUDA Graph\n(cudaGraphExecStreamCaptureUpdate\n或 graph destroy callback)"]

    E --> F["Graph 被反复重放\nPlan 数据必须保持有效\n(工作缓冲区不变)"]
```

### 2.3 销毁与回收

```mermaid
flowchart TD
    A["CUDA Graph 销毁"] --> B["persistentDestructor 回调"]

    B --> C["遍历 Plan 链表"]
    C --> D["每个 Plan 加入\ncomm->callbackQueue"]

    D --> E["异步回收 (callbackQueue 处理)"]
    E --> E1{workStorageType == Persistent?}
    E1 -->|"是"| E2["cudaThreadExchangeStreamCaptureMode\n放松捕获模式\n(允许在捕获中 free)"]
    E2 --> E3["释放 workBufPersistent"]
    E1 -->|"否"| E4["正常释放"]

    E3 --> F["递减引用计数:\nsharedRes->persistentRefs--\ncomm->localPersistentRefs--"]
    E4 --> F
```

---

## 3. 引用计数与同步

### 3.1 两个引用计数器

```mermaid
flowchart TD
    subgraph "per-device sharedRes"
        A["persistentRefs\n同一设备上所有通信器的持久化 Plan 总数"]
    end

    subgraph "per-comm"
        B["localPersistentRefs\n该通信器的持久化 Plan 数"]
    end
```

### 3.2 同步影响

```mermaid
flowchart TD
    A["非持久化内核启动\n(ncclLaunchKernelBefore)"] --> B{persistentRefs > 0?}
    B -->|"是"| C["需要检查 host stream 的\nserial event 是否完成\n(因为持久化 Plan 可能\n正在使用 host stream)"]
    B -->|"否"| D["跳过检查\n(更快的路径)"]
```

### 3.3 Host Stream 处理

```mermaid
flowchart TD
    A["ncclLaunchKernelAfter_NoCuda"] --> B{Plan 持久化?}
    B -->|"否"| C{serial event 完成?}
    C -->|"是"| D["需要 host stream 工作\nhostStreamPlanTask"]
    C -->|"否"| E["跳过 host stream"]
    B -->|"是"| F["始终使用 host stream\n(Graph 内核不能依赖\n普通流序)"]
```

---

## 4. CUDA Graph 捕获模式

### 4.1 流捕获切换

在持久化 Plan 需要执行 CUDA 操作（如释放工作缓冲区）时：

```mermaid
flowchart TD
    A["需要释放 workBufPersistent\n(在 Graph 捕获模式下)"] --> B["cudaThreadExchangeStreamCaptureMode\n(cudaStreamCaptureModeRelaxed)"]
    B --> C["cudaFree(workBufPersistent)"]
    C --> D["恢复捕获模式\n(cudaStreamCaptureModeThreadLocal)"]
```

### 4.2 Graph 重放约束

```mermaid
flowchart TD
    A["CUDA Graph 重放"] --> B["工作缓冲区位置固定\n(同一次捕获的所有重放\n使用相同的 GPU 地址)"]
    B --> C["序列号使用常量\nGRAPH_SYNC_VALUE = 1\n(不能使用递增序列号\n因为 Graph 是固定的)"]
    C --> D["Ready/Complete flags\n重放后重置为 0\n(避免跨重放的状态泄漏)"]
```

---

## 5. 持久化与缓冲区注册

### 5.1 图注册 vs 普通注册

| 方面 | ncclCommRegister | ncclCommGraphRegister |
|------|-----------------|----------------------|
| 引用计数 | localRefs++ | graphRefs++ |
| 注册优先级 | 本地注册 | 优先图注册，回退本地 |
| 生命周期 | deregister 时递减 | graph deregister 时递减 |
| 两计数器独立 | localRefs 和 graphRefs 分别计数 | 仅当两者都为 0 时才清理 |

### 5.2 注册在 Plan 中的应用

```mermaid
flowchart TD
    A["ncclTasksRegAndEnqueue"] --> B{Plan 持久化 且\nncclParamGraphRegister?}
    B -->|"是"| C["使用 ncclCommGraphRegister\n注册 NVLS/CollNet 缓冲区"]
    B -->|"否"| D["使用 ncclCommRegister\n或跳过注册"]
```

---

## 6. 关键源文件

| 文件 | 功能 |
|------|------|
| `src/enqueue.cc` (ncclLaunchPrepare) | Plan 创建，检测 Graph 捕获 |
| `src/enqueue.cc` (uploadWork) | 工作存储类型选择 (Fifo/Persistent) |
| `src/enqueue.cc` (ncclLaunchKernelAfter) | 持久化 Plan 的 host stream 处理 |
| `src/enqueue.cc` (persistentDestructor) | Graph 销毁时的 Plan 回收 |
| `src/register/register.cc` | 图注册 vs 普通注册的引用计数 |
| `src/include/comm.h` | persistentRefs / localPersistentRefs 字段 |
