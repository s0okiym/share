# AMD CLR Event 机制深度解析

## 1. 概述与目的

在 AMD CLR（Compute Language Runtime）中，**Event** 机制是整个 GPU 计算运行时的心脏。它不仅是 HIP 与 OpenCL 两大前端共享的底层同步原语，也是连接主机端（Host）与设备端（Device）、连接软件调度层与硬件执行层的关键桥梁。

Event 机制的核心目的可以概括为以下几点：

- **命令状态追踪**：每一个提交到 GPU 的操作（内核启动、内存拷贝、内存填充等）都对应一个 Event 对象，用于精确追踪该命令从入队（Queued）、提交（Submitted）、执行（Running）到完成（Complete）的全生命周期状态。
- **主机-设备同步**：允许主机线程以阻塞或非阻塞方式等待一个或多个 GPU 操作完成，这是 `hipEventSynchronize`、`clWaitForEvents` 等 API 的根基。
- **流间/队列间同步**：支持不同执行流（HIP Stream / OpenCL Command Queue）之间的显式依赖管理，例如 `hipStreamWaitEvent` 和 OpenCL 的 `eventWaitList`。
- **性能剖析（Profiling）**：Event 对象承载了高精度的时间戳信息（Queued、Submitted、Start、End），为 `hipEventElapsedTime` 和 OpenCL Profiling 提供数据来源。
- **跨进程通信（IPC）**：通过 HIP 的 `hipEventInterprocess` 标志，Event 可以被导出为 IPC 句柄，实现多进程间的 GPU 执行同步。

从架构视角来看，Event 机制统一了 HIP 的 Stream 语义和 OpenCL 的 Command Queue 语义。无论是 HIP 用户还是 OpenCL 用户，其最终都落地到同一套 `amd::Event` / `amd::Command` 体系之上，通过 HSA/ROCr 的 Signal 与 AQL Barrier Packet 实现硬件级同步。

---

## 2. 设计目标

CLR Event 机制在设计上遵循了以下关键目标：

### 2.1 统一抽象

HIP 与 OpenCL 的前端 API 风格迥异，但底层运行时（ROCclr）要求共享同一套设备抽象。Event 机制必须同时满足：
- HIP 的隐式流内同步（in-order stream）与显式 `hipStreamWaitEvent`。
- OpenCL 的细粒度 `eventWaitList`、User Event、Marker 与 Barrier。

因此，`amd::Event` 被设计为与具体前端无关的基础类，而 `amd::Command` 继承自 `amd::Event`，将“同步原语”与“命令执行”天然绑定。

### 2.2 零开销（Zero-overhead）的常规路径

对于最常见的单流顺序执行场景，Event 机制应避免不必要的同步开销：
- **Direct Dispatch 模式**（HIP 默认）下，命令直接由调用线程提交到 GPU，不经过额外的队列线程转发，Event 的状态转换也尽量在硬件信号回调中完成，避免主机锁竞争。
- 只有在涉及**跨队列依赖**或**主机显式等待**时，才按需创建 Marker 或 HSA Signal。

### 2.3 精确的状态机与内存序

Event 状态转换涉及多线程并发（主机业务线程、队列线程、HSA 信号中断处理线程）。设计采用了 `std::atomic<int32_t>` 配合 Compare-And-Swap（CAS）操作，确保状态从 `CL_QUEUED` → `CL_SUBMITTED` → `CL_RUNNING` → `CL_COMPLETE` 的单调前进。

### 2.4 可扩展的硬件后端

CLR 需要同时支持 Linux 上的 HSA/ROCr 后端与 Windows 上的 PAL 后端。Event 的硬件抽象通过 `void* hw_event_`（在 ROCr 后端实际为 `roc::ProfilingSignal*`）实现，上层代码不直接依赖 HSA 头文件，从而保持后端无关性。

### 2.5 回调与通知机制

OpenCL 规范要求支持 `clSetEventCallback`，HIP 生态也有异步回调需求。Event 内部维护了一个无锁链表 `std::atomic<CallBackEntry*> callbacks_`，允许在状态到达特定阈值时触发用户回调，同时保证回调注册的线程安全。

---

## 3. 核心架构与概念

### 3.1 Event 类层次结构

CLR 中的 Event 相关类构成了一个清晰的分层体系：

```
RuntimeObject
└── amd::Event                         // 基础同步对象
    └── amd::Command                   // 带执行语义的事件
        ├── amd::UserEvent             // 主机端显式控制的 OpenCL User Event
        ├── amd::ClGlEvent             // OpenGL Interop Event
        ├── amd::OneMemoryArgCommand   // 单内存参数命令族
        │   ├── ReadMemoryCommand
        │   ├── WriteMemoryCommand
        │   ├── FillMemoryCommand
        │   ├── MapMemoryCommand
        │   └── SignalCommand
        ├── amd::TwoMemoryArgsCommand  // 双内存参数命令族
        │   └── CopyMemoryCommand
        ├── amd::NDRangeKernelCommand  // 内核启动命令
        ├── amd::NativeFnCommand       // 主机线程回调命令
        ├── amd::Marker                // 零操作同步命令（Barrier 核心）
        ├── amd::AccumulateCommand
        ├── amd::MigrateMemObjectsCommand
        └── ... (SVM, Batch, PerfCounter, ThreadTrace 等)
```

在 HIP 前端，还封装了更高层的对象：

```
hip::Event
├── hip::EventDD          // Direct Dispatch 优化版本
├── hip::EventMarker      // 继承自 amd::Marker，附加缓存作用域信息
└── hip::IPCEvent         // 支持跨进程共享的 Event
```

### 3.2 Event 与 Command 的“一体两面”

在 CLR 的设计哲学中，**“没有无 Event 的 Command，也没有无 Command 的 Event（纯 Marker 除外）”**。

- `amd::Event` 负责维护**状态**、**时间戳**、**回调**、**硬件信号**和**等待线程的通知**。
- `amd::Command` 在此基础上增加了**队列归属**（`HostQueue*`）、**等待列表**（`EventWaitList`）、**批次链接**（`next_`、`batch_head_`）以及**具体提交到硬件虚拟设备**的接口（`submit(device::VirtualDevice&)`）。

这种设计的巨大优势在于：**任何需要等待的操作，天然就有一个 Event 与之关联**。例如，当用户调用 `hipStreamWaitEvent` 时，底层只需要将这个 Event 加入后续 Command 的 `eventWaitList`，无需额外创建同步对象。

### 3.3 HostQueue 与 VirtualGPU

- **`amd::HostQueue`**：代表主机端的命令队列。它维护了一个命令批次链表（Direct Dispatch 模式下为 `head_` / `tail_`）或线程安全的并发队列（`ConcurrentLinkedQueue`，非 Direct Dispatch 模式）。HIP 的 `hip::Stream` 直接继承自 `amd::HostQueue`。
- **`device::VirtualDevice`**（如 `roc::VirtualGPU`）：代表与物理 GPU 对应的软件虚拟设备。`HostQueue` 通过 `virtualDevice_` 指针将命令最终提交到 GPU。
- **`HwQueueTracker`**：内嵌于 `VirtualGPU`，负责管理 HSA Signal 的分配、回收以及 Barrier Packet 的构建。

### 3.4 Hardware Signal（硬件信号）

在 ROCr 后端，真正的 GPU 完成通知依赖于 **HSA Signal**：
- 每个需要独立追踪完成状态的 Command，在提交时由 `HwQueueTracker::ActiveSignal()` 分配一个 `ProfilingSignal`（内含 `hsa_signal_t`）。
- 该 Signal 的句柄被存入 `amd::Event::hw_event_`。
- AQL Packet（如内核分发包）的 `completion_signal` 字段指向此 Signal。
- 当 GPU 硬件执行完该 Packet 后，会自动将 Signal 值减至 `0`。
- CLR 注册了一个 HSA 回调或轮询机制，在 Signal 归零后调用 `amd::Event::setStatus(CL_COMPLETE)`。

对于不需要独立追踪的命令（例如批次中间的命令），它们可能共享同一个 Signal 或根本不分配 Signal，从而节省资源。

### 3.5 Marker 与 Barrier 的统一

`amd::Marker` 是一个特殊的 `Command`，它没有任何实际的 GPU 计算或内存操作，唯一的目的是**制造一个执行顺序上的边界**。

在 ROCr 后端，Marker 被翻译为 **HSA AQL Barrier-AND Packet**。该 Packet 的 `dep_signal` 列表可以引用多个前置 Signal（来自同队列的前序命令，或来自其他队列的外部事件）。GPU 硬件保证：在该 Barrier Packet 完成之前，其依赖的所有 Signal 都必须归零；而在 Barrier Packet 完成之后，后续的所有 Packet 才能开始执行。

**在 CLR 的当前架构中，Marker 与 Barrier 本质上是同一操作**。这是因为 CLR 的队列默认是 in-order 的，加上 Barrier-AND Packet 已经足以表达“等待某些事件完成后再继续”的语义。OpenCL 的 `clEnqueueMarkerWithWaitList` 和 `clEnqueueBarrierWithWaitList` 在底层都映射为 `amd::Marker`。

### 3.6 Event 类型总览

在 CLR 中，**Event 是一个广义概念**。从最底层的硬件 Signal，到中间的 `amd::Event` 状态对象，再到承载具体 GPU 操作的 `amd::Command`，以及 HIP/OpenCL 前端的各类封装，整个体系涵盖了数十种不同的类型。下表按照软件栈层次进行了全面梳理。

> **说明**：`amd::Command` 继承自 `amd::Event`，因此下表中所有 Command 子类本质上也是一种 Event（具备状态、可等待、可剖析）。表格中的“类别”列用于区分其语义角色：`Event 基类` 与 `显式事件` 属于纯同步对象；`Command 基类` 与 `Command 子类` 属于带执行语义的事件；`HIP 封装` 是前端面向用户的对象；`Backend 抽象` 则对应硬件信号与设备调度结构。

#### Platform 层 — 基础 Event 与显式事件（`rocclr/platform/`）

| 类型名称 | 父类 | 类别 | 说明 |
|---------|------|------|------|
| `amd::Event` | `RuntimeObject` | Event 基类 | 所有同步对象的根基。封装状态机、时间戳、回调链表、硬件信号句柄 `hw_event_`。 |
| `amd::Command` | `amd::Event` | Command 基类 | 所有可提交 GPU 操作的抽象基类。增加队列归属、等待列表 `eventWaitList_`、批次链接与 `submit()` 接口。 |
| `amd::UserEvent` | `amd::Command` | 显式事件 | OpenCL 用户事件。由主机端通过 `clCreateUserEvent` 创建，并显式调用 `clSetUserEventStatus` 触发状态变更。 |
| `amd::ClGlEvent` | `amd::Command` | 显式事件 | OpenCL-OpenGL 互操作围栏同步对象。用于在 OpenCL 与 OpenGL 共享对象之间建立执行顺序。 |
| `amd::Marker` | `amd::Command` | 同步命令 | 零操作同步命令。仅用于制造执行边界，在 ROCr 后端被翻译为 AQL Barrier-AND Packet。 |

#### Platform 层 — 具体 Command 子类（`rocclr/platform/`）

| 类型名称 | 父类 | 类别 | 说明 |
|---------|------|------|------|
| `amd::ReadMemoryCommand` | `OneMemoryArgCommand` | 内存命令 | 从设备内存（Buffer/Image）读取数据到主机内存。 |
| `amd::WriteMemoryCommand` | `OneMemoryArgCommand` | 内存命令 | 从主机内存写入数据到设备内存（Buffer/Image）。 |
| `amd::FillMemoryCommand` | `OneMemoryArgCommand` | 内存命令 | 以指定模式填充设备内存区域。 |
| `amd::CopyMemoryCommand` | `TwoMemoryArgsCommand` | 内存命令 | 设备内部内存拷贝（Buffer ↔ Buffer，Image ↔ Image 等）。 |
| `amd::CopyMemoryP2PCommand` | `CopyMemoryCommand` | 内存命令 | 点对点（Peer-to-Peer）跨设备内存拷贝。 |
| `amd::MapMemoryCommand` | `OneMemoryArgCommand` | 内存命令 | 映射设备内存对象到主机可访问地址空间。 |
| `amd::UnmapMemoryCommand` | `OneMemoryArgCommand` | 内存命令 | 解除先前映射的设备内存对象。 |
| `amd::MigrateMemObjectsCommand` | `amd::Command` | 内存命令 | 将一组内存对象迁移到目标设备，以优化访问局部性。 |
| `amd::StreamOperationCommand` | `OneMemoryArgCommand` | 流内存命令 | 流式等待值（wait-value）或写值（write-value）操作，用于细粒度设备内存信号。 |
| `amd::BatchMemoryOperationCommand` | `amd::Command` | 流内存命令 | 批量执行多个流内存操作（如多个 wait/write），合并为单次提交以减少开销。 |
| `amd::BatchCopyMemoryCommand` | `amd::Command` | 流内存命令 | 批量执行多个 Buffer-to-Buffer 拷贝，合并为单次提交。 |
| `amd::SignalCommand` | `OneMemoryArgCommand` | 流内存命令 | 在设备内存指定偏移处写入一个 32-bit 标记值，用于轻量级信号通知。 |
| `amd::NDRangeKernelCommand` | `amd::Command` | 计算命令 | NDRange 内核启动命令。支持普通内核、合作组（cooperative groups）及多设备联合启动。 |
| `amd::NativeFnCommand` | `amd::Command` | 计算命令 | 在主机队列线程上执行一个原生主机函数回调（Host Callback）。 |
| `amd::ExternalSemaphoreCmd` | `amd::Command` | 计算命令 | 对外部信号量（如 Vulkan/DX12 semaphore）执行等待或信号操作。 |
| `amd::AccumulateCommand` | `amd::Command` | 分析命令 | 累积前序命令的剖析时间戳与内核名称，用于汇总统计。 |
| `amd::PerfCounterCommand` | `amd::Command` | 分析命令 | 开始或结束性能计数器采集。 |
| `amd::ThreadTraceCommand` | `amd::Command` | 分析命令 | 控制线程追踪的 begin/end/pause/resume。 |
| `amd::ThreadTraceMemObjectsCommand` | `amd::Command` | 分析命令 | 为线程追踪机制绑定内存对象。 |
| `amd::AcquireExtObjectsCommand` | `ExtObjectsCommand` | 外部对象命令 | 从外部 API（如 OpenGL）获取共享对象的访问权。 |
| `amd::ReleaseExtObjectsCommand` | `ExtObjectsCommand` | 外部对象命令 | 将共享对象释放回外部 API。 |
| `amd::SvmFreeMemoryCommand` | `amd::Command` | SVM 命令 | 释放一组 SVM 指针。支持通过用户回调异步执行释放。 |
| `amd::SvmCopyMemoryCommand` | `amd::Command` | SVM 命令 | 在两个 SVM 指针之间执行拷贝。 |
| `amd::SvmFillMemoryCommand` | `amd::Command` | SVM 命令 | 对 SVM 区域执行模式填充。 |
| `amd::SvmMapMemoryCommand` | `amd::Command` | SVM 命令 | 映射 SVM 共享缓冲区以供主机访问。 |
| `amd::SvmUnmapMemoryCommand` | `amd::Command` | SVM 命令 | 解除 SVM 共享缓冲区的映射。 |
| `amd::SvmPrefetchAsyncCommand` | `amd::Command` | SVM 命令 | 异步预取 SVM 内存到指定设备或主机。 |
| `amd::SvmPrefetchBatchAsyncCommand` | `amd::Command` | SVM 命令 | 批量异步预取多个 SVM 区间到各自的目标设备。 |
| `amd::VirtualMapCommand` | `amd::Command` | 虚拟内存命令 | 为指针执行虚拟内存映射或解除映射（用于 VA 预留机制）。 |
| `amd::MakeBuffersResidentCommand` | `amd::Command` | 虚拟内存命令 | 使一组缓冲区常驻（resident），暴露总线地址以支持 P2P 访问。 |

#### HIP 层 — Event 封装与扩展（`hipamd/src/`）

| 类型名称 | 父类/关联基类 | 类别 | 说明 |
|---------|--------------|------|------|
| `hip::Event` | — | HIP 封装 | 标准 HIP Event 包装器。内部持有 `amd::Event*`（通常指向一个 Marker 的 event），提供 `record`、`query`、`synchronize` 和 `elapsedTime`。 |
| `hip::EventDD` | `hip::Event` | HIP 封装 | Direct Dispatch 优化版 Event。绕过部分软件路径，直接通过硬件事件查询与时间戳采集，降低主机开销。 |
| `hip::IPCEvent` | `hip::Event` | HIP 封装 | 跨进程事件。基于 POSIX 共享内存（`ihipIpcEventShmem_t`）维护一个 32 槽 GPU Signal 环形缓冲区，实现多进程间的执行同步。 |
| `hip::EventMarker` | `amd::Marker` | HIP 封装 | HIP 专用的 Marker 命令。在 `amd::Marker` 基础上增加缓存作用域（cache scope）控制和 profiling 标志，用于 `hipEventRecord`。 |
| `hip::StreamCallback` | — | 回调封装 | 流回调抽象基类。定义了回调在队列线程上的执行接口。 |
| `hip::StreamAddCallback` | `StreamCallback` | 回调封装 | 包装传统的 `hipStreamCallback_t` 回调。 |
| `hip::LaunchHostFuncCallback` | `StreamCallback` | 回调封装 | 包装 `hipHostFn_t` 回调，用于 `hipLaunchHostFunc`。 |
| `hip::GraphEventWaitNode` | `hip::GraphNode` | 图节点 | HIP Graph 中的事件等待节点。图执行时在该点插入对指定 `hipEvent_t` 的等待。 |
| `hip::GraphEventRecordNode` | `hip::GraphNode` | 图节点 | HIP Graph 中的事件记录节点。图执行时在指定流中记录 `hipEvent_t`。 |
| `hip::FreeAsyncCommand` | `amd::Command` | 图命令 | 延迟释放命令。用于 `hipFreeAsync`，避免与图内存分配发生竞争。 |
| `hip::VirtualMemAllocNode` | `amd::VirtualMapCommand` | 图命令 | 图内部命令：先分配物理内存，再将其映射到预留的虚拟地址（VA）中。 |
| `hip::VirtualMemFreeNode` | `amd::VirtualMapCommand` | 图命令 | 图内部命令：先解除 VA 映射，再释放底层物理内存分配。 |

#### OpenCL 层

OpenCL 前端**没有定义独立的 C++ Event 子类**，而是直接复用 Platform 层的 `amd::Event`、`amd::Command`、`amd::UserEvent` 和 `amd::ClGlEvent`。`cl_event` 在实现层面就是一个指向 `amd::Event` 实例的不透明句柄。

#### Device / Backend 层 — 硬件 Signal 与调度抽象

| 类型名称 | 所属/父类 | 类别 | 说明 |
|---------|----------|------|------|
| `amd::device::Signal` | `HeapObject` | Signal 基类 | 后端无关的信号抽象。定义等待条件（equal/less-than/greater-than 等）与状态查询接口。 |
| `amd::roc::Signal` | `device::Signal` | ROCr Signal | Linux ROCr 后端的 HSA Signal 包装器。直接封装 `hsa_signal_t`，提供原子等待与加载操作。 |
| `amd::pal::Signal` | `device::Signal` | PAL Signal | Windows PAL 后端的信号实现。包装 `amd_signal_t` 与 `Util::Event`。 |
| `amd::roc::ProfilingSignal` | `ReferenceCountedObject` | ROCr Signal | 带引用计数的 HSA Signal，缓存时间戳数据（queued/submitted/start/end），用于 profiling 分发。 |
| `amd::roc::Timestamp` | `ReferenceCountedObject` | ROCr 计时 | 追踪单个命令的 GPU 开始/结束时间。可聚合多个 `ProfilingSignal` 的时间戳。 |
| `amd::roc::VirtualGPU::HwQueueTracker` | `VirtualGPU` 内嵌 | 队列追踪 | 管理 HSA Signal 池（`signal_list_`）与外部信号列表（`external_signals_`）。负责为每个提交分配 `ActiveSignal` 并构建 Barrier Packet。 |
| `amd::roc::AmdEvent` | — | 设备调度 | ROCr 设备端调度结构。用于设备入队（device enqueue）场景，描述子队列中的事件状态与计数器。 |
| `amd::roc::AmdAqlWrap` | — | 设备调度 | ROCr AQL 包包装器。为设备端调度附加状态机信息。 |
| `amd::roc::SchedulerParam` | — | 设备调度 | ROCr 调度核函数的参数块，用于控制子队列分发。 |
| `amd::pal::AmdEvent` | — | 设备调度 | PAL 设备端事件结构。功能与 ROCr 的 `AmdEvent` 对应，用于 PAL 后端的设备调度与 profiling。 |
| `amd::pal::AmdAqlWrap` | — | 设备调度 | PAL AQL 包包装器。功能与 ROCr 的 `AmdAqlWrap` 对应。 |
| `amd::pal::SchedulerParam` | — | 设备调度 | PAL 调度核函数的参数块。 |
| `pal::GpuEvent` | — | PAL 描述符 | PAL GPU 事件描述符。包含事件 ID、修改位与引擎 ID，用于 PAL 命令缓冲区提交。 |
| `amd::Device::HwEventPatch` | `Device` 内嵌 | 硬件补丁 | 描述在 AQL 包或 Barrier 包中需要补丁写入硬件事件句柄的位置。用于 Direct Dispatch 下的快速路径。 |

---

## 4. 实现机制详解

### 4.1 Event 状态机

`amd::Event` 内部维护一个原子变量 `status_`，其状态值遵循 OpenCL 规范定义（HIP 内部复用了同样的枚举值）：

| 状态值 | 宏定义 | 含义 |
|--------|--------|------|
| 3 | `CL_QUEUED` | 命令已创建并加入主机队列 |
| 2 | `CL_SUBMITTED` | 命令已从主机队列取出，提交给 HSA/PAL |
| 1 | `CL_RUNNING` | 命令开始在 GPU 上执行 |
| 0 | `CL_COMPLETE` | 命令执行完毕 |
| <0 | 错误码 | 执行过程中发生错误 |

状态转换规则：
- **只允许数值递减**（从 3 → 0），不允许回退。
- 使用 `compare_exchange_strong` 进行原子更新，确保多线程竞争下的正确性。
- 当状态到达 `CL_COMPLETE` 时，触发一连串收尾动作：
  1. 释放 `eventWaitList` 中对其他 Event 的引用（`releaseResources`）。
  2. 调用用户注册的回调（`processCallbacks`）。注意：HIP 与 OpenCL 的回调触发时机略有不同，HIP 在状态 CAS 之前触发，OpenCL 在之后。
  3. 通过 `Monitor::notifyAll()` 唤醒所有在 `awaitCompletion()` 中阻塞的主机线程。
  4. 减少 Event 的引用计数（`release()`），如果降至零则自动销毁。

### 4.2 Direct Dispatch 与 Threaded Queue

CLR 支持两种命令提交流程，它们对 Event 的生命周期有直接影响：

#### Direct Dispatch（HIP 默认）

在此模式下，`Command::enqueue()` 直接由用户调用线程执行：
1. 将 Command 加入 `HostQueue` 的批次链表（`head_` / `tail_`）。
2. 如果命令带有 `eventWaitList`，遍历列表并调用 `event->notifyCmdQueue()`，确保被等待的事件产生对应的硬件 Signal。
3. 调用 `FormSubmissionBatch()` 构建提交批次。
4. 直接调用 `command->submit(*vdev())`，将 AQL Packet 写入 HSA Queue。
5. 如果是 Marker 或批次已满，调用 `FlushSubmissionBatch()`，触发 HSA Doorbell。

**特点**：低延迟、无锁（大多数情况下）、Event 状态转换路径短。

#### Threaded Queue（OpenCL 传统模式）

在此模式下，`HostQueue` 内部运行一个独立的 `loop()` 线程：
1. 用户调用 `enqueue()` 将 Command 放入 `ConcurrentLinkedQueue`。
2. `loop()` 线程不断出队。
3. 对于跨队列依赖，`loop()` 线程会调用 `awaitCompletion()` 阻塞等待外部 Event 完成。
4. 外部 Event 完成后，`loop()` 线程继续将当前 Command 提交到 `VirtualDevice`。

**特点**：适合复杂的跨队列依赖场景，但引入了线程上下文切换和锁竞争。

### 4.3 跨队列同步的实现

当 Command A 在 Queue 1 执行，而 Command B 在 Queue 2 执行且需要等待 A 完成时，CLR 采用以下策略：

1. **Signal 传播**：Command A 提交时分配了 HSA Signal `S_A`。
2. **外部 Signal 注册**：在 Command B 提交到 `VirtualGPU` 时，`roc::VirtualGPU::profilingBegin()` 遍历 B 的 `eventWaitList`，将 A 的 Signal `S_A` 加入 `HwQueueTracker::external_signals_` 列表。
3. **Barrier-AND Packet 注入**：Command B 在写入其 AQL Packet 之前，会先写入一个 Barrier-AND Packet，其 `dep_signal` 数组包含 `S_A`。
4. **硬件等待**：ROCr 驱动保证 Barrier-AND Packet 在 `S_A` 归零前不会通过，从而确保 Command B 及其后续操作严格等待 Command A。

如果 Command A 尚未分配硬件 Signal（例如 A 还在主机队列中未刷新），CLR 会先在 A 的队列中插入一个 `amd::Marker`（通过 `notifyCmdQueue()`），强制 A 的队列刷新并产生 Signal。

### 4.4 主机等待策略（CPU Wait）

当主机线程调用 `hipEventSynchronize` 或 `clWaitForEvents` 时，CLR 采用分层等待策略：

1. **快速路径**：检查 `amd::Event::hw_event_` 是否已分配。如果已分配，调用 `Device::IsHwEventReady(..., wait=true)`，后者通过 `hsa_signal_wait_scacquire` 或自旋/退让（spin/yield）策略在**不持有软件锁**的情况下等待硬件 Signal。这是最轻量的路径。
2. **通知路径**：如果该 Event 没有硬件 Signal（例如命令尚未刷新到 GPU），调用 `Event::notifyCmdQueue(kCpuWait)`。这会在 Event 所属的队列中插入一个内部 Marker，强制队列刷新。Marker 提交后会分配 Signal。
3. **软件锁路径**：如果硬件 Signal 仍然不可用（例如某些特殊的 CPU-wait Marker），主机线程进入 `Event::awaitCompletion()`，在 `Monitor`（条件变量）上睡眠，直到 `setStatus(CL_COMPLETE)` 被调用并触发 `notifyAll()`。

### 4.5 Profiling 时间戳

`amd::Event::profilingInfo_` 维护四个关键时间点：
- `queued_`：`Command` 对象创建的时间。
- `submitted_`：命令从主机队列提交到 `VirtualDevice` 的时间。
- `start_`：GPU 硬件实际开始执行的时间（通过 HSA Signal 的 `start` 回调或 ATOMIC_ADD Packet 捕获）。
- `end_`：GPU 硬件执行完成的时间（通过 HSA Signal 的 `completion` 回调捕获）。

HIP 的 `hipEventElapsedTime` 计算的是两个 `hip::Event` 之间 `end_` 与 `start_` 的差值（单位为毫秒）。

---

## 5. 关键流程图

### 5.1 Event 生命周期总览

```mermaid
graph TD
    A["前端 API 调用<br/>hipEventCreate / clCreateUserEvent"] --> B{"分配策略"}
    B -->|"普通 HIP Event"| C["创建 hip::Event"]
    B -->|"Direct Dispatch"| D["创建 hip::EventDD"]
    B -->|"跨进程"| E["创建 hip::IPCEvent"]
    B -->|"OpenCL"| F["创建 amd::UserEvent<br/>或 amd::Marker"]
    C --> G["注册到全局 eventSet"]
    D --> G
    E --> G
    F --> H["绑定到 amd::Context<br/>初始化 status_ = CL_QUEUED"]
    G --> I["Event 处于可记录/可等待状态"]
    H --> I
    I --> J["通过 hipEventRecord 或 enqueue<br/>将 Event 与 Command 关联"]
    J --> K["Command 提交到 GPU<br/>分配 HSA Signal -> hw_event_"]
    K --> L{"GPU 执行完成?"}
    L -->|"HSA Signal 归零"| M["异步回调触发<br/>setStatus(CL_COMPLETE)"]
    M --> N["释放 eventWaitList 引用"]
    N --> O["执行用户回调"]
    O --> P["Monitor::notifyAll()<br/>唤醒主机等待线程"]
    P --> Q["引用计数减一<br/>可能自动销毁"]
    Q --> R["hipEventDestroy / clReleaseEvent<br/>确认释放内存"]
    I --> S["前端显式销毁"]
    S --> R
```

### 5.2 hipEventRecord 内部流程

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户线程"
    participant HIP as "hip::Event"
    participant Stream as "hip::Stream<br/>(amd::HostQueue)"
    participant VDev as "roc::VirtualGPU"
    participant HSA as "HSA Queue"

    User->>HIP: "hipEventRecord(event, stream)"
    HIP->>HIP: "检查 event 是否已关联旧 command<br/>若是，先释放旧引用"
    HIP->>HIP: "recordCommand()<br/>创建 hip::EventMarker 对象"
    Note over HIP: "EventMarker 继承自 amd::Marker<br/>userVisible = true, marker_ts_ = true"
    HIP->>Stream: "enqueueRecordCommand(marker)"
    Stream->>Stream: "将 Marker 加入批次链表<br/>FormSubmissionBatch()"
    Stream->>VDev: "marker.submit(virtualDevice)"
    VDev->>VDev: "submitMarker()<br/>分配 ActiveSignal -> hw_event_"
    VDev->>HSA: "写入 AQL Barrier-AND Packet<br/>signal = completion_signal"
    VDev->>HSA: "必要时调用 FlushSubmissionBatch()<br/>写 Doorbell"
    HSA-->>VDev: "GPU 完成所有前置工作<br/>Barrier Packet 完成"
    VDev->>HIP: "HSA 回调 -> event.setStatus(CL_COMPLETE)"
    HIP->>User: "event 变为可用状态"
```

### 5.3 hipEventSynchronize 等待流程

```mermaid
graph TD
    A["hipEventSynchronize(event)"] --> B["获取 event 关联的 amd::Event"]
    B --> C{"event_ == nullptr ?"}
    C -->|"是"| D["直接返回 hipSuccess<br/>Event 尚未记录或已完成"]
    C -->|"否"| E["读取当前 status"]
    E --> F{"status <= CL_COMPLETE ?"}
    F -->|"是"| D
    F -->|"否"| G["尝试硬件快速路径"]
    G --> H["Device::IsHwEventReady<br/>event.HwEvent(), wait=true"]
    H --> I{"Signal 已归零 ?"}
    I -->|"是"| D
    I -->|"否"| J["检查 NotifyEvent"]
    J --> K["notify_event_ 的 HwEvent 是否就绪 ?"]
    K -->|"是"| D
    K -->|"否"| L["进入软件等待路径"]
    L --> M["event.notifyCmdQueue(kCpuWait)<br/>在所属队列插入 Marker 强制刷新"]
    M --> N["event.awaitCompletion()<br/>Monitor::wait() 睡眠"]
    N --> O{"GPU 完成触发 setStatus"}
    O --> P["Monitor::notifyAll()"]
    P --> Q["线程唤醒，返回 hipSuccess"]
```

### 5.4 hipStreamWaitEvent 跨流同步流程

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户线程"
    participant Stream as "目标 hip::Stream<br/>(等待方)"
    participant E as "源 hip::Event<br/>(被等待方)"
    participant QSrc as "源 Event 所属队列"
    participant VDev as "roc::VirtualGPU<br/>(目标流)"
    participant HSA as "HSA Queue<br/>(目标流)"

    User->>Stream: "hipStreamWaitEvent(stream, event, flags)"
    Stream->>E: "event.streamWait(this, flags)"
    E->>E: "检查 event.ready()"
    alt "Event 已完成"
        E-->>Stream: "直接返回 hipSuccess"
    else "Event 未完成"
        E->>QSrc: "event_->notifyCmdQueue()"
        Note over QSrc: "确保源 Event 有 HW Signal<br/>若无，插入 Marker 刷新"
        E->>Stream: "streamWaitCommand(command, stream)"
        Note over Stream: "创建 amd::Marker<br/>eventWaitList = [event_]"
        Stream->>Stream: "command->enqueue()"
        Stream->>VDev: "command->submit()"
        VDev->>VDev: "profilingBegin()<br/>将源 Event 的 hw_event_<br/>加入 external_signals_"
        VDev->>HSA: "先写 Barrier-AND Packet<br/>dep_signal = 源 Signal"
        VDev->>HSA: "后续 Packet 依赖此 Barrier"
        HSA-->>VDev: "源 Signal 归零后<br/>Barrier 通过"
        VDev->>Stream: "Marker 完成<br/>目标流继续执行"
    end
    Stream-->>User: "返回 hipSuccess"
```

### 5.5 Barrier/Marker 提交到 GPU 的硬件流程

```mermaid
graph LR
    A["amd::Marker::submit<br/>VirtualGPU::submitMarker"] --> B{"CpuWaitRequested()?"}
    B -->|"是"| C["flush(batch_head)<br/>主机线程阻塞等待 GPU 完成"]
    B -->|"否"| D["构建 AQL Barrier-AND Packet"]
    D --> E{"releaseFlags 类型"}
    E -->|"kCacheStateIgnore"| F["使用 kBarrierVendorPacketNopScopeHeader<br/>无缓存作用域"]
    E -->|"默认 / System Scope"| G["使用 kBarrierVendorPacketHeader<br/>HSA_FENCE_SCOPE_SYSTEM"]
    F --> H["填充 dep_signal 数组<br/>同队列前序 Signal + 外部 Signal"]
    G --> H
    H --> I["写入 AQL Ring Buffer"]
    I --> J["更新 Write Index<br/>发送 Doorbell"]
    J --> K["GPU 调度器识别 Barrier Packet"]
    K --> L["等待所有 dep_signal 归零"]
    L --> M["Barrier Packet 完成<br/>后续 Packet 开始执行"]
    M --> N["HSA Signal 回调<br/>setStatus(CL_COMPLETE)"]
```

### 5.6 Event 状态转换时序

```mermaid
stateDiagram-v2
    [*] --> QUEUED: "Command 创建"
    QUEUED --> SUBMITTED: "enqueue() / submit()<br/>主机队列提交到 VirtualDevice"
    SUBMITTED --> RUNNING: "GPU 开始执行<br/>HSA Signal start 回调"
    RUNNING --> COMPLETE: "GPU 执行结束<br/>HSA Signal completion 回调"
    COMPLETE --> [*]: "release() 引用计数归零"
    QUEUED --> ERROR: "提交失败或非法参数"
    SUBMITTED --> ERROR: "GPU 页错误或非法访问"
    RUNNING --> ERROR: "GPU 执行异常终止"
    ERROR --> [*]: "错误处理与清理"
    COMPLETE: "CL_COMPLETE<br/>释放资源/回调/唤醒线程"
    QUEUED: "CL_QUEUED<br/>仅存在于主机内存"
    SUBMITTED: "CL_SUBMITTED<br/>已写入 HSA Queue"
    RUNNING: "CL_RUNNING<br/>GPU 正在执行"
    ERROR: "负值错误码<br/>如 CL_OUT_OF_RESOURCES"
```

---

## 6. Event、同步与 Barrier 的关系

在 GPU 编程模型中，Event、Synchronization 和 Barrier 是三个紧密耦合但又层次分明的概念。在 CLR 中，它们的关系可以归纳为：

### 6.1 Event 是“状态”，同步是“动作”，Barrier 是“机制”

- **Event（状态）**：回答“某个操作完成了吗？”这个问题。它是一个被动对象，通过 `status_` 和 `hw_event_` 暴露完成状态。
- **同步（动作）**：回答“如何让两个操作按顺序发生？”这个问题。它是一个主动行为，可以是主机等待（`hipEventSynchronize`）、流等待（`hipStreamWaitEvent`）或命令等待列表（`eventWaitList`）。
- **Barrier（机制）**：回答“GPU 硬件如何保证顺序？”这个问题。它是实现同步的底层硬件机制，具体表现为 HSA AQL Barrier-AND Packet。

**三者的协作链**：
> 用户发起同步动作（如 `hipStreamWaitEvent`）→ 运行时创建/复用 Event（追踪被等待操作的状态）→ 运行时通过 Barrier 机制（Marker/Barrier Packet）让 GPU 硬件强制执行顺序。

### 6.2 流内隐式同步 vs 流间显式同步

- **流内隐式同步**：在 HIP 中，同一个 Stream 内的命令天然按入队顺序执行。CLR 通过 in-order 的 HSA Queue 和批次提交保证这一点，**不需要为每个命令创建独立的 Event 或 Barrier**。只有在需要精确 Profiling 或主机等待时，才为特定命令分配 Signal。
- **流间显式同步**：当 Stream A 需要等待 Stream B 的某个事件时，CLR 必须在 Stream A 的 HSA Queue 中插入一个 Barrier Packet，让其 `dep_signal` 指向 Stream B 中对应命令的 Signal。这是 Event 与 Barrier 结合最典型的场景。

### 6.3 Barrier 与 Marker 的等价性

在传统的 OpenCL 1.x 中，Marker（标记点）和 Barrier（屏障）有细微区别：
- **Marker**：等待一个事件列表完成后，本身变成完成状态，但**不阻塞**队列中后续命令。
- **Barrier**：等待一个事件列表完成后，本身变成完成状态，并且**阻塞**队列中后续命令直到 Barrier 完成。

然而，在 CLR 的当前架构中，由于队列是 in-order 的，Marker Packet 一旦被提交到 HSA Queue，它天然就会阻塞后续 Packet（因为 HSA Queue 顺序消费）。因此：

> **CLR 中 Marker 与 Barrier 是同一实现**。`clEnqueueMarkerWithWaitList` 和 `clEnqueueBarrierWithWaitList` 都映射到 `amd::Marker`，其底层都生成 Barrier-AND Packet。唯一的区别可能在于前端 API 是否暴露 `cl_event` 给用户查询。

### 6.4 主机同步 vs 设备同步

| 维度 | 主机同步（Host-side Sync） | 设备同步（Device-side Sync） |
|------|--------------------------|---------------------------|
| 代表 API | `hipEventSynchronize`、`clWaitForEvents` | `hipStreamWaitEvent`、`eventWaitList` |
| 阻塞对象 | 主机线程（CPU） | GPU 硬件执行管线 |
| 实现机制 | `Monitor` 条件变量、`hsa_signal_wait` | HSA Barrier-AND Packet、`dep_signal` |
| 性能影响 | 主机线程挂起/唤醒开销 | 零主机开销，纯硬件调度 |
| 适用场景 | 需要 CPU 读取 GPU 结果、同步多线程 | 流间依赖、DAG 执行图 |

CLR 的设计尽量让设备同步走硬件路径，只有当主机必须介入（如 CPU 等待或 Signal 尚未分配）时才退回到软件锁路径。

---

## 7. Event 机制的用法与适用场景

### 7.1 HIP API 用法

#### 基本生命周期

```cpp
// 1. 创建 Event
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

// 2. 记录 Event（插入到 Stream 中）
hipEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
hipEventRecord(stop, stream);

// 3. 同步等待（主机阻塞）
hipEventSynchronize(stop);

// 4. 计算耗时
float elapsed_ms;
hipEventElapsedTime(&elapsed_ms, start, stop);

// 5. 销毁
hipEventDestroy(start);
hipEventDestroy(stop);
```

#### 流间同步

```cpp
hipStream_t streamA, streamB;
hipStreamCreate(&streamA);
hipStreamCreate(&streamB);
hipEvent_t event;
hipEventCreate(&event);

// Stream A 执行一些工作
kernelA<<<..., streamA>>>(...);
hipEventRecord(event, streamA);

// Stream B 在继续前等待 Stream A 的 event
hipStreamWaitEvent(streamB, event, 0);
kernelB<<<..., streamB>>>(...);  // 保证在 kernelA 完成后才开始
```

#### 跨进程 Event（IPC）

```cpp
// 进程 A：创建并导出 Event
hipEvent_t ipcEvent;
hipEventCreateWithFlags(&ipcEvent, hipEventInterprocess | hipEventDisableTiming);
hipIpcEventHandle_t handle;
hipIpcGetEventHandle(&handle, ipcEvent);
// 通过 socket/shared memory 将 handle 发送给进程 B

// 进程 B：导入并使用 Event
hipIpcEventHandle_t handle;  // 从进程 A 接收
hipEvent_t importedEvent;
hipIpcOpenEventHandle(&importedEvent, handle);
hipStreamWaitEvent(localStream, importedEvent, 0);
```

### 7.2 OpenCL API 用法

```cpp
// 创建 User Event（主机端控制）
cl_event userEvent = clCreateUserEvent(context, &err);

// 将 User Event 作为依赖提交到队列
cl_event kernelEvent;
clEnqueueNDRangeKernel(queue, kernel, ..., 1, &userEvent, &kernelEvent);

// 主机端在某个时机手动触发 Event
clSetUserEventStatus(userEvent, CL_COMPLETE);

// 等待单个或多个 Event
clWaitForEvents(1, &kernelEvent);

// Marker / Barrier（在当前 CLR 中行为一致）
cl_event markerEvent;
clEnqueueMarkerWithWaitList(queue, 1, &kernelEvent, &markerEvent);
clEnqueueBarrierWithWaitList(queue, 1, &kernelEvent, NULL);
```

### 7.3 适用场景总结

| 场景 | 推荐做法 | 底层机制 |
|------|---------|---------|
| 测量单个 Kernel 执行时间 | `hipEventRecord` 前后各一个 Event | Marker + Profiling Signal |
| 测量多 Stream 流水线总耗时 | 在首尾 Stream 记录 Event | 同上 |
| Stream A 产出数据，Stream B 消费 | `hipStreamWaitEvent` | Barrier-AND + external_signal |
| 主机线程等待 GPU 某一步完成 | `hipEventSynchronize` | `hsa_signal_wait` / `Monitor` |
| 多进程共享 GPU 执行顺序 | `hipIpcEventHandle` + `hipStreamWaitEvent` | 共享内存 Signal |
| OpenCL 主机端控制异步流 | `clCreateUserEvent` + `clSetUserEventStatus` | UserEvent 手动 setStatus |
| 构建复杂 DAG 执行图 | `eventWaitList` 组合 | 多 external_signal + Barrier |

---

## 8. 性能考量与最佳实践

### 8.1 避免过度创建 Event

每个 `hip::Event`（尤其是非 `EventDD` 模式）都涉及主机内存分配、引用计数管理和可能的 HSA Signal 分配。在高频调用的循环中，应复用 Event 对象（池化），而非每次都 `create/destroy`。

### 8.2 谨慎使用 `hipEventSynchronize`

`hipEventSynchronize` 会阻塞主机线程。如果只是为了流间同步，优先使用 `hipStreamWaitEvent`，让同步在 GPU 硬件层面完成，不占用 CPU 资源。

### 8.3 Direct Dispatch 下的批次行为

在 Direct Dispatch 模式下，命令是批量提交的。`hipEventRecord` 会隐式刷新当前批次（因为 Marker 需要产生 Signal），这可能导致比预期更频繁的 Doorbell 写入。如果连续记录大量 Event，可能会增加主机开销。建议在高吞吐量场景下减少不必要的 Event 记录。

### 8.4 IPC Event 的局限性

IPC Event 依赖共享内存和进程间信号可见性。它仅保证执行顺序，**不保证内存一致性**（除非配合显式的内存拷贝或统一虚拟寻址）。在多进程场景下，通常需要 IPC Event + `hipMemcpy`/`hipMemcpyPeer` 组合使用。

### 8.5 Profiling 与并发

`hipEventElapsedTime` 要求两个 Event 都开启了 Timing 支持。如果创建 Event 时使用了 `hipEventDisableTiming`，则无法获取时间戳。此外，在大量并发 Stream 中，HSA Signal 池的容量可能成为瓶颈（虽然通常很大）。

### 8.6 错误处理

Event 可能因 GPU 异常（如页错误、非法指令）而进入错误状态。主机等待 API（`hipEventSynchronize`）在错误状态下也会返回，但错误码需要通过 `hipGetLastError` 获取。长时间挂起的 Event 应设置超时或检查设备健康状态。

---

## 9. Event 机制与 AMD KMD（amdgpu）的关系

在前面的章节中，我们重点描述了 CLR Event 机制在用户态的实现。然而，GPU 是一个受内核态驱动严格管理的硬件资源，CLR 的 Event 最终必须依赖内核驱动才能与 GPU 硬件交互。本节将系统梳理 CLR Event 与 Linux AMDGPU 内核驱动（KMD）之间的依赖关系。

### 9.1 核心结论：CLR 不直接调用 KMD，但间接深度依赖

通过对 CLR 源码（`rocclr/`、`hipamd/`）的全面审计，可以得出以下明确结论：

- **CLR 自身不包含任何直接调用 amdgpu/KFD 的代码**。在 CLR 源码中搜索 `ioctl`、`libhsakmt`、`/dev/kfd`、`/dev/dri`、`mmap`（针对 GPU 寄存器）等关键词，结果均为零或仅有注释提及。
- CLR 将所有的内核边界交互**完全委托给 ROCr Runtime**（`libhsa-runtime64.so`）。ROCr 是 CLR 与内核之间的唯一中介层。
- 然而，CLR 的 Event 机制（AQL Queue、HSA Signal、Doorbell、内存分配）**在功能上深度依赖 AMDGPU/KFD 提供的内核服务**。没有 KFD，ROCr 无法创建队列和信号，CLR 的 Event 体系将无法运转。

因此，CLR 与 KMD 的关系是：**逻辑解耦、物理依赖**。CLR 通过 ROCr API 抽象了底层硬件，但这些 API 的实体完全由 KFD 和 amdgpu 内核模块支撑。

### 9.2 软件栈全景

CLR Event 机制从用户态到内核态的完整调用栈如下：

```mermaid
graph TD
    A["CLR<br/>libamdhip64.so / libamdocl64.so"] -->|"hsa_signal_wait<br/>hsa_queue_create<br/>hsa_amd_memory_pool_allocate"| B["ROCr Runtime<br/>libhsa-runtime64.so"]
    B -->|"hsaKmtCreateQueue<br/>hsaKmtCreateEvent<br/>hsaKmtWaitOnMultipleEvents"| C["ROCt Thunk<br/>libhsakmt.so"]
    C -->|"ioctl<br/>mmap"| D["KFD<br/>amdgpu.ko /dev/kfd"]
    D -->|"回调 / 服务"| E["AMDGPU Core<br/>amdgpu.ko"]
    E -->|"MMIO / Ring Buffer"| F["GPU Hardware<br/>CP / SDMA / IH"]
```

各层职责简述：

| 层级 | 组件 | 在 Event 机制中的职责 |
|------|------|----------------------|
| **用户态前端** | CLR (`libamdhip64.so`) | 提供 `hip::Event`、`amd::Event`、`amd::Command` 等对象，构建 Barrier-AND Packet，管理 Command 生命周期。 |
| **用户态运行时** | ROCr (`libhsa-runtime64.so`) | 实现 HSA 规范。管理 `hsa_queue_t`、`hsa_signal_t`、Interrupt Signal，提供 ROCr 扩展（如 `hsa_amd_signal_value_pointer`）。 |
| **用户态 Thunk** | ROCt (`libhsakmt.so`) | 轻量级封装库。将 ROCr 的调用翻译为对 `/dev/kfd` 的 `ioctl`，并处理 `mmap` 映射。 |
| **内核态计算驱动** | KFD (`amdgpu.ko` 的一部分) | 提供 `/dev/kfd` 字符设备。负责创建用户态队列、分配 Signal Page、处理 GPU 中断并唤醒等待线程。 |
| **内核态显示/图形核心** | AMDGPU Core (`amdgpu.ko`) | 提供 TTM/GEM 内存管理、GPUVM 页表、Interrupt Handler (IH) 环读取、Doorbell 页分配、MQD/HQD 寄存器编程等基础服务。KFD 通过 `kgd2kfd` 回调调用这些服务。 |
| **硬件** | GPU (CP, SDMA, IH) | 执行 AQL Packet，写入完成状态，产生中断。 |

### 9.3 KFD 对 Event 机制的三大核心支撑

虽然 CLR 不直接感知 KFD，但 KFD 在以下三个方面为 CLR Event 提供了不可或缺的基础设施。

#### 9.3.1 AQL Queue 的创建与管理

CLR 中的 `amd::HostQueue` / `hip::Stream` 最终需要一个硬件队列来执行命令。这个硬件队列就是 HSA AQL Queue，其创建流程如下：

1. **内存分配**：ROCr 通过 ROCt 向 KFD 发起 `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` 和 `AMDKFD_IOC_MAP_MEMORY_TO_GPU`，由 amdgpu 的 TTM/GEM 子系统分配队列环缓冲区（Ring Buffer）、读指针（RPTR）、写指针（WPTR）、End-of-Packet（EOP）缓冲区以及上下文保存/恢复区。这些缓冲区被映射到进程的 GPUVM 地址空间。
2. **队列注册**：ROCr 调用 `AMDKFD_IOC_CREATE_QUEUE`，KFD 内部执行：
   - 查找当前进程对应的 `kfd_process` 和目标 GPU 的 `kfd_dev`。
   - 调用 `pqm_create_queue()`，通过 `device_queue_manager` 操作集分配一个硬件队列槽位。
   - 分配 **MQD**（Memory Queue Descriptor），其中保存队列的完整状态（GPU VA、Doorbell 偏移、WPTR/RPTR 地址等）。
   - 通过 **KIQ**（Kernel Interface Queue）或 **MES** 固件将 MQD 加载到 **HQD**（Hardware Queue Descriptor）寄存器中，使 GPU 调度器能够识别并调度该队列。
3. **Doorbell 分配**：KFD 从 GPU 的 Doorbell Aperture 中为该进程分配一段 Doorbell 页，并通过 `mmap` 将其映射到用户态地址空间。CLR 后续通过 `hsa_queue_t::doorbell_signal` 写入该地址来通知 GPU 有新工作到达，**无需系统调用**。

#### 9.3.2 Signal Page 与中断唤醒

CLR 中所有 `amd::Event::hw_event_` 的本质都是 `hsa_signal_t`，而 `hsa_signal_t` 的内核实现依赖 KFD 的 Signal Page 机制：

1. **Signal Page 分配**：当 ROCr 首次创建 HSA Signal（通过 `hsaKmtCreateEvent` → `AMDKFD_IOC_CREATE_EVENT`）时，KFD 分配一个物理连续的 **Signal Page**（最多容纳 4096 个 64-bit Signal Slot）。该页被初始化为 `UNSIGNALED_EVENT_SLOT`。
2. **用户态映射**：KFD 通过 `remap_pfn_range` 将该 Signal Page 映射到用户态虚拟地址空间。在独立显卡（dGPU）上，Signal Page 位于 GPU 可见的系统内存（GTT）中，以便 GPU CP 可以直接写入。
3. **GPU 完成通知（零内核介入路径）**：当 GPU 完成一个 AQL Packet（如内核分发或 Barrier-AND）后，CP 直接将完成值写入 Signal Page 中对应的 Slot。**此步骤完全不需要内核参与**，是纯粹的内存写操作。这也是 CLR 的硬件快速等待路径（`hsa_signal_wait_scacquire` 轮询用户态内存）能够零 syscall 工作的根本原因。
4. **CPU 阻塞与中断唤醒**：当主机线程需要阻塞等待（如 `hipEventSynchronize` 走软件路径）时，ROCr 通过 `AMDKFD_IOC_WAIT_EVENTS` 进入内核。KFD 为每个待等待的 Signal 创建一个 `kfd_event_waiter`，将其挂到内核 `waitqueue_head_t` 上并睡眠。当 GPU 完成工作并触发中断时，KFD 中断处理程序解析 IH Ring 中的事件，找到对应的 `kfd_event`，设置 `signaled = true`，并调用 `wake_up_all()` 唤醒阻塞线程。

#### 9.3.3 Doorbell 映射

Doorbell 是用户态通知 GPU "有新命令请处理" 的关键机制：
- KFD 通过 amdgpu 的 `AMDGPU_GEM_DOMAIN_DOORBELL` 域分配 Doorbell 页。
- 该页被 `mmap` 到用户态后，ROCr 将其地址缓存到 `hsa_queue_t::doorbell_signal`。
- CLR 在 `FlushSubmissionBatch()` 时执行一次对该用户态地址的写操作（`__atomic_store_n`），即完成硬件通知。该写操作通过 PCI BAR 直接到达 GPU 的 Doorbell 接收逻辑，无需陷入内核。

### 9.4 GPU 完成到 CLR 的完整中断路径

下图展示了从 GPU 硬件完成一个命令，到最终 CLR 中 `amd::Event::setStatus(CL_COMPLETE)` 被调用的完整内核-用户态协作链路：

```mermaid
graph TD
    A["GPU CP / SDMA 完成 AQL Packet"] -->|"写完成值到 Signal Page Slot<br/>零内核介入"| B["Signal Page<br/>用户态可见内存"]
    A -->|"写入 IH Ring Entry<br/>携带 PASID + context_id"| C["GPU IH Ring Buffer<br/>中断处理环"]
    C -->|"触发 MSI/IRQ"| D["CPU 中断入口"]
    D --> E["amdgpu_irq_handler<br/>amdgpu_ih_process"]
    E -->|"按 client_id/source_id 分发"| F["kgd2kfd_interrupt"]
    F --> G["KFD event_interrupt_wq_v9<br/>解析 SOC15_INTSRC_CP_END_OF_PIPE<br/>或 SOC15_INTSRC_SDMA_TRAP"]
    G --> H["kfd_signal_event_interrupt<br/>按 PASID 查找 kfd_process"]
    H --> I["set_event_from_interrupt<br/>acknowledge_signal + wake_up_all"]
    I --> J["阻塞在 WAIT_EVENTS ioctl<br/>的主机线程被唤醒"]
    J --> K["ROCr: InterruptSignal::WaitRelaxed 返回"]
    K --> L["CLR: amd::Event::setStatus<br/>CL_COMPLETE"]
    B -->|"用户态轮询发现 Signal 归零<br/>零 syscall 路径"| M["ROCr: hsa_signal_wait_scacquire 返回"]
    M --> L
```

**两条完成通知路径的对比**：

| 路径 | 触发条件 | 是否涉及内核 | 延迟特征 | 适用场景 |
|------|---------|-------------|---------|---------|
| **用户态轮询** | Signal Page Slot 被 GPU 直接写为完成值 | 否（纯用户态内存读） | 最低延迟（微秒级），但消耗 CPU | `HSA_WAIT_STATE_ACTIVE`、短超时、高频查询 |
| **中断唤醒** | GPU 写 IH Ring → CPU IRQ → KFD 唤醒 | 是（完整中断处理链） | 较高延迟（数十至数百微秒），零 CPU 开销 | `HSA_WAIT_STATE_BLOCKED`、长时间等待、主机线程睡眠 |

CLR 的 `hipEventSynchronize` 和 `amd::Event::awaitCompletion` 会根据策略（active wait / yield / blocked）在两条路径之间自适应选择。

### 9.5 KFD Event 与 dma-fence 的区别

在 AMDGPU 内核生态中，存在两种看似相似但用途截然不同的同步原语，理解它们的区别对于定位问题至关重要：

| 维度 | **KFD Event / HSA Signal** | **dma-fence** |
|------|---------------------------|---------------|
| **内核结构** | `kfd_event`、`signal page`、`waitqueue_head_t` | `dma_fence`、`drm_sched_fence`、`dma_resv` |
| **用户态接口** | `/dev/kfd` ioctl（`CREATE_EVENT`、`WAIT_EVENTS`） | 无直接用户态接口，主要在内核内部使用 |
| **使用场景** | 计算队列（AQL Queue）完成通知、用户态 Event 同步 | DRM GPU Scheduler 作业调度、BO 驻留管理、TTM Eviction、跨驱动 Buffer 共享 |
| **GPU 通知方式** | GPU CP 直接写 Signal Page（内存写） | 硬件 Ring 完成中断 → `amdgpu_fence_process()` → `dma_fence_signal()` |
| **CLR 是否使用** | **是**。所有 `hipEvent`、`streamWait` 都基于此。 | **否**。CLR 的 AQL Queue 绕过 DRM Scheduler，不直接使用 dma-fence。但 KFD 在 TTM Eviction 时会使用 dma-fence 作为侧翼机制（`amdgpu_amdkfd_fence`）。 |
| **路径归属** | KFD 用户态计算路径 | AMDGPU DRM 图形/调度路径 |

**简言之**：CLR 的 Event 机制走的是 **KFD 用户态计算路径**，其核心是 Signal Page + KFD Interrupt；而 `dma-fence` 是 AMDGPU DRM 子系统内部用于缓冲区对象（BO）生命周期管理和图形作业调度的机制。两者在内核中并行存在，服务于不同的子系统，但在 TTM 内存回收时会产生交互（例如 KFD 需要等待 dma-fence 才能安全迁移 BO）。

### 9.6 总结

CLR 的 Event 机制虽然在源码层面与 AMDGPU KMD 完全解耦，但在运行时却构建于 KFD 提供的三大基石之上：**AQL Queue 管理**、**Signal Page 中断唤醒**和 **Doorbell 用户态映射**。理解这一边界，有助于开发者定位以下类型的问题：

- **Event 不触发 /  hang**：可能需要检查 KFD 中断是否被正确路由（`amdgpu.irq` 参数、IH ring 溢出）。
- **Event 延迟异常高**：可能是主机等待策略退化为阻塞路径，或 KFD 中断处理线程（`event_interrupt_wq`）调度延迟。
- **多进程 Event 同步失败**：IPC Event 依赖 KFD 的 PASID 隔离和 Signal Page 的进程级可见性，需要确认 KFD 是否启用了必要的进程共享支持。

---

## 10. 总结

AMD CLR 的 Event 机制是一个高度统一、层次分明的同步体系：

- **软件层**：`amd::Event` / `amd::Command` 提供了与前端无关的状态追踪、回调管理和主机通知能力。
- **调度层**：`amd::HostQueue` 与 `hip::Stream` 通过 Direct Dispatch 或 Threaded Queue 将命令高效地喂给 GPU。
- **硬件抽象层**：`roc::VirtualGPU` 与 `HwQueueTracker` 利用 HSA Signal 和 Barrier-AND Packet 实现了低开销的硬件同步。
- **内核支撑层**：KFD / AMDGPU 提供了 AQL Queue、Signal Page 和 Doorbell 等基础设施，使上述用户态机制得以落实在真实 GPU 硬件之上。

Event、同步与 Barrier 三者相辅相成：Event 提供“是否完成”的查询能力，同步是用户显式要求的“顺序保证”动作，而 Barrier 是运行时落实在硬件上的“顺序 enforcement”机制。理解从 CLR 到 ROCr、从 ROCr 到 KFD、再从 KFD 到 GPU 硬件的完整链条，有助于开发者写出既正确又高性能的 GPU 程序。

---

*本文档基于 AMD CLR 代码库中 `rocclr/platform/`、`hipamd/src/` 及 `opencl/amdocl/` 等模块的源码分析，并结合 AMDGPU Kernel Driver 文档与 ROCr/ROCt 架构资料整理而成。*
