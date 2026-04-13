# NCCL 错误处理与中断机制

NCCL 通过双 abort 标志（主机端 + 设备端）实现多层次的中断传播，确保通信器出错时各组件（socket、代理线程、GPU 内核）都能及时退出，避免死锁。

---

## 1. 双 Abort 标志架构

### 1.1 标志位置

```mermaid
flowchart TD
    subgraph "ncclComm 中的 abort 标志"
        A["abortFlag — 主机端\nncclCalloc 分配\n可被主机代码和 socket 操作检查"]
        B["abortFlagDev — 设备端\nncclCudaHostCalloc 分配\n可被 CUDA 内核访问 (pinned host memory)"]
        C["abortFlagRefCount — 引用计数\n共享通信器时多个 owner"]
        D["childAbortFlag — 指向子通信器的主机标志"]
        E["childAbortFlagDev — 指向子通信器的设备标志"]
    end
```

### 1.2 标志设置

```mermaid
flowchart TD
    A["setCommAbortFlags(comm, value)"] --> B["原子存储 value → abortFlag\n(__ATOMIC_RELEASE)"]
    B --> C["原子存储 value → abortFlagDev\n(__ATOMIC_RELEASE)"]
    C --> D{childAbortFlag 存在?}
    D -->|"是"| E["级联写入 childAbortFlag"]
    E --> F["级联写入 childAbortFlagDev"]
    D -->|"否"| G["完成"]
    F --> G
```

---

## 2. 中断传播路径

### 2.1 主机端传播

```mermaid
flowchart TD
    A["错误触发\n通信器撤销 / 异步错误"] --> B["setCommAbortFlags"]
    B --> C["Socket 操作"]
    B --> D["Bootstrap 操作"]
    B --> E["代理线程"]
    B --> F["Group 作业"]
    B --> G["子通信器级联"]

    C --> C1["所有阻塞 socket 操作\n(connect, accept, send, recv)\n检查 abortFlag\n__ATOMIC_ACQUIRE\nif 非零: 返回 ncclInternalError"]

    D --> D1["bootstrapSend/Recv\n检查 abortFlag\n防止初始化期间挂起"]

    E --> E1["Service 线程:\n每次 poll 循环检查 abortFlag\nif 非零: break 退出\n注意: 永不让 proxy service\n阻塞在 poll 中"]
    E --> E2["Progress 线程:\n检查 abortFlag\nif 非零: 停止推进"]

    F --> F1["groupLaunch 检测:\ngroupAbortFlag\nerrorJobAbortFlag\n→ 传播到各 comm 的\nabortFlag + abortFlagDev"]

    G --> G1["setCommAbortFlags\n级联写入\nchildAbortFlag\nchildAbortFlagDev"]
```

### 2.2 设备端传播

```mermaid
flowchart TD
    A["abortFlagDev 被设置\n(pinned host memory)"] --> B["GPU 内核检测"]

    B --> C["testAbort(abortFlag, steps)\n在关键自旋循环中调用"]
    C --> C1["steps 计数器递增\n每 10000 次迭代才实际读取标志\n(amortize 原子加载开销)"]
    C1 --> C2["if *abortFlag != 0:\n返回 true → 内核提前退出"]

    B --> D["checkAbort(abortCache, abortValue, spins)\n在 Primitives 循环中调用"]
    D --> D1["维护 per-thread abortCache\n一旦读到非零值，缓存"]
    D1 --> D2["NCCL_SPINS_BEFORE_CHECK_ABORT\n次自旋后检查一次"]

    B --> E["实现方式"]
    E --> E1["NVCC 编译:\ncuda::atomic_ref<uint32_t>\n内存语义: relaxed load"]
    E --> E2["其他编译器:\nvolatile 指针解引用"]
```

---

## 3. NCCLWAIT 宏

用于主机端阻塞等待中检查 abort：

```mermaid
flowchart TD
    A["NCCLWAIT / NCCLWAITGOTO"] --> B["自旋等待循环"]
    B --> C["每次迭代:\n__atomic_load_n(abortFlag, __ATOMIC_ACQUIRE)"]
    C --> D{abortFlag != 0?}
    D -->|"是"| E["提前退出:\nNCCLWAIT → return ncclInternalError\nNCCLWAITGOTO → goto cleanup"]
    D -->|"否"| F["继续等待"]
```

---

## 4. Socket 集成

所有 socket 操作（connect、accept、send、recv）都存储 `abortFlag` 指针：

```mermaid
flowchart TD
    A["ncclSocketConnect\nncclSocketAccept\nncclSocketSend\nncclSocketRecv"] --> B["在阻塞等待中\n每次 poll 迭代检查 abortFlag"]

    B --> C{abortFlag 非零?}
    C -->|"是"| D["返回 ncclInternalError\n中断阻塞等待"]
    C -->|"否"| E["继续 poll"]
```

同样适用于 IPC socket (`ipcsocket.h`, `ipcsocket.cc`)。

---

## 5. 代理线程 Abort 集成

```mermaid
flowchart TD
    subgraph "Service Thread"
        S1["每次 poll 循环开始:\nif (*abortFlag) → stop = PROXY_ABORT"]
        S2["poll timeout:\n有 asyncOps → timeout=0\n无 asyncOps → timeout=500ms\n确保不会长时间阻塞"]
        S3["注释: never let proxy service\nthread blocks in poll\nor it cannot receive abortFlag"]
    end

    subgraph "Progress Thread"
        P1["循环条件:\nwhile (!stop || active ops) && !abortFlag"]
        P2["abortFlag 设置后:\n停止接受新操作\n等待现有操作完成或超时"]
    end
```

---

## 6. Group Abort 传播

```mermaid
flowchart TD
    A["groupLaunch 中检测错误"] --> B{groupAbortFlag 或\nerrorJobAbortFlag?}
    B -->|"是"| C["传播到 group 中每个 comm:\nsetCommAbortFlags(comm, 1)"]
    C --> D["级联到子通信器:\nchildAbortFlag\nchildAbortFlagDev"]
    D --> E["所有相关组件收到中断信号:\nsocket → 退出阻塞\nproxy → 停止循环\nkernel → 提前退出"]
```

---

## 7. Abort 标志的引用计数

```mermaid
flowchart TD
    A["通信器创建\n(init.cc)"] --> B{shareResources?}
    B -->|"是"| C["共享 parent 的 abortFlag + abortFlagDev\nabortFlagRefCount++"]
    B -->|"否"| D["新分配 abortFlag + abortFlagDev\nabortFlagRefCount = 1"]

    E["通信器销毁\n(init.cc)"] --> F["abortFlagRefCount--"]
    F --> G{RefCount == 0?}
    G -->|"是"| H["释放 abortFlag\n释放 abortFlagDev"]
    G -->|"否"| I["保留 (其他 owner 仍在使用)"]
```

---

## 8. Shrink Abort 模式中的 Abort 使用

```mermaid
flowchart TD
    A["ncclCommShrink\n(shrinkFlags & NCCL_SHRINK_ABORT)"] --> B["1. 设置 abort 标志\nsetCommAbortFlags(comm, 1)"]
    B --> C["2. cudaStreamSynchronize\n等待所有内核完成\n(内核检测到 abort 后提前退出)"]
    C --> D["3. 清除 abort 标志\nsetCommAbortFlags(comm, 0)"]
    D --> E["4. 创建子通信器\n(此时内核已停止，安全创建)"]
```

这防止了 Shrink 操作期间的死锁：如果内核仍在运行且持有资源，直接创建子通信器可能导致资源竞争。

---

## 9. 关键源文件

| 文件 | 功能 |
|------|------|
| `src/include/comm.h` | abortFlag/abortFlagDev/childAbortFlag 字段 |
| `src/init.cc` | setCommAbortFlags、引用计数、Shrink abort |
| `src/group.cc` | Group abort 传播 |
| `src/proxy.cc` | 代理线程 abort 检查 |
| `src/misc/socket.cc` | Socket abort 检查 |
| `src/include/nccl_device/utility.h` | 设备端 testAbort |
| `src/device/primitives.h` | 设备端 checkAbort |
| `src/include/checks.h` | NCCLWAIT/NCCLWAITGOTO 宏 |
