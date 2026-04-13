# NCCL 集合操作入队与执行

入队调度层是 NCCL 用户 API 到 GPU 内核之间的桥梁，负责任务排序、算法选择、通道分配和内核启动。

---

## 1. 单独集合操作完整流程

以 `ncclAllReduce` 为例，用户调用时 NCCL 自动包装为 group 执行：

```mermaid
flowchart TD
    A["ncclAllReduce(sendbuff, recvbuff,\ncount, datatype, op, comm, stream)"]
    A --> B["构造 ncclInfo\n{func=ncclFuncAllReduce, buffers, count, ...}"]
    B --> C["ncclEnqueueCheck(&info)"]

    C --> C1["CommCheck — 通信器有效性"]
    C1 --> C2["ncclGroupStartInternal — depth=1"]
    C2 --> C3["ncclCommEnsureReady — 确保通信器就绪"]
    C3 --> C4["ArgsCheck — 参数验证"]
    C4 --> C5["taskAppend → collTaskAppend"]
    C5 --> C6["ncclGroupCommJoin — 加入 group"]
    C6 --> C7["ncclTaskCollSorterInsert — 按大小排序插入"]
    C7 --> C8["ncclGroupEndInternal — depth=0\n触发执行"]

    C8 --> D["创建 ncclGroupJob"]
    D --> E["groupLaunch"]

    E --> E1["ncclPrepareTasks\n任务排序 + 算法选择 + 通道分配"]
    E1 --> E2["ncclCollPreconnect\n运行时通道连接 (若需要)"]
    E2 --> E3["ncclTasksRegAndEnqueue\n缓冲区注册 + 工作结构构建"]
    E3 --> E4["doLaunches\n内核启动"]

    E4 --> E4a["ncclLaunchPrepare\n创建 kernel plan"]
    E4a --> E4b["ncclLaunchKernelBefore\nuploadWork (拷贝到 GPU)"]
    E4b --> E4c["ncclLaunchKernel\ncuLaunchKernelEx (GPU 内核!)"]
    E4c --> E4d["ncclLaunchKernelAfter\nuploadProxyOps + ncclProxyStart"]
    E4d --> E4e["ncclLaunchFinish\n记录事件, 流排序"]
```

---

## 2. Group 操作流程

### 2.1 显式 Group

```mermaid
flowchart TD
    A["ncclGroupStart"] --> A1["ncclGroupStartInternal\nncclGroupDepth=1"]

    A1 --> B["ncclAllReduce(comm0, ...)"]
    B --> B1["ncclEnqueueCheck"]
    B1 --> B2["ncclGroupStartInternal — depth=2\n不触发执行"]
    B2 --> B3["taskAppend(comm0)\ncollTaskAppend → collSorter"]
    B3 --> B4["ncclGroupEndInternal — depth=1\n不触发执行"]

    B4 --> C["ncclReduce(comm1, ...)"]
    C --> C1["taskAppend(comm1)"]

    C1 --> D["ncclGroupEnd"]
    D --> D1["ncclGroupEndInternal — depth=0\n触发执行"]
    D1 --> D2["创建 ncclGroupJob\n收集所有 comm 的任务"]

    D2 --> E{blocking 模式?}
    E -->|"阻塞 (默认)"| F["groupLaunch — 同步执行"]
    E -->|"非阻塞"| G["创建线程\nncclAsyncJobMain → groupLaunchNonBlocking"]
    G --> G1["返回 ncclInProgress"]
    G1 --> G2["后续: ncclGroupJobComplete\n等待线程完成"]
```

### 2.2 Group 内部执行

```mermaid
flowchart TD
    A["groupLaunch"] --> B["asyncJobLaunch\nP2P 预连接 (并行线程)"]

    B --> C["Per-clique:\nncclPrepareTasksAndCollPreconnect\n(并行 per comm)"]
    C --> C1["ncclPrepareTasks:\n算法选择 + 通道分配"]
    C --> C2["ncclCollPreconnect:\n运行时连接"]

    C1 --> D["Per-comm:\nncclTasksRegAndEnqueue\n(顺序, per device)"]
    D --> D1["ncclRegisterCollBuffers\n缓冲区注册"]
    D1 --> D2["构建 ncclDevWorkColl"]
    D2 --> D3["创建 ncclWorkList 入队"]

    D3 --> E["doLaunches\n(barrier 同步 per clique)"]
    E --> E1["ncclLaunchPrepare\nPlan 创建 + 任务调度"]
    E1 --> E2["内核启动 per plan"]
    E2 --> E3["ncclGroupCommLeave + 清理"]
```

---

## 3. 任务类型与路由

```mermaid
flowchart TD
    A["taskAppend(comm, info)"] --> B{操作类型?}
    B -->|"Send/Recv"| C["p2pTaskAppend\n追加 ncclTaskP2p\n到 peers[peer].sendQueue/recvQueue"]
    B -->|"集合操作"| D["collTaskAppend\n追加 ncclTaskColl\n到 planner->collSorter"]
    B -->|"CE 集合"| E["ceCollTaskAppend\n追加到 planner->collCeTaskQueue"]
    B -->|"RMA"| F["rmaTaskAppend\n追加 ncclTaskRma"]
```

---

## 4. 算法选择 (ncclPrepareTasks 内部)

### 4.1 选择流程

```mermaid
flowchart TD
    A["ncclPrepareTasks"] --> B["从 collSorter 按大小降序出队"]
    B --> C["按 (func, op, datatype) 分桶"]
    C --> D["每桶执行算法选择"]

    D --> D1["ncclGetCollNetSupport\n检查 CollNet 可用性"]
    D1 --> D2["ncclGetAlgoInfo\n选择最优算法/协议"]
    D2 --> D2a["updateCollCostTable\n计算所有 (algo, proto) 组合的代价"]
    D2a --> D2b["ncclTopoGetAlgoTime\ntime = latency + nBytes/bandwidth"]
    D2b --> D2c{tuner 插件?}
    D2c -->|"是"| D2d["tuner->getCollInfo\n插件覆盖代价"]
    D2c -->|"否"| D2e["topoGetAlgoInfo\n选择最小代价"]
    D2d --> D2e

    D2e --> D3["计算 nMaxChannels, nWarps"]
    D3 --> E["按 (isCollnet, isNvls) 再分桶"]
    E --> F["拼接: collnet 优先, 然后 standard"]
    F --> G["注册 NVLS 缓冲区\n构建 ncclDevWorkColl\n检查 runtimeConn"]
```

### 4.2 通道分配 (scheduleCollTasksToPlan)

```mermaid
flowchart TD
    A["scheduleCollTasksToPlan"] --> B["将任务按大小降序排列"]
    B --> C["计算 trafficPerChannel 配额"]
    C --> D["将 count 分配到连续通道范围"]
    D --> D1["channelLo..channelHi"]
    D1 --> D2["countLo (首个通道部分)\ncountMid (完整通道)\ncountHi (末个通道部分)"]

    D2 --> E{算法类型?}
    E -->|"CollNet"| F["channelLo=0, channelHi=nChannels-1\n所有通道参与"]
    E -->|"Standard"| G["按配额分配连续通道"]
```

---

## 5. Kernel Plan 构建

### 5.1 Plan 创建流程

```mermaid
flowchart TD
    A["ncclLaunchPrepare"] --> B["分配 ncclKernelPlan"]
    B --> C{任务类型?}
    C -->|"RMA"| D["scheduleRmaTasksToPlan"]
    C -->|"CE Coll"| E["构建 CE plan"]
    C -->|"Symmetric"| F["ncclSymmetricTaskScheduler"]
    C -->|"Standard"| G["scheduleCollTasksToPlan\ndrain colls → bcasts → p2p"]
    C -->|"P2P"| H["scheduleP2pTasksToPlan\nncclP2pChannelForPart 轮询"]

    D --> I["finishPlan"]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> I1["打包工作批次到 kernel args"]
    I1 --> I2["合并排序 proxy ops"]
    I2 --> I3["设置流同步 + host 回调"]
```

### 5.2 工作存储类型

| 存储类型 | 用途 | 预算 |
|---------|------|------|
| `ncclDevWorkStorageTypeFifo` | 非持久化 Plan | FIFO 大小 / 2 |
| `ncclDevWorkStorageTypePersistent` | CUDA Graph 捕获的 Plan | 1 << 30 bytes |

---

## 6. 内核启动

### 6.1 启动序列

```mermaid
flowchart TD
    A["doLaunches"] --> B["Per comm:\nncclLaunchPrepare\n(创建所有 plans)"]

    B --> C["Per comm, per plan:"]
    C --> C1["ncclLaunchKernelBefore_NoUncapturedCuda\nuploadWork: 拷贝工作描述到 GPU"]
    C1 --> C2["ncclLaunchKernel\ncuLaunchKernelEx 或 cuLaunchKernel\ngrid={nChannels,1,1}\nblock={threadPerBlock,1,1}"]
    C2 --> C3["ncclLaunchKernelAfter_NoCuda\nhostStreamPlanTask\n→ uploadProxyOps + ncclProxyStart"]

    C3 --> D["ncclLaunchFinish\n记录事件, 设置流排序"]
```

### 6.2 uploadWork 路径

```mermaid
flowchart TD
    A["uploadWork(comm, plan)"] --> B{Plan 持久化?}
    B -->|"否"| C["写入 Work FIFO\n(FIFO 循环缓冲区)"]
    B -->|"是"| D["写入 Persistent Buffer\n(固定位置, Graph 可重放)"]
    C --> E["设置 kernel args 中的\nworkOffset + channelMask"]
    D --> E
```

---

## 7. 关键数据结构

| 结构体 | 用途 |
|--------|------|
| `ncclInfo` | API 调用参数 (func, buffers, count, datatype, op, comm, stream) |
| `ncclTaskColl` | 集合任务 (算法/协议/通道范围/工作描述) |
| `ncclTaskP2p` | P2P 任务 (peer/缓冲区/通道) |
| `ncclKernelPlan` | 内核启动计划 (工作批次 + proxy ops) |
| `ncclDevWorkColl` | 设备端集合工作描述 |
| `ncclDevWorkP2p` | 设备端 P2P 工作描述 |
| `ncclDevWorkBatch` | 设备端批次头 (每通道每 plan 一个) |
| `ncclProxyOp` | 代理操作描述 (CPU 端) |
| `ncclGroupJob` | Group 执行上下文 |
| `ncclKernelPlanner` | 每通信器规划器 (任务队列 + sorter + plan) |

---

## 8. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/enqueue.cc` | 3097 | 入队、任务调度、Plan 构建、内核启动 |
| `src/group.cc` | ~800 | Group 管理、groupLaunch |
| `src/collectives.cc` | ~200 | API 入口 (ncclAllReduce 等) |
| `src/include/enqueue.h` | ~50 | 入队函数声明 |
| `src/include/group.h` | ~150 | Group 内联辅助、数据结构 |
