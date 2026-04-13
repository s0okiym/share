# NCCL 通信器分裂与收缩

Split 和 Shrink 机制允许从父通信器创建子通信器，分别用于 rank 分组和故障恢复。两者共享统一的子通信器创建逻辑，关键差异在于资源共享策略。

---

## 1. API

| API | 签名 | 用途 |
|-----|------|------|
| **ncclCommSplit** | `(comm, color, key, newcomm, config)` | 按 color 分组，按 key 排序 |
| **ncclCommShrink** | `(comm, excludeRanksList, count, newcomm, config, shrinkFlags)` | 移除故障 rank |
| **ncclCommGrow** | `(comm, newcomm, config)` | 扩展通信器 (添加新 rank) |

---

## 2. 统一子通信器创建

```mermaid
flowchart TD
    A["ncclCommSplit / ncclCommShrink"] --> B["ncclCommInitChildComm\n(统一子通信器创建)"]
    B --> C["ncclCommInitRankDev\n(标准初始化路径)"]
    C --> D["ncclCommInitRankFunc\n(核心初始化函数)"]
    D --> E["commAlloc(comm, parent, nranks, rank)"]
    E --> F["bootstrapSplit\n(创建子 bootstrap ring)"]
    F --> G["initTransportsRank\n(拓扑+通道+连接)"]
```

---

## 3. Split 流程

### 3.1 Rank 确定 (commGetSplitInfo)

```mermaid
flowchart TD
    A["commGetSplitInfo(comm, color, key, commSplitInfo)"] --> B["每个 rank 填充:\nsplitInfo[rank].color = color\nsplitInfo[rank].key = key"]

    B --> C["bootstrapAllGather\n交换所有 rank 的 (color, key)"]

    C --> D["按 color 分组"]
    D --> E{color == NCCL_SPLIT_NOCOLOR (负数)?}
    E -->|"是"| F["该 rank 不加入任何子通信器\nnewcomm = NULL"]
    E -->|"否"| G["同 color 的 rank 分到一组\n组内按 key 排序确定新 rank"]

    G --> H["构建 parentRanks[]:\nchildRank → parentRank 映射"]
```

### 3.2 Hash 派生

```mermaid
flowchart TD
    A["子通信器 Hash 派生"] --> B["commHash = digestHash(\nparentHash, childCount, color)"]
    B --> C["childCount 递增\n确保多次 split 的 Hash 唯一"]
    C --> D["示例: parent split 为 color=0 和 color=1\n→ 两个不同的 commHash"]
```

### 3.3 Bootstrap Split

```mermaid
flowchart TD
    A["bootstrapSplit(commHash, comm, parent,\ncolor, key, parentRanks)"] --> B["从父 bootstrap 获取子集 rank 地址"]
    B --> C["创建新的 ring socket 连接\n仅包含子集 rank"]
    C --> D["执行 ring AllGather\n交换子集 rank 的 peer 地址"]
    D --> E["子通信器拥有独立的 bootstrap ring"]
```

---

## 4. Shrink 流程

### 4.1 Rank 确定 (getParentRanks)

```mermaid
flowchart TD
    A["getParentRanks(comm, excludeRanksList,\nexcludeRanksCount, parentRanks)"]
    A --> B["排序 excludeRanksList"]
    B --> C["从 parent rank 列表中\n移除被排除的 rank"]
    C --> D["存活 rank 重新编号\n获得连续的新 rank 号"]
    D --> E["构建 parentRanks[]:\nnewRank → oldRank 映射"]
```

### 4.2 Shrink Abort 模式

```mermaid
flowchart TD
    A["ncclCommShrink"] --> B{shrinkFlags & NCCL_SHRINK_ABORT?}
    B -->|"是"| C["设置父 comm 的 abort 标志\nsetCommAbortFlags"]
    C --> D["cudaStreamSynchronize\n等待内核完成"]
    D --> E["清除 abort 标志"]
    E --> F["创建子通信器\n(防止死锁)"]
    B -->|"否"| G["直接创建子通信器"]
    F --> G
```

---

## 5. 资源共享

### 5.1 共享决策

```mermaid
flowchart TD
    A["shareResources 决策"] --> B{parent 被撤销 (revokedFlag)?}
    B -->|"是"| C["shareResources = false"]

    B -->|"否"| D{Split 或 Shrink?}
    D -->|"Split"| E{config.splitShare 启用?\n(NCCL_COMM_SPLIT_SHARE_RESOURCES)"]
    E -->|"是"| F["shareResources = true"]
    E -->|"否"| C

    D -->|"Shrink"| G{NCCL_SHRINK_ABORT 标志?}
    G -->|"是"| H["shareResources = false\n(abort 模式不共享)"]
    G -->|"否"| I{config.shrinkShare 启用?\n(NCCL_COMM_SHRINK_SHARE_RESOURCES)"]
    I -->|"是"| F
    I -->|"否"| C
```

### 5.2 共享 vs 独立资源

| 资源 | 共享路径 | 独立路径 |
|------|---------|---------|
| **SharedResources** | `comm->sharedRes = parent->sharedRes` refcount++ | 全新分配 (deviceStream, hostStream, launchEvent, scratchEvent, peers[]) |
| **Proxy 状态** | `comm->proxyState = parent->sharedRes->proxyState` refcount++ | ncclProxyCreate (新代理线程) |
| **网络插件** | ncclNetInitFromParent (复用) | ncclNetInit (新实例) |
| **GIN 状态** | ncclGinInitFromParent (复用) | ncclGinInit (新实例) |
| **内存管理器** | `comm->memManager = parent->memManager` refcount++ | 新实例 |
| **Abort 标志** | 共享 parent 的 abortFlag/abortFlagDev refcount++ | 新分配 |
| **通道 Peers** | 复用 sharedRes->peers[channelId] (collnetPeers/nvlsPeers 可共享) | 全新分配 |

### 5.3 共享通道 Peer 机制

ncclChannel 中标记为 "comm split sharable" 的字段：

```mermaid
flowchart TD
    subgraph "ncclChannel 可共享字段"
        A["collnetPeers — CollNet peer 数组"]
        B["collnetDevPeers — CollNet GPU peer 数组"]
        C["nvlsPeers — NVLS peer 数组"]
        D["nvlsDevPeers — NVLS GPU peer 数组"]
    end

    subgraph "共享方式"
        E["子通信器引用父通信器的指针\nvia sharedRes->peers[channelId]"]
        F["topParentRanks 映射\n确保子 rank 到正确 peer 的映射"]
    end

    A --> E
    B --> E
    C --> E
    D --> E
```

---

## 6. Grow 流程

Grow 用于向现有通信器添加新 rank：

```mermaid
flowchart TD
    A["ncclCommGrow"] --> B["isGrow = true"]
    B --> C["commHash = hashCombine(baseMagic, nranks)"]
    C --> D["commAlloc(comm, parent, newNranks, myrank)"]
    D --> E["bootstrapInit (包含 parent 的 rank 交换)"]
    E --> F["initTransportsRank\n(重新构建拓扑和通道)"]
```

---

## 7. 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_COMM_SPLIT_SHARE_RESOURCES` | 0 | Split 时是否共享资源 |
| `NCCL_COMM_SHRINK_SHARE_RESOURCES` | 0 | Shrink 时是否共享资源 |
| `NCCL_SPLIT_SHARE` | — | 等同于 SPLIT_SHARE_RESOURCES |

---

## 8. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/init.cc` (ncclCommSplit) | ~100 | Split API 入口 |
| `src/init.cc` (ncclCommShrink) | ~100 | Shrink API 入口 |
| `src/init.cc` (ncclCommInitChildComm) | ~200 | 统一子通信器创建 |
| `src/init.cc` (commGetSplitInfo) | ~80 | Split rank 确定 |
| `src/init.cc` (getParentRanks) | ~40 | Shrink rank 确定 |
| `src/init.cc` (commAlloc) | ~130 | 通信器分配 + 资源共享 |
| `src/include/comm.h` | — | shareResources/isGrow 等字段 |
