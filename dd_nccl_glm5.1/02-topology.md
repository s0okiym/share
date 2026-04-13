# NCCL 拓扑发现与图计算

拓扑系统是 NCCL 的核心基础设施之一，负责检测硬件拓扑、计算 rank 间路径、搜索最优通道分配，并指导算法和协议选择。拓扑信息决定了 NCCL 如何将通信映射到物理硬件——哪些 GPU 之间走 NVLink、哪些需要经 CPU 中转、跨节点走哪个 NIC，这些决策都基于拓扑系统提供的数据。

---

## 1. 拓扑节点与链路类型

### 1.1 节点类型 (7 种)

| 类型 | 值 | 说明 |
|------|---|------|
| GPU | 0 | GPU 设备，携带 rank、cudaCompCap、gdrSupport |
| PCI | 1 | PCI 交换机，携带 device 信息（vendor/device ID） |
| NVS | 2 | NVSwitch，多 GPU 间的 NVLink 汇聚点 |
| CPU | 3 | NUMA 域，携带 arch、vendor、model、affinity |
| NIC | 4 | 网络接口卡，携带 dev、pciId、bw、gdrSupport、collSupport、maxChannels |
| NET | 5 | 网络端点 |
| GIN | 6 | GIN 设备 |

每个节点的 ID 由 `systemId<<56 | localId` 构成，`systemId` 标识主机，`localId` 标识主机内的位置。

### 1.2 链路类型

| 类型 | 说明 | 典型带宽 |
|------|------|---------|
| LINK_LOC | 自身 | LOC_BW (5000 GB/s，表示无瓶颈) |
| LINK_NVL | NVLink | 架构相关 (12-40 GB/s per link) |
| LINK_C2C | C2C 芯片间直连 | 架构相关 |
| LINK_PCI | PCIe | width * speed / 80.0 GB/s |
| LINK_SYS | SMP 互连 (跨 NUMA) | 架构相关 (6-40 GB/s) |
| LINK_NET | 网络 | NIC 带宽 |

同类型链路到同一远端节点会自动聚合（带宽累加），而不是创建多条链路记录。例如两个 GPU 之间有 4 条 NVLink，会合并为一条 LINK_NVL 链路，带宽为 4 倍单链带宽。

### 1.3 路径类型 (12 种，按距离排序)

| 类型 | 值 | 说明 | 示例 |
|------|---|------|------|
| PATH_LOC | 0 | 自身 | GPU → 自身 |
| PATH_NVL | 1 | 直接 NVLink | 同 NVLink 域 GPU |
| PATH_NVB | 2 | 经中间 GPU 的 NVLink | GPU→NVS→GPU (经另一 GPU) |
| PATH_C2C | 3 | C2C 链路 | GPU→CPU (C2C) |
| PATH_PIX | 4 | 单 PCIe 桥 | 同 PCI switch 下 |
| PATH_PXB | 5 | 多 PCIe 桥 (不经 CPU) | 多级 PCI switch |
| PATH_P2C | 6 | GPU→C2C→CPU→PCI→NIC | C2C 路径到 NIC |
| PATH_PXN | 7 | GPU→NVLink→中间GPU→PCI→NIC | PXN 路径 |
| PATH_PHB | 8 | 经 PCIe 主桥/CPU | 跨 NUMA 但同主机 |
| PATH_SYS | 9 | 跨 NUMA SMP 互连 | 跨 CPU socket |
| PATH_NET | 10 | 经网络 | 跨节点 |
| PATH_DIS | 11 | 断开 | 不可达 |

路径类型在 BFS 过程中通过 `max(当前路径类型, 新链路类型)` 递增。这确保路径类型反映路径中最"远"的段。

---

## 2. 核心数据结构

### 2.1 拓扑节点

```mermaid
classDiagram
    class ncclTopoNode {
        +type: int
        +id: uint64_t
        +nlinks: int
        +links: ncclTopoLink[]
        +paths: ncclTopoLinkList[]
        +used: uint64_t
        +gpu: {dev, rank, cudaCompCap, gdrSupport}
        +net: {dev, pciId, bw, gdrSupport, collSupport, maxChannels}
        +cpu: {arch, vendor, model, affinity}
        +pci: {device}
    }

    class ncclTopoLink {
        +type: int
        +bw: float
        +remNode: ncclTopoNode*
    }

    class ncclTopoLinkList {
        +list: ncclTopoLink*[]
        +count: int
        +bw: float
        +type: int
    }

    ncclTopoNode --> ncclTopoLink : links[]
    ncclTopoNode --> ncclTopoLinkList : paths[]
    ncclTopoLink --> ncclTopoNode : remNode
    ncclTopoLinkList --> ncclTopoLink : list[]
```

`ncclTopoLinkList` 表示一条完整路径，其中 `bw` 是瓶颈带宽（路径中最窄的链路带宽），`count` 是跳数，`type` 是路径类型分类。每个节点为每种节点类型维护一组路径，可以快速查询到任意类型节点的最优路径。

### 2.2 拓扑系统

```mermaid
classDiagram
    class ncclTopoSystem {
        +systemId: int
        +hostHashes: uint64_t[]
        +nHosts: int
        +nodes: ncclTopoNodeSet[7]
        +maxBw: float
        +totalBw: float
        +inter: int
    }

    class ncclTopoNodeSet {
        +count: int
        +nodes: ncclTopoNode[]
    }

    ncclTopoSystem --> ncclTopoNodeSet : nodes[7]
    ncclTopoNodeSet --> ncclTopoNode : nodes[]
```

`ncclTopoSystem` 按 7 种节点类型组织所有拓扑节点。`maxBw` 和 `totalBw` 在搜索阶段用于确定通道带宽上限。`inter=1` 表示多节点拓扑，会影响通道搜索策略（需要考虑跨节点带宽）。

---

## 3. 拓扑发现流程

### 3.1 ncclTopoGetSystem 完整流程

```mermaid
flowchart TD
    A["ncclTopoGetSystem(comm, &comm->topo)"] --> B{"NCCL_TOPO_FILE 环境变量?"}
    B -->|"是"| C["读取指定 XML 拓扑文件"]
    B -->|"否"| D["读取 /var/run/nvidia-topologyd/virtualTopology.xml\n若不存在则创建空 system 节点"]
    C --> E["检测本地 GPU\nNVML bus ID → rank, gdrSupport\n标记 keep=1"]
    D --> E
    E --> F["导入网络设备到拓扑\nmutex 保护，顺序: GIN → CollNet → NET\nncclTopoProcessNet 处理每个 NIC\n合并同设备多端口 NIC"]
    F --> G["裁剪无 keep=1 的 XML 分支\nncclTopoTrimXml"]
    G --> H{"MNNVL 模式?"}
    H -->|"是"| I["使用 clique ranks\n分配更大 XML 缓冲区"]
    H -->|"否"| J["从 hostHash 计算本地 rank"]
    I --> K["intra-node AllGather XML\nbootstrapIntraNodeAllGather\n每个 rank 序列化自己的 XML 然后交换"]
    J --> K
    K --> L{"MNNVL?"}
    L -->|"是"| M["ncclTopoFuseXml\n融合所有 peer XML 为一体"]
    L -->|"否"| N["ncclTopoGetSystemFromXml"]
    M --> N
    N --> N1["Pass 1: ncclTopoAddCpu\n创建 CPU 节点 + NUMA 信息\n递归处理 PCI 子树"]
    N1 --> N2["Pass 2: ncclTopoAddNvLinks\n创建 LINK_NVL\nGPU-GPU 或 GPU-NVS"]
    N2 --> N3["Pass 3: ncclTopoAddC2c\nGPU-CPU LINK_C2C"]
    N3 --> N4["Pass 4: ncclTopoAddPciLinks\nPCI 交换机间 LINK_LOC"]
    N4 --> N5["ncclTopoFlattenBcmSwitches\n扁平化 Broadcom Gen4 交换机"]
    N5 --> N6["ncclTopoConnectCpus\n同主机 CPU 间 LINK_SYS"]
    N6 --> N7["ncclTopoSortSystem\nNVLink 优先, PCI 下行/上行, SYS"]
```

**关键细节**：

- **GPU 检测**：只检测当前进程管理的 GPU（通过 `comm->peerInfo[comm->rank].busId`），然后标记 `keep=1`。其他 rank 的 GPU 信息通过后续的 AllGather 交换获得。

- **NIC 合并**：多端口 NIC 会被合并为单个 NIC 节点，合并策略由 `NCCL_NET_MERGE_LEVEL` 控制（默认 `PATH_PORT`，即同端口级别的 NIC 合并）。

- **MNNVL**（Multi-Node NVLink）：当 GPU 跨节点通过 NVLink 连接时，使用 clique 信息确定本地 rank 集合，并分配更大的 XML 缓冲区来容纳跨节点拓扑。

### 3.2 NVLink 带宽与计算能力的关系

NVLink 带宽由 `ncclTopoNVLinkBw(cudaCompCap)` 确定，单链带宽乘以链路数即为两 GPU 间总带宽：

| Compute Capability | NVLink 带宽 (per link) | 典型链路数 | 总带宽 |
|-------------------|----------------------|-----------|--------|
| SM60 (Pascal) | 18 GB/s | 4 | 72 GB/s |
| SM70 (Volta) | 20 GB/s | 6 | 120 GB/s |
| SM80 (Ampere A100) | 20 GB/s | 12 | 240 GB/s |
| SM86 (Ampere A30) | 12 GB/s | — | — |
| SM90 (Hopper) | 20.6 GB/s | 18 | 370.8 GB/s |
| SM100 (Blackwell) | 40.1 GB/s | 18 | 721.8 GB/s |

### 3.3 CPU 互连带宽

同主机不同 NUMA 域之间的带宽取决于 CPU 架构：

| CPU 架构 | 带宽 (GB/s) | 代表产品 |
|---------|------------|---------|
| BDW (Broadwell) | 6 | Xeon E5 v4 |
| SKL (Skylake) | 10 | Xeon SP |
| SRP (Sapphire Rapids) | 22 | Xeon 4th Gen |
| ERP (Emerald Rapids) | 40 | Xeon 5th Gen |
| AMD | 16 | EPYC |
| P9 (Power9) | 32 | POWER9 |
| ARM | 6 | Ampere |

CPU 型号通过 `familyId` 和 `modelId` 自动识别：Intel ERP (>=0xCF)、SRP (>=0x8F)、SKL (>=0x55)，其余为 BDW。

---

## 4. 路径计算 (BFS)

### 4.1 ncclTopoComputePaths 算法

`ncclTopoComputePaths` 对拓扑中的所有节点对计算最优路径，使用 BFS（广度优先搜索）从每个节点展开。

```mermaid
flowchart TD
    A["ncclTopoComputePaths(topo, comm)"] --> B["清除所有现有路径"]
    B --> C["对每种节点类型\nCPU, GPU, NET, GIN, NVS"]
    C --> D["对每个该类型节点"]
    D --> E["ncclTopoSetPaths(baseNode, system)\nBFS 从 baseNode 展开"]
    E --> F["初始化: basePath = 自身\ncount=0, bw=LOC_BW, type=PATH_LOC"]
    F --> G["逐层 BFS"]
    G --> H["对每条 link from 当前节点"]
    H --> I["候选 BW = min(path.bw, link.bw)\n瓶颈带宽取最小值"]
    I --> J{"允许经过 GPU?"}
    J -->|"仅 1 跳或 NVL-to-NVL 且 NVB 未禁用"| K["接受路径"]
    J -->|"否: 防止数据绕经无关 GPU"| L["跳过此路径"]
    K --> M{"新路径更优?\n更短 且 BW 更高"}
    M -->|"是"| N["更新路径\n确定路径类型"]
    M -->|"否"| O["保留原路径"]
    N --> P["加入下一层 BFS"]
    O --> G
    P --> G
    G --> G1["P2P 可达性检查\nncclTopoCheckP2p"]
    G1 --> G2["P2P 不可达 → addInterStep\n经本地 CPU 中转"]
    G2 --> G3["PXN 路径优化\n如果另一 GPU 有更近的 NIC"]
    G3 --> G4["无 GDR 的 NIC → 经本地 CPU"]
    G4 --> G5["预计算 NIC 的本地 GPU\nlocalGpu 字段"]
```

**BFS 的关键约束——GPU 路由限制**：数据不允许经过不相关的 GPU 中转（除非是单跳 NVLink）。具体规则是：只有当路径只有 1 跳、且链路类型为 NVLink、远端为 GPU 时，才允许路径经过 GPU。这防止了数据在 GPU 间"绕路"，把通信延迟转嫁给无辜的 GPU。违反此约束的路径被跳过，不会进入 BFS 的下一层。

**路径类型确定**：最终路径类型为 `max(当前路径类型, 新链路类型)`，取路径中最"远"的类型。特殊情况：PHB + C2C = P2C（GPU 经 C2C 到 CPU 再到 NIC），这是 Hopper 架构的典型路径。

### 4.2 后处理优化

BFS 完成后，还有几个重要的后处理步骤：

1. **P2P 检查**：对每对 GPU，检查路径类型是否在允许的 `p2pLevel` 内（默认 `PATH_PXB`）。超出时，通过 `addInterStep` 在路径中插入 CPU 中转节点。

2. **PXN 优化**：当 GPU `g` 到 NIC 的路径不佳（需经 CPU），但同节点另一个 GPU `g'` 通过 NVLink 连接到 `g` 且到 NIC 路径更好时，使用 `g'` 作为中继。路径变为 `g → NVLink → g' → PCI → NIC`，避免了 CPU 中转。

3. **GDR 检查**：如果 GPU 到 NIC 的路径优于 PHB 但 GDR 被禁用，则强制插入 CPU 中转步骤。

---

## 5. 通道搜索与图计算

### 5.1 图结构 (ncclTopoGraph)

```mermaid
classDiagram
    class ncclTopoGraph {
        +id: int
        +pattern: int
        +crossNic: int
        +collNet: int
        +minChannels: int
        +maxChannels: int
        +nChannels: int
        +bwIntra: float
        +bwInter: float
        +typeIntra: int
        +typeInter: int
        +sameChannels: int
        +intra: int[]
        +inter: int[]
    }
```

`ncclTopoGraph` 描述了一种通道拓扑方案。`intra[]` 数组定义每个通道内 GPU 的排列顺序（影响 ring/tree 的构建），`inter[]` 数组定义每个通道使用的 NIC。`bwIntra` 和 `bwInter` 分别是节点内和跨节点每通道带宽。

### 5.2 搜索模式

| 模式 | 节点内 | 跨节点 |
|------|--------|--------|
| RING | GPUa→GPUb→...→GPUx→GPUa | NETn→GPUa→...→GPUx→NETn |
| TREE | GPUa→GPUb→...→GPUx | NETn→GPUa→...→GPUx, GPUa→NETn |
| SPLIT_TREE | 同 TREE | 发送和接收使用不同 NIC |
| NVLS | N/A | NETn→GPUhead, 经 NVSwitch |
| COLLNET_DIRECT | 所有 GPU 星形到 head | NETn→GPUhead→分发 |

### 5.3 两阶段搜索算法

通道搜索采用"先找到可行解，再优化带宽"的两阶段策略。

```mermaid
flowchart TD
    A["ncclTopoCompute(topo, graph)"] --> B["阶段 1: 寻找可行解"]
    B --> C["sameChannels=1\n所有通道使用相同 GPU 排序\n减少搜索空间"]
    C --> D["ncclTopoSearchRec\n从 NET/GPU0 递归搜索\n有超时限制"]
    D --> E{"找到解?"}
    E -->|"是"| F["阶段 2: 尝试提升带宽"]
    E -->|"否"| G["放松 sameChannels=0\n允许不同通道不同排序"]
    G --> H{"找到解?"}
    H -->|"是"| F
    H -->|"否"| I["SM90+: 尝试更简单的 TREE 模式"]
    I --> J["放松 typeIntra/typeInter\n允许更远路径类型"]
    J --> K["尝试 crossNic=2\n交替使用不同 NIC"]
    K --> L["降低带宽目标\n下一级速度数组"]
    L --> D
    F --> F1{"图类型?"}
    F1 -->|"RING"| F2["提升 bwIntra 和 bwInter"]
    F1 -->|"NVLS"| F3["仅提升 bwInter\n节点内走 NVLink 已经很快"]
    F1 -->|"TREE"| F4["独立提升 bwIntra / bwInter"]
    F2 --> F5["ncclTopoCompareGraphs 选择最优"]
    F3 --> F5
    F4 --> F5
```

**阶段 1 的核心逻辑**：从最高带宽目标开始尝试。先要求所有通道使用相同的 GPU 排序（`sameChannels=1`），这极大缩小了搜索空间。如果找不到解，逐步放松约束——允许不同排序、允许更远路径、允许跨 NIC、最终降低带宽目标。每次放松后重新搜索，直到找到可行解。

**阶段 2 的优化**：找到可行解后，尝试在不增加通道数的前提下提升单通道带宽。通过递增速度数组中更高带宽的目标值来尝试。Ring 同时提升 intra 和 inter 带宽，Tree 可以独立优化两者。

### 5.4 图比较优先级

`ncclTopoCompareGraphs()` 按以下优先级选择最优方案：

1. **更多通道** (nChannels) — 并行度优先
2. **更高总带宽** (nChannels × bwIntra) — 吞吐量优先
3. **更少跳数** (nHops) — 延迟优先

### 5.5 GPU 排序启发式

`ncclTopoSearchNextGpuSort()` 按多维度权重排序候选下一个 GPU：

1. **interBw** (最重要) — 到该 GPU 的网络带宽，高优先
2. **interPciBw** — PCI 带宽
3. **interNhops** — 更少跳数优先
4. **intraBw** — 节点内带宽
5. **intraNhops** — 更少节点内跳数

在 NVSwitch 系统中，搜索被限制在相邻 GPU（索引相邻），这大幅减少了搜索空间，因为 NVSwitch 拓扑下所有 GPU 对称等价。

### 5.6 通道复制优化

`ncclTopoDupChannels` 在满足条件时将通道数翻倍（每通道带宽减半），条件是：
- 非 NVLS 模式
- 带宽 >= 25 GB/s
- SM90+ 时仅当带宽 < 50 GB/s 且通道数 > 4

这利用了"更多通道比更高单通道带宽更有利于并行"的特性。

---

## 6. Ring 和 Tree 构建

### 6.1 Ring 构建 (ncclBuildRings)

```mermaid
flowchart TD
    A["ncclBuildRings(rings, prev, next, nranks, nChannels)"] --> B["对每个 channel"]
    B --> C["从 rank 0 开始\n跟随 next 链接遍历 nranks 步"]
    C --> D["验证: 环回到起始 rank\n否则报错 ring does not loop back"]
    D --> E["验证: 所有 rank 恰好出现一次\n使用 bitmask 快速检查"]
    E --> F["输出 rings 数组\n每个 channel 的 rank 排列"]
```

Ring 构建本质是验证——由搜索算法确定的 `prev[]` 和 `next[]` 数组是否构成合法环。验证包括闭环检查和完备性检查（每个 rank 出现且仅出现一次）。

### 6.2 双二叉树构建 (ncclGetDtree)

```mermaid
flowchart TD
    A["ncclGetDtree(rank, nranks)"] --> B["Tree 0: ncclGetBtree(rank, nranks)\n标准二叉树"]
    A --> C["Tree 1"]
    C --> D{"nranks 奇偶?"}
    D -->|"奇数"| E["旋转 btree: shift by 1\nTree 1 的根为 rank 1"]
    D -->|"偶数"| F["镜像 btree: rank → nranks-1-rank\nTree 1 的根为 nranks-1"]
    B --> G["ncclGetBtree 算法:\n找最低设置 bit\n确定 parent 和 children"]
```

二叉树构建基于最低设置位（lowest set bit）算法。对于 rank `r`，找到其最低非零位 `bit`，则：
- **父节点**：`up = (r ^ bit) | (bit << 1)`，若超出 nranks 则 `up = r ^ bit`
- **子节点**：`down0 = r - bit/2`，`down1 = r + bit/2`

双树的关键特性：**每个 rank 至少在一个树中是叶子节点**。这使得 AllReduce 的 reduce 和 broadcast 可以流水线执行——在一个树的叶子节点完成 reduce 的同时，另一个树的根节点可以开始 broadcast，互不干扰。

Tree 0 结构 (16 rank):
```
        0──────────8
           /    \
          4      12
        /  \    /  \
       2    6  10   14
      /\  /\  /\   /\
     1 3 5 7 9 11 13 15
```

---

## 7. 关键环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_TOPO_FILE` | 指定 XML 拓扑文件路径，跳过自动检测 |
| `NCCL_TOPO_DUMP_FILE` | 导出检测到的拓扑到文件，用于调试 |
| `NCCL_P2P_LEVEL` | 覆盖 P2P 路径类型阈值（默认 PATH_PXB） |
| `NCCL_P2P_DISABLE` | 禁用 P2P 直连，所有 GPU 间通信经 CPU |
| `NCCL_P2P_DISABLE_NVB` | 禁用 NVB 路径（经中间 GPU 的 NVLink） |
| `NCCL_SHM_DISABLE` | 禁用 SHM 传输 |
| `NCCL_CROSS_NIC` | 允许跨 NIC 通道 |
| `NCCL_IGNORE_DISABLED_P2P` | 忽略 NVML 报告的 P2P 禁用 |
| `NCCL_NET_MERGE_LEVEL` | NIC 合并级别 |

---

## 8. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/graph/topo.cc` | ~2000 | 拓扑发现、XML 解析、节点/链路创建、NIC 合并 |
| `src/graph/topo.h` | ~200 | 核心数据结构定义、链路类型、路径类型常量 |
| `src/graph/paths.cc` | ~800 | BFS 路径计算、P2P/GDR/PXN 检查与优化 |
| `src/graph/search.cc` | ~800 | 通道搜索算法、GPU 排序启发式、带宽优化 |
| `src/graph/rings.cc` | ~100 | Ring 构建与合法性验证 |
| `src/graph/trees.cc` | ~150 | 双二叉树构建 |
| `src/graph/xml.cc` | ~600 | XML 拓扑文件解析与生成 |
| `src/graph/connect.cc` | ~300 | 连接建立辅助函数 |
