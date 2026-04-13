# NCCL 通道系统

通道 (Channel) 是 NCCL 中并行度的基本单位。每个通道代表一条独立的通信路径，拥有自己的 Ring/Tree 拓扑和 send/recv 连接器。多个通道并发运行以饱和硬件带宽。

---

## 1. 核心数据结构

### 1.1 主机端通道结构

```mermaid
classDiagram
    class ncclChannel {
        +peers: ncclChannelPeer** (nRanks+1+nvlsRanks)
        +devPeers: ncclDevChannelPeer** (GPU 端镜像)
        +devPeersHostPtr: ncclDevChannelPeer** (主机端指针)
        +ring: ncclRing
        +devRingUserRanks: int* (GPU 端 ring 排序)
        +tree: ncclTree
        +collnetChain: ncclTree
        +collnetDirect: ncclDirect
        +nvls: ncclNvls
        +id: int (通道索引)
        +workFifoProduced: uint32_t
        +collnetPeers: ncclChannelPeer* (CollNet 专用)
        +collnetDevPeers: ncclDevChannelPeer*
        +nvlsPeers: ncclChannelPeer* (NVLS 专用)
        +nvlsDevPeers: ncclDevChannelPeer*
    }

    class ncclChannelPeer {
        +send: ncclConnector[2] (0=direct, 1=proxy)
        +recv: ncclConnector[2]
        +refCount: int
    }

    class ncclConnector {
        +conn: ncclConnInfo (连接信息)
        +transportComm: ncclTransportComm* (传输回调)
        +proxyConn: ncclProxyConnection* (代理连接)
    }

    ncclChannel --> ncclChannelPeer : peers[]
    ncclChannelPeer --> ncclConnector : send/recv[2]
```

### 1.2 连接器信息

```mermaid
classDiagram
    class ncclConnInfo {
        +head: uint32_t* (生产者步进指针)
        +tail: uint32_t* (消费者步进指针)
        +buffs: void*[] (数据缓冲区, per protocol)
        +stepSize: int (每步字节数)
        +ptrExchange: void** (指针交换槽)
        +proxyConn: ncclProxyConnection*
        +direct: int (是否直连模式)
    }
```

### 1.3 设备端算法数据结构

```mermaid
classDiagram
    class ncclRing {
        +prev: int (前驱 rank 索引)
        +next: int (后继 rank 索引)
        +userRanks: int* (内部索引→用户 rank 映射)
        +rankToIndex: int* (反向查找)
    }

    class ncclTree {
        +up: int (父节点 rank, -1=根)
        +down: int[3] (子节点 rank, -1=无)
        +depth: int
    }

    class ncclDirect {
        +depth: int
        +out: int
        +nHeads: int
        +headRank: int (-1=非 head)
        +shift: int
        +heads: int[8]
        +up: int[7]
        +down: int[7]
    }

    class ncclNvls {
        +out: int
        +nHeads: int
        +headRank: int
        +up: int[32]
        +down: int
        +treeUp: int
        +treeDown: int[3]
    }
```

---

## 2. 通道初始化

### 2.1 标准通道初始化

```mermaid
flowchart TD
    A["initChannel(comm, channelId)"] --> B{id != -1? (已初始化)}
    B -->|"是"| C["直接返回"]
    B -->|"否"| D["设置 channel.id = channelId"]

    D --> E["计算 nPeers = nRanks + 1(collnet) + nvlsRanks"]
    E --> F["分配共享 peers:\nsharedRes->peers[channelId]"]
    F --> G["映射每个 rank 的 peer\nvia topParentRanks"]

    G --> H["分配 GPU 端 devPeers"]
    H --> I["cudaMemcpyAsync 拷贝地址"]

    I --> J["分配 Ring 数据:\nuserRanks, rankToIndex, devRingUserRanks"]
```

### 2.2 NVLS 通道初始化

```mermaid
flowchart TD
    A["initNvlsChannel(comm, channelId, parent, share)"] --> B{share=true?}
    B -->|"是"| C["复用 parent 的 NVLS peer 数组"]
    B -->|"否"| D["分配新的 NVLS peer 数组"]
    C --> E["映射 NVLS peers 到 peers[nRanks+1+r]"]
    D --> E
```

### 2.3 CollNet 通道初始化

```mermaid
flowchart TD
    A["initCollnetChannel(comm, channelId, parent, share)"] --> B{share=true?}
    B -->|"是"| C["复用 parent 的 CollNet peer 数组"]
    B -->|"否"| D["分配新的 CollNet peer 数组"]
    C --> E["映射 CollNet peer 到 peers[nRanks]"]
    D --> E
```

---

## 3. P2P 通道计算

### 3.1 每对 peer 的最小通道数

```mermaid
flowchart TD
    A["ncclTopoComputeP2pChannelsPerPeer(comm)"] --> B["遍历所有 peer 对"]
    B --> C{本地 NVLink peer?}
    C -->|"是"| D["minCh = 2 * max(1, bw/nvlBw)"]
    C -->|"否"| E{远程 peer?}
    E -->|"是"| F["minCh = max(1, NIC数, bw/nicBw)"]
    E -->|"否"| G["minCh = 0"]
    D --> H["全局取最大值"]
    F --> H
    G --> H
```

### 3.2 通道数圆整

```mermaid
flowchart TD
    A["ncclTopoComputeP2pChannels(comm)"] --> B["圆整到 2 的幂"]
    B --> C["上限: NCCL_MAX_P2P_NCHANNELS"]
    C --> D["上限: ncclParamMaxP2pNChannels"]
    D --> E["初始化额外通道直到 p2pnChannels"]
    E --> F["initChannel(comm, channelId)"]
```

---

## 4. 通道中的连接器

### 4.1 双连接器设计

每个 peer 的 send/recv 各有 2 个连接器：

| 索引 | 名称 | 用途 |
|------|------|------|
| 0 | Direct | 直接 P2P 读写 (NVLink/PCIe)，无需代理 |
| 1 | Proxy | 代理辅助，用于 NET/SHM 传输 |

```mermaid
flowchart TD
    subgraph "Channel Peer (rank → peer)"
        S0["send[0]: Direct\nP2P 直接写"]
        S1["send[1]: Proxy\n经代理发送"]
        R0["recv[0]: Direct\nP2P 直接读"]
        R1["recv[1]: Proxy\n经代理接收"]
    end

    S0 --> S0_1["conn.buffs → 远端 GPU 缓冲区\nconn.head/tail → 步进指针\nconn.ptrExchange → 指针交换"]
    S1 --> S1_1["proxyConn → 代理连接\ntransportComm → 传输回调"]
```

### 4.2 连接器选择逻辑

算法根据传输类型自动选择连接器：

- **P2P 传输**: 使用 direct 连接器 (connIndex=0)
- **NET 传输**: 使用 proxy 连接器 (connIndex=1)
- **SHM 传输**: 使用 proxy 连接器
- **CollNet 传输**: 使用 proxy 连接器
- **NVLS 传输**: 使用 direct + 特殊多播机制

---

## 5. 通道与算法数据的关系

### 5.1 每通道独立拓扑

每个通道拥有独立的 Ring/Tree 拓扑数据：

```mermaid
flowchart TD
    subgraph "Channel 0"
        R0["Ring: 0→1→2→3→0\nTree: 0→{1,2}"]
    end
    subgraph "Channel 1"
        R1["Ring: 3→2→1→0→3\nTree: 3→{2,1}"]
    end
    subgraph "Channel 2"
        R2["Ring: 1→0→3→2→1\nTree: 1→{0,3}"]
    end
```

不同通道使用不同的 GPU 排序，实现：
- 负载均衡：各通道带宽均匀分配
- 容错：某通道故障时其他通道仍可工作
- 带宽饱和：多路径并行传输

### 5.2 算法到通道的映射

| 算法 | 使用的通道数据 | 通道数 |
|------|---------------|--------|
| RING | ncclRing (prev/next) | comm->nChannels |
| TREE | ncclTree (up/down) | comm->nChannels |
| COLLNET_CHAIN | ncclTree (collnetChain) | comm->nChannels |
| COLLNET_DIRECT | ncclDirect (collnetDirect) | comm->nChannels |
| NVLS | ncclNvls | comm->nvlsChannels |

---

## 6. 通道在内核中的使用

### 6.1 blockIdx → channelId 映射

内核启动时，grid 大小 = nChannels，每个 block 对应一个通道：

```mermaid
flowchart TD
    A["GPU Kernel Launch\ngrid={nChannels,1,1}"] --> B["blockIdx.x → channelId"]
    B --> C["使用 channelMask (population count)\n跳过未使用的通道"]
    C --> D["从 ncclDevChannel 加载通道数据\nring/tree/nvls 拓扑"]
    D --> E["加载该通道的连接器信息\nncclDevChannelPeer.send/recv"]
```

### 6.2 通道间工作分配

集合操作的数据被分割到多个通道：

```mermaid
flowchart TD
    A["数据总量: count 个元素"] --> B["分配到 nChannels 个通道"]
    B --> C["channelLo..channelHi 范围\n每个通道处理部分数据"]
    C --> D["countLo + countMid + countHi\n覆盖连续通道范围"]
```

---

## 7. 关键环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_MIN_NCHANNELS` | 最小通道数 |
| `NCCL_MAX_NCHANNELS` | 最大通道数 |
| `NCCL_MAX_P2P_NCHANNELS` | P2P 最大通道数 |
| `NCCL_P2P_DISABLE` | 禁用 P2P，影响通道传输选择 |
| `NCCL_SPLIT_SHARE` | Split 时共享通道资源 |

---

## 8. 关键源文件

| 文件 | 功能 |
|------|------|
| `src/channel.cc` | 通道初始化、NVLS/CollNet 通道管理 |
| `src/include/channel.h` | 通道函数声明、P2P 通道计算 |
| `src/include/comm.h` | ncclChannel 结构定义、ncclComm 中的 channels 数组 |
| `src/include/device.h` | ncclRing/Tree/Direct/Nvls 设备端结构 |
| `src/include/transport.h` | ncclChannelPeer、ncclConnector、ncclConnInfo |
