# NCCL 缓冲区注册机制

缓冲区注册将用户缓冲区在 NIC/GPU/NVLS 上提前注册，避免每次集合操作重复注册开销。注册缓存 (RegCache) 管理所有已注册的缓冲区，支持引用计数和多种传输类型的注册。

---

## 1. 注册缓存架构

### 1.1 核心数据结构

```mermaid
classDiagram
    class ncclReg {
        +begAddr: uintptr_t (页对齐)
        +endAddr: uintptr_t (页对齐)
        +localRefs: int
        +graphRefs: int
        +state: uint32_t (完成标志位掩码)
        +netHandleHead: ncclRegNetHandles* (链表)
        +regAddr: void* (NVLS 注册地址)
        +regUCSize: size_t (NVLS UC 大小)
        +regMCSize: size_t (NVLS MC 大小)
        +dev: int (NVLS 设备)
        +mcHandle: CUmemGenericAllocationHandle
        +caddrs: void*[] (跨 rank NVLS 缓冲区地址)
        +collnetHandle: void*
        +collnetProxyconn: ncclProxyConnector*
        +ginMhandles: void*[]
        +ginHandles: void*[]
        +regIpcAddrs: ncclPeerRegIpcAddr
        +ipcInfos: ncclIpcRegInfo[] (per local rank)
    }

    class ncclRegCache {
        +slots: ncclReg** (排序数组)
        +capacity: int
        +population: int
        +pageSize: int
    }

    class ncclRegNetHandles {
        +netHandle: void*
        +proxyConn: ncclProxyConnector*
        +next: ncclRegNetHandles*
    }

    ncclRegCache --> ncclReg : slots[] (sorted by begAddr)
    ncclReg --> ncclRegNetHandles : netHandleHead (linked list)
```

### 1.2 注册状态标志

| 标志 | 说明 |
|------|------|
| `NET_REG_COMPLETE` | 网络注册完成 |
| `NVLS_REG_COMPLETE` | NVLS 注册完成 |
| `NVLS_REG_POSSIBLE` | NVLS 注册可行 |
| `NVLS_REG_NO_SUPPORT` | NVLS 不支持 |
| `COLLNET_REG_COMPLETE` | CollNet 注册完成 |
| `IPC_REG_COMPLETE` | IPC 注册完成 |

---

## 2. 注册流程

### 2.1 用户 API

```mermaid
flowchart TD
    A["ncclCommRegister(comm, buff, size, &handle)"] --> B{NCCL_LOCAL_REGISTER=0\n或 P2P uses memcpy?}
    B -->|"跳过"| C["handle = NULL, return success"]
    B -->|"注册"| D["ncclRegister(comm, buff, size, isGraph=false)"]

    E["ncclCommGraphRegister(comm, buff, size, &handle)"] --> F["ncclRegister(comm, buff, size, isGraph=true)"]

    G["ncclCommDeregister(comm, handle)"] --> H["递减 localRefs"]
    I["ncclCommGraphDeregister(comm, handle)"] --> J["递减 graphRefs"]

    H --> K{localRefs==0 && graphRefs==0?}
    J --> K
    K -->|"是"| L["regCleanup()"]
    K -->|"否"| M["保留注册"]
```

### 2.2 ncclRegister 内部流程

```mermaid
flowchart TD
    A["ncclRegister(comm, buff, size, isGraph)"] --> B["页对齐:\nbegAddr = alignDown(buff)\nendAddr = alignUp(buff+size)"]
    B --> C["遍历排序缓存 slots[]"]
    C --> D{已有 slot 完全包含 [begAddr, endAddr]?}
    D -->|"是"| E["增加引用计数\nlocalRefs++ 或 graphRefs++\n返回该 slot"]
    D -->|"否"| F["在排序位置插入新 slot"]
    F --> F1{capacity 够?}
    F1 -->|"否"| G["容量翻倍 (初始32)\nrealloc + memmove"]
    F1 -->|"是"| H["memmove 腾出空间"]
    G --> H
    H --> I["创建新 ncclReg\n设置 begAddr/endAddr\n初始化引用计数\nisGraph? graphRefs=1 : localRefs=1"]
    I --> J["返回 ncclReg* 作为 handle"]
```

### 2.3 缓存查找 (ncclRegFind)

```mermaid
flowchart TD
    A["ncclRegFind(cache, data, size)"] --> B["遍历排序 slots[]"]
    B --> C{slot->begAddr <= data\n&& data+size <= slot->endAddr?}
    C -->|"是"| D["返回该 ncclReg*"]
    C -->|"否"| E["继续搜索"]
    E --> F{遍历完?}
    F -->|"是"| G["返回 NULL (未找到)"]
```

---

## 3. 注销与清理

```mermaid
flowchart TD
    A["regCleanup(reg)"] --> B{state & NET_REG_COMPLETE?}
    B -->|"是"| C["遍历 netHandleHead 链表\nncclNetDeregBuffer(handle)\n释放链表节点"]
    B -->|"否"| D{state & NVLS_REG_COMPLETE?}
    C --> D

    D -->|"是"| E["ncclNvlsDeregBuffer\n释放 CUmem handle\n释放 regAddr/regMCSize 相关资源"]
    D -->|"否"| F{state & COLLNET_REG_COMPLETE?}
    E --> F

    F -->|"是"| G["ncclCollnetDeregBuffer\nvia collnetProxyconn"]
    F -->|"否"| H{state & IPC_REG_COMPLETE?}
    G --> H

    H -->|"是"| I["遍历 ipcInfos[]\nncclIpcDeregBuffer(ipcInfo)\n释放 host/device peer 远端地址数组\nregIpcAddrs.devPtr / regIpcAddrs.hostPtr"]
    H -->|"否"| J["清理完成"]
    I --> J
```

---

## 4. 集合级注册策略

### 4.1 注册路径选择

```mermaid
flowchart TD
    A["ncclRegisterCollBuffers(comm, task)"] --> B{算法类型?}

    B -->|"NVLS / NVLS_TREE"| C["ncclRegisterCollNvlsBuffers\n优先图注册, 回退本地注册"]
    C --> C1["ncclNvlsGraphRegisterBuffer → ncclNvlsLocalRegisterBuffer"]
    C1 --> C2["多节点: 额外注册 CollNet 网络缓冲区"]
    C2 --> C3["调整 nMaxChannels\n(单节点4-6, 多节点更少)"]

    B -->|"Simple 协议"| D{子算法?}
    D -->|"COLLNET_DIRECT"| E["1. 注册 IPC 缓冲区 (P2P peers)\nregisterCheckP2PConnection 确定需要的 peer\n2. 注册 CollNet 网络缓冲区\n3. 1-RPN 配置: channels 降为 1"]
    D -->|"RING"| F["1. 找网络 peer (send/recv NIC connector)\n2. 注册 IPC (节点内 P2P peers)\n3. 注册 Net 缓冲区 (跨节点 GDR)\n使用 ncclRegFind 检查已有注册"]
    D -->|"TREE / COLLNET_CHAIN"| G["1. 收集 Tree peers\n   (up + down[0..NCCL_MAX_TREE_ARITY-1])\n2. 注册 IPC 缓冲区\n3. 可选: CollNet Chain 1RPN 注册"]
```

### 4.2 registerCheckP2PConnection

```mermaid
flowchart TD
    A["registerCheckP2PConnection\n(comm, channelId, peer, connIndex)"] --> B{连接已建立?}
    B -->|"是"| C{P2P READ 或 WRITE?}
    C -->|"是"| D["需要注册"]
    C -->|"否"| E["不需要注册"]
    B -->|"否"| F["调用 canConnect 检查\n如果可以连接: 需要注册"]
```

### 4.3 IPC-only 优化

当单节点且仅 IPC 注册成功时，通道数限制为 16 (如果原通道数在 16-24 之间)。

---

## 5. P2P Send/Recv 注册

```mermaid
flowchart TD
    A["ncclRegisterP2pNetBuffer\n(comm, data, size, peer, channelId, connIndex)"] --> B{设备类型 == UNPACK?}
    B -->|"是"| C["跳过 (UNPACK 不需要注册)"]
    B -->|"否"| D["尝试图注册\nncclNetGraphRegisterBuffer"]
    D --> E{成功?}
    E -->|"否"| F["回退本地注册\nncclNetLocalRegisterBuffer"]
    E -->|"是"| G["完成"]

    H["ncclRegisterP2pIpcBuffer\n(comm, data, size, peer, channelId, connIndex)"] --> I["尝试图注册\nncclIpcGraphRegisterBuffer"]
    I --> J{成功?}
    J -->|"否"| K["回退本地注册\nncclIpcLocalRegisterBuffer"]
    J -->|"是"| L["计算远端地址\npeerRmtAddrs + offset"]
```

---

## 6. 图注册 vs 本地注册

| 方式 | 适用场景 | 特点 |
|------|---------|------|
| **Graph 注册** | CUDA Graph 捕获 | 缓冲区在 Graph 生命周期内有效，可跨重放复用 |
| **本地注册** | 非捕获路径 | 每次注册独立，引用计数管理生命周期 |

图注册优先尝试，失败后回退到本地注册。

---

## 7. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/register/register.cc` | ~400 | 注册缓存、注册/注销、缓存查找 |
| `src/register/coll_reg.cc` | ~600 | 集合级缓冲区注册 (NVLS/IPC/Net/CollNet) |
| `src/register/sendrecv_reg.cc` | ~200 | P2P send/recv 缓冲区注册 |
| `src/include/register.h` | ~80 | 核心数据结构定义 |
| `src/include/register_inline.h` | ~30 | ncclRegFind 内联查找 |
