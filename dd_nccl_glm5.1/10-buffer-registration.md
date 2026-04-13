# NCCL 缓冲区注册机制

缓冲区注册将用户缓冲区在 NIC/GPU/NVLS 上提前注册，避免每次集合操作重复注册开销。注册缓存 (RegCache) 管理所有已注册的缓冲区，支持引用计数和多种传输类型的注册。注册的本质是告诉硬件"这个内存区域我后续会频繁使用，请提前做好访问准备"——例如在 NIC 上注册用于 RDMA，在 NVSwitch 上注册用于多播。

---

## 1. 注册缓存架构

### 1.1 核心数据结构

```mermaid
classDiagram
    class ncclReg {
        +begAddr: uintptr_t
        +endAddr: uintptr_t
        +localRefs: int
        +graphRefs: int
        +state: uint32_t
        +netHandleHead: ncclRegNetHandles*
        +regAddr: void*
        +regUCSize: size_t
        +regMCSize: size_t
        +dev: int
        +mcHandle: CUmemGenericAllocationHandle
        +caddrs: void*[]
        +collnetHandle: void*
        +ipcInfos: ncclIpcRegInfo[]
    }

    class ncclRegCache {
        +slots: ncclReg**
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

`ncclRegCache` 使用按地址排序的数组存储所有注册条目，支持 O(log n) 的二分查找。每个 `ncclReg` 条目记录一段连续内存区域（页对齐），包含多种传输类型的注册句柄。

`localRefs` 和 `graphRefs` 是两个独立的引用计数器：`localRefs` 由 `ncclCommRegister` 递增，`graphRefs` 由 `ncclCommGraphRegister` 递增。只有两者都为 0 时才清理注册。

### 1.2 注册状态标志

| 标志 | 说明 |
|------|------|
| `NET_REG_COMPLETE` | 网络注册完成（NIC 上注册了 RDMA 访问） |
| `NVLS_REG_COMPLETE` | NVLS 注册完成（NVSwitch 上绑定了多播组） |
| `NVLS_REG_POSSIBLE` | NVLS 注册可行（硬件支持） |
| `NVLS_REG_NO_SUPPORT` | NVLS 不支持此缓冲区 |
| `COLLNET_REG_COMPLETE` | CollNet 注册完成 |
| `IPC_REG_COMPLETE` | IPC 注册完成（跨进程共享） |

---

## 2. 注册流程

### 2.1 用户 API

```mermaid
flowchart TD
    A["ncclCommRegister(comm, buff, size, &handle)"] --> B{"NCCL_LOCAL_REGISTER=0\n或 P2P uses memcpy?"}
    B -->|"跳过"| C["handle = NULL, return success\n避免不必要的注册开销"]
    B -->|"注册"| D["ncclRegister(comm, buff, size, isGraph=false)"]

    E["ncclCommGraphRegister(comm, buff, size, &handle)"] --> F["ncclRegister(comm, buff, size, isGraph=true)\nCUDA Graph 路径专用"]

    G["ncclCommDeregister(comm, handle)"] --> H["递减 localRefs"]
    I["ncclCommGraphDeregister(comm, handle)"] --> J["递减 graphRefs"]

    H --> K{"localRefs==0 && graphRefs==0?"}
    J --> K
    K -->|"是"| L["regCleanup()\n释放所有传输类型的注册"]
    K -->|"否"| M["保留注册\n其他引用仍在使用"]
```

Graph 注册和普通注册的区别：Graph 注册的缓冲区在 CUDA Graph 生命周期内必须保持有效，因此使用独立的引用计数器。

### 2.2 ncclRegister 内部流程

```mermaid
flowchart TD
    A["ncclRegister(comm, buff, size, isGraph)"] --> B["页对齐:\nbegAddr = alignDown(buff)\nendAddr = alignUp(buff+size)\n注册粒度为内存页"]
    B --> C["遍历排序缓存 slots[]"]
    C --> D{"已有 slot 完全包含\n[begAddr, endAddr]?"}
    D -->|"是: 缓冲区已注册"| E["增加引用计数\nisGraph? graphRefs++ : localRefs++\n返回该 slot"]
    D -->|"否: 新注册"| F["在排序位置插入新 slot"]
    F --> F1{"capacity 够?"}
    F1 -->|"否"| G["容量翻倍 (初始32)\nrealloc + memmove"]
    F1 -->|"是"| H["memmove 腾出空间"]
    G --> H
    H --> I["创建新 ncclReg\n设置 begAddr/endAddr\n初始化引用计数"]
    I --> J["返回 ncclReg* 作为 handle"]
```

缓存的核心优化：如果新注册的缓冲区已经被现有条目包含（页对齐后），直接增加引用计数而不创建新条目。这避免了重复注册同一内存页的开销。

---

## 3. 注销与清理

当引用计数归零时，`regCleanup` 按传输类型依次清理：

```mermaid
flowchart TD
    A["regCleanup(reg)"] --> B{"state & NET_REG_COMPLETE?"}
    B -->|"是"| C["遍历 netHandleHead 链表\nncclNetDeregBuffer(handle)\n释放链表节点"]
    B -->|"否"| D{"state & NVLS_REG_COMPLETE?"}
    C --> D

    D -->|"是"| E["ncclNvlsDeregBuffer\n释放 CUmem handle\n释放 regAddr 相关资源"]
    D -->|"否"| F{"state & COLLNET_REG_COMPLETE?"}
    E --> F

    F -->|"是"| G["ncclCollnetDeregBuffer\nvia collnetProxyconn"]
    F -->|"否"| H{"state & IPC_REG_COMPLETE?"}
    G --> H

    H -->|"是"| I["遍历 ipcInfos[]\nncclIpcDeregBuffer(ipcInfo)\n释放 peer 远端地址数组"]
    H -->|"否"| J["清理完成"]
    I --> J
```

---

## 4. 集合级注册策略

集合操作启动时，`ncclRegisterCollBuffers` 根据算法类型决定需要注册哪些传输路径：

```mermaid
flowchart TD
    A["ncclRegisterCollBuffers(comm, task)"] --> B{"算法类型?"}

    B -->|"NVLS / NVLS_TREE"| C["ncclRegisterCollNvlsBuffers\n优先图注册, 回退本地注册\n多节点: 额外注册 CollNet"]
    C --> C1["ncclNvlsGraphRegisterBuffer\n→ ncclNvlsLocalRegisterBuffer\n绑定 UC 内存到 MC 多播组"]
    C1 --> C2["多节点: 注册 CollNet 网络缓冲区\n调整 nMaxChannels"]

    B -->|"Simple 协议"| D{"子算法?"}
    D -->|"COLLNET_DIRECT"| E["1. 注册 IPC 缓冲区 (节点内 P2P)\n2. 注册 CollNet 网络缓冲区\n3. 1-RPN 配置: channels 降为 1"]
    D -->|"RING"| F["1. 找网络 peer (send/recv NIC)\n2. 注册 IPC (节点内 P2P)\n3. 注册 Net 缓冲区 (跨节点 GDR)\nncclRegFind 检查已有注册避免重复"]
    D -->|"TREE / COLLNET_CHAIN"| G["1. 收集 Tree peers\n   up + down[0..2]\n2. 注册 IPC 缓冲区\n3. 可选: CollNet Chain 1RPN 注册"]
```

注册策略的关键原则：
- **按需注册**：只注册算法实际需要的传输路径，避免浪费
- **避免重复**：使用 `ncclRegFind` 检查已有注册，相同内存区域不重复注册
- **优先图注册**：CUDA Graph 路径优先使用 `GraphRegister`，缓冲区在 Graph 生命周期内有效

---

## 5. 图注册 vs 本地注册

| 方式 | 适用场景 | 引用计数 | 生命周期 |
|------|---------|---------|---------|
| **Graph 注册** | CUDA Graph 捕获 | graphRefs++ | Graph 销毁时 graphRefs-- |
| **本地注册** | 非捕获路径 | localRefs++ | Deregister 时 localRefs-- |

两者独立计数，只有当 `localRefs == 0 && graphRefs == 0` 时才清理注册。这意味着即使在 CUDA Graph 捕获期间调用 `ncclCommDeregister`，注册也不会被过早清理（因为 graphRefs 仍为正）。

---

## 6. 关键源文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/register/register.cc` | ~400 | 注册缓存、注册/注销、缓存查找 |
| `src/register/coll_reg.cc` | ~600 | 集合级缓冲区注册 (NVLS/IPC/Net/CollNet) |
| `src/register/sendrecv_reg.cc` | ~200 | P2P send/recv 缓冲区注册 |
| `src/include/register.h` | ~80 | 核心数据结构定义 |
| `src/include/register_inline.h` | ~30 | ncclRegFind 内联查找 |
