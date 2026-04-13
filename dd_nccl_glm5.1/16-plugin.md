# NCCL 插件系统

NCCL 插件系统允许外部库覆盖或扩展 NCCL 的核心行为，包括网络传输、算法选择、性能分析、环境变量和 GPU 发起网络。

---

## 1. 五种插件类型

```mermaid
flowchart TD
    A["NCCL 插件系统"] --> B["Net 插件 (ncclNet_t)\n自定义网络传输\nv6-v11 API"]
    A --> C["Tuner 插件 (ncclTuner_t)\n自定义算法/协议选择\nv2-v5 API"]
    A --> D["Profiler 插件 (ncclProfiler_t)\n性能分析钩子\nv1-v6 API"]
    A --> E["Env 插件 (ncclEnvPluginGetEnv)\n覆盖 NCCL_* 环境变量\nv1 API"]
    A --> F["GIN 插件 (ncclGin_t)\nGPU-Initiated Networking\nv1 API"]
```

| 插件类型 | 环境变量 | 加载时机 | 接口版本 |
|---------|---------|---------|---------|
| Net | `NCCL_NET_PLUGIN` | ncclNetPluginInit | v6-v11 |
| Tuner | `NCCL_TUNER_PLUGIN` | initTransportsRank | v2-v5 |
| Profiler | `NCCL_PROFILER_PLUGIN` | ncclProfilerPluginInit | v1-v6 |
| Env | `NCCL_ENV_PLUGIN` | ncclInitEnv | v1 |
| GIN | `NCCL_GIN_PLUGIN` | ncclGinInit (commAlloc) | v1 |

---

## 2. 插件加载通用流程

```mermaid
flowchart TD
    A["插件加载"] --> B["dlopen(pluginPath)\nRTLD_NOW | RTLD_LOCAL"]
    B --> C{dlopen 成功?}
    C -->|"否"| D["回退到内置实现"]
    C -->|"是"| E["dlsym(symbolName)\n查找插件符号"]

    E --> F{符号存在?}
    F -->|"否"| G["dlclose + 回退"]
    F -->|"是"| H["获取插件 vtable"]
    H --> I["调用 init() 初始化"]
    I --> J["注册到 comm/sharedRes"]
```

---

## 3. Net 插件

### 3.1 接口 (ncclNet_t)

```mermaid
classDiagram
    class ncclNet_t {
        +name: char*
        +pciPath: char*
        +guid: uint64_t
        +init(logger)
        +getProperties(props)
        +listen(dev, handle, listenComm)
        +connect(dev, handle, sendComm)
        +accept(listenComm, recvComm)
        +regMr(comm, data, size, mhandle)
        +regMrDmaBuf(comm, data, size, fd, mhandle)
        +deregMr(comm, mhandle)
        +isend(sendComm, data, size, tag, mhandle, request)
        +irecv(recvComm, data, size, tag, mhandle, request)
        +iflush(recvComm, data, size, mhandle, request)
        +test(request, done, size)
        +closeSend(sendComm)
        +closeRecv(recvComm)
        +closeListen(listenComm)
    }
```

### 3.2 版本差异

| 版本 | 新增能力 |
|------|---------|
| v6 | 基础接口 |
| v7 | regMrDmaBuf (DMA-BUF 注册) |
| v8 | 多 recv (irecv 支持多个缓冲区) |
| v9 | pullProxy (拉取式代理) |
| v10 | collNet 支持 |
| v11 | 完整 collNet + GIN 集成 |

### 3.3 内置 Net 实现

| 实现 | 文件 | 传输方式 |
|------|------|---------|
| Socket | transport/net_socket.cc | TCP Socket |
| IB | transport/net_ib/ | InfiniBand Verbs |

---

## 4. Tuner 插件

### 4.1 接口 (ncclTuner_t)

```mermaid
classDiagram
    class ncclTuner_t {
        +name: char*
        +init(comm, nRanks, nvlsSupport, collNetSupport)
        +getCollInfo(comm, coll, size, numPipeOps, algo, proto, nChannels)
        +destroy(comm)
    }
```

### 4.2 工作方式

```mermaid
flowchart TD
    A["ncclGetAlgoInfo"] --> B["updateCollCostTable\n计算所有 (algo,proto) 代价"]
    B --> C{tuner 插件存在?}
    C -->|"是"| D["tuner->getCollInfo\n插件可修改:\nalgo, proto, nChannels"]
    C -->|"否"| E["topoGetAlgoInfo\n选择最小代价"]
    D --> E
```

Tuner 插件可以覆盖 NCCL 的自动算法选择，适用于特定工作负载的优化。

---

## 5. Profiler 插件

### 5.1 接口 (ncclProfiler_t)

```mermaid
classDiagram
    class ncclProfiler_t {
        +name: char*
        +init(comm, nRanks, nvlsSupport, collNetSupport)
        +startColl(comm, coll, size, nChannels)
        +endColl(comm, coll)
        +startOperation(comm, opType, opId)
        +endOperation(comm, opType, opId)
        +recordEvent(comm, eventType, eventId)
        +destroy(comm)
    }
```

### 5.2 钩子点

| 事件 | 时机 |
|------|------|
| startColl | 集合操作开始 |
| endColl | 集合操作结束 |
| startOperation | 内核/代理操作开始 |
| endOperation | 内核/代理操作结束 |
| recordEvent | 自定义事件记录 |

---

## 6. Env 插件

### 6.1 接口 (ncclEnv_v1_t)

```mermaid
classDiagram
    class ncclEnv_v1_t {
        +init(major, minor, patch, suffix)
        +finalize()
        +getEnv(name): char*
    }
```

### 6.2 双插件架构

```mermaid
flowchart TD
    A["ncclGetEnv(name)"] --> B["ncclEnvPluginGetEnv()"]
    B --> C{外部插件?}
    C -->|"有"| D["外部插件 getEnv(name)\n最高优先级"]
    C -->|"无"| E["内置插件\nstd::getenv(name)"]

    F["ncclEnvPluginInit"] --> G["initEnv: 加载配置文件\nNCCL_CONF_FILE → ~/.nccl.conf → /etc/nccl.conf"]
    G --> H["ncclEnvPluginLoad\ndlopen(NCCL_ENV_PLUGIN)"]
    H --> I{加载成功?}
    I -->|"是"| J["外部插件激活"]
    I -->|"否"| K["内置插件激活"]
    J --> L["调用 plugin->init()"]
    K --> L
    L --> M["atexit(ncclEnvPluginFinalize)"]
```

### 6.3 配置文件解析

```mermaid
flowchart TD
    A["initEnv"] --> B{NCCL_CONF_FILE 设置?}
    B -->|"是"| C["读取指定文件"]
    B -->|"否"| D["读取 ~/.nccl.conf"]
    C --> E["读取 /etc/nccl.conf\n(补充，不覆盖)"]
    D --> E

    E --> F["逐行解析\n跳过 # 注释\nkey = value\nsetenv 注入进程环境"]
```

---

## 7. GIN 插件

参见 [15-gin.md](15-gin.md)。接口定义在 `ncclGin_t` 中，通过 `NCCL_GIN_PLUGIN` 环境变量加载。

---

## 8. 插件示例

NCCL 仓库内置了多个插件示例，位于 `plugins/` 目录：

| 目录 | 类型 | 说明 |
|------|------|------|
| `plugins/net/` | Net | 网络传输插件模板和示例 |
| `plugins/tuner/example/` | Tuner | Tuner 示例 + 测试 |
| `plugins/tuner/basic/` | Tuner | 基础 Tuner 实现 |
| `plugins/profiler/` | Profiler | Profiler 接口定义 |
| `plugins/profiler/example/` | Profiler | Profiler 示例 |
| `plugins/profiler/inspector/` | Profiler | Inspector Profiler 实现 |
| `plugins/env/` | Env | Env 插件示例 |
| `plugins/mixed/` | Mixed | 混合插件 (同时提供多种类型) |

---

## 9. 关键源文件

| 文件 | 功能 |
|------|------|
| `src/plugin/net.cc` | Net 插件加载 |
| `src/plugin/tuner.cc` | Tuner 插件加载 |
| `src/plugin/profiler.cc` | Profiler 插件加载 |
| `src/plugin/env.cc` | Env 插件加载和双插件调度 |
| `src/plugin/env/env_v1.cc` | 内置 Env 插件 (getenv) |
| `src/include/plugin/` | 插件接口定义头文件 |
