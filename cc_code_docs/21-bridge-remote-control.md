# Bridge远程控制系统

## 概述

Bridge远程控制系统是Claude Code实现远程控制功能的核心模块，位于`src/bridge/`目录，包含约30个文件。该系统允许用户通过claude.ai网站或移动端应用远程操控本地运行的Claude Code CLI会话。Bridge系统采用"环境-工作项-会话"三层架构，通过长轮询（long polling）获取服务器分派的工作项，建立WebSocket/SSE连接实现双向消息流，并提供完整的崩溃恢复和重连机制。

## 核心类型定义

### BridgeConfig - 桥接配置

`src/bridge/types.ts`中定义了BridgeConfig类型，包含桥接实例运行所需的全部配置信息：

- `dir`：工作目录
- `machineName`：机器名称
- `branch`：当前Git分支
- `gitRepoUrl`：Git仓库URL（可为null）
- `maxSessions`：最大会话数
- `spawnMode`：会话生成模式（`single-session` | `worktree` | `same-dir`）
- `bridgeId`：客户端生成的UUID，标识桥接实例
- `workerType`：工作者类型（`claude_code` | `claude_code_assistant`）
- `environmentId`：客户端生成的UUID，用于幂等环境注册
- `reuseEnvironmentId`：服务端签发的environment_id，用于重连
- `apiBaseUrl`：桥接连接的API基础URL
- `sessionIngressUrl`：WebSocket连接的Session Ingress基础URL
- `sessionTimeoutMs`：会话超时毫秒数

### BridgeApiClient - API客户端接口

BridgeApiClient定义了与Environments API交互的全部方法：

1. **registerBridgeEnvironment(config)**：注册桥接环境，返回`environment_id`和`environment_secret`
2. **pollForWork(environmentId, environmentSecret, signal?, reclaimOlderThanMs?)**：长轮询获取工作项，返回WorkResponse或null
3. **acknowledgeWork(environmentId, workId, sessionToken)**：确认接收到工作项
4. **stopWork(environmentId, workId, force)**：停止工作项
5. **deregisterEnvironment(environmentId)**：注销桥接环境（优雅关闭时）
6. **sendPermissionResponseEvent(sessionId, event, sessionToken)**：发送权限响应到会话
7. **archiveSession(sessionId)**：归档会话，使其不再显示为活跃
8. **reconnectSession(environmentId, sessionId)**：重新连接现有会话（强制停止陈旧工作者并重新排队）
9. **heartbeatWork(environmentId, workId, sessionToken)**：发送心跳延长工作项租约

### SessionHandle - 会话句柄

SessionHandle表示一个正在运行的会话，提供：

- `sessionId`：会话ID
- `done`：Promise<SessionDoneStatus>，会话完成时解析
- `kill()` / `forceKill()`：终止会话
- `activities`：最近活动的环形缓冲区
- `currentActivity`：最近一次活动
- `accessToken`：session_ingress_token
- `writeStdin(data)`：写入子进程stdin
- `updateAccessToken(token)`：更新访问令牌

### BridgeLogger - 日志接口

BridgeLogger定义了桥接系统的全部日志方法，包括横幅打印、会话开始/完成/失败日志、状态更新、空闲状态显示、重连状态显示、QR码切换、多会话显示管理等。

## initBridgeCore - 主入口函数

`src/bridge/replBridge.ts`是Bridge系统的核心文件（约2400行），`initBridgeCore`函数是主入口，实现了完整的桥接生命周期。

### 初始化流程

initBridgeCore接收`BridgeCoreParams`参数，执行以下步骤：

1. **读取崩溃恢复指针**：perpetual模式下读取`bridgePointer.json`获取先前的environmentId和sessionId
2. **注册桥接环境**：调用`api.registerBridgeEnvironment(bridgeConfig)`，获取environmentId和environmentSecret
3. **重连就位尝试**：如果存在先前的环境，调用`tryReconnectInPlace()`尝试复用现有会话
4. **创建会话**：若未复用，调用`createSession()`创建新会话
5. **写入崩溃恢复指针**：写入`bridgePointer.json`以便崩溃后恢复
6. **初始化去重集合**：`recentPostedUUIDs`（BoundedUUIDSet，容量2000）用于回声过滤，`recentInboundUUIDs`用于入站去重
7. **启动工作轮询循环**：`startWorkPollLoop()`持续轮询服务器分派的工作项
8. **注册清理函数**：优雅关闭时执行teardown

### 环境重连机制

当环境丢失（轮询返回404）时，系统尝试两级重连策略：

**策略1：原地重连（Reconnect-in-place）**
- 使用`reuseEnvironmentId`进行幂等重新注册
- 如果后端返回相同的environmentId，调用`reconnectSession()`重新排队现有会话
- currentSessionId保持不变，用户手机上的URL仍然有效
- previouslyFlushedUUIDs被保留，历史消息不会重发

**策略2：新建会话回退**
- 如果后端返回不同的environmentId（原始环境已过期）或reconnectSession失败
- 归档旧会话，在当前注册的环境上创建新会话
- 重置lastTransportSequenceNum为0
- 清空recentInboundUUIDs
- 重置userMessageCallbackDone
- 清空previouslyFlushedUUIDs，允许初始消息重新发送

重连限制：最多3次环境重建（MAX_ENVIRONMENT_RECREATIONS），超出后放弃。

### SSE序列号传递

`lastTransportSequenceNum`是跨传输交换的SSE事件流高水位标记。在传输关闭时捕获当前序列号，新传输使用该序列号作为`from_sequence_num`参数，避免服务器重放完整会话历史。种子值仅在复用先前的会话时使用initialSSESequenceNum，否则重置为0。

### 初始消息刷写与容量限制

首次WebSocket连接时，系统刷写初始消息（initialMessages）到服务器：

- 使用`isEligibleBridgeMessage()`过滤合格消息
- 排除`previouslyFlushedUUIDs`中已刷写的消息
- 应用`initialHistoryCap`（默认200）截断历史，防止大量事件写入造成服务器压力
- 刷写期间，FlushGate阻止新消息交错发送
- 刷写完成后，通过`drainFlushGate()`发送排队消息

### 回声去重（Echo Dedup）

`BoundedUUIDSet`（容量2000）追踪已发送消息的UUID，用于：
1. 过滤WebSocket回弹的自身消息
2. writeMessages中的二次去重（安全网）

### 容量感知轮询

工作轮询循环根据当前容量状态调整行为：
- **非满载**：以`poll_interval_ms_not_at_capacity`间隔轮询
- **满载**（transport已连接）：以`poll_interval_ms_at_capacity`间隔心跳轮询
- 满载期间启用非独占心跳模式（`non_exclusive_heartbeat_interval_ms`），通过`heartbeatWork()`API延长300秒的工作项租约

### 进程挂起检测

轮询循环中的睡眠超时检测机制：如果`setTimeout`超出截止时间60秒以上，判定为进程挂起（笔记本电脑合盖、SIGSTOP、VM暂停），强制执行一次快速轮询周期。

### 崩溃恢复指针（bridgePointer.json）

`src/bridge/bridgePointer.ts`实现了崩溃恢复机制：

- **写入时机**：会话创建后立即写入，每个onWorkReceived刷新mtime，perpetual模式每小时定时刷新
- **读取检查**：检查mtime（非内嵌时间戳），超过4小时（BRIDGE_POINTER_TTL_MS）视为过期
- **清除时机**：优雅关闭时清除；perpetual模式下不清除，允许下次启动恢复
- **Worktree感知**：`readBridgePointerAcrossWorktrees()`跨git worktree兄弟目录查找最新指针

### Teardown - 拆卸序列

拆卸流程按以下顺序执行：

1. 停止定时器（指针刷新、保活、SIGUSR2处理器）
2. 中止轮询循环
3. 捕获当前传输的SSE序列号
4. **Perpetual模式**：仅停止轮询，不发送result/stopWork/close，让后端自动超时
5. **普通模式**：
   - 发送result消息
   - 并行执行stopWork(force=true)和archiveSession
   - 关闭传输
   - 注销环境
   - 清除崩溃恢复指针

## initReplBridge - REPL包装器

`src/bridge/initReplBridge.ts`是REPL特定的包装器，负责读取引导状态并执行前置检查：

### 初始化步骤

1. **运行时门控**：`isBridgeEnabledBlocking()`检查Bridge是否启用
2. **OAuth检查**：必须有claude.ai OAuth令牌
3. **组织策略检查**：`isPolicyAllowed('allow_remote_control')`
4. **跨进程死令牌退避**：如果先前3个进程连续遇到相同过期令牌，跳过启动
5. **OAuth令牌主动刷新**：`checkAndRefreshOAuthTokenIfNeeded()`在首次API调用前刷新
6. **过期不可刷新令牌跳过**：令牌过期且刷新失败时，持久化失败计数到全局配置
7. **会话标题派生**：优先级为显式initialName > /rename > 最后有意义用户消息 > 生成slug
8. **获取组织UUID**：必须成功才能继续
9. **Env-less Bridge路径**：如果`isEnvLessBridgeEnabled()`且非perpetual，走v2路径
10. **v1路径**：收集Git上下文，委托给`initBridgeCore()`

### 会话标题派生策略

标题派生采用"计数1和3"策略：
- **第1条用户消息**：快速占位符（deriveTitle截断到50字符），然后异步调用generateSessionTitle（Haiku模型）升级
- **第3条用户消息**：基于完整对话重新生成标题
- 显式标题（initialName或/rename）后不再自动覆盖
- 生成序列号（genSeq）防止乱序覆盖

## bridgeApi - HTTP客户端

`src/bridge/bridgeApi.ts`实现了Environments API的HTTP客户端：

- 使用axios发送请求，设置`environments-2025-11-01` beta头
- 所有服务端提供的ID通过`validateBridgeId()`校验（防止路径遍历）
- `withOAuthRetry()`模式：401时尝试令牌刷新，成功后重试一次
- `BridgeFatalError`区分可恢复错误和致命错误（401/403/404/410/429）
- 409状态码（已归档）被视为幂等成功

## 配置解析

### bridgeConfig.ts

`src/bridge/bridgeConfig.ts`提供共享的认证和URL解析：

- `getBridgeTokenOverride()`：ant-only的CLAUDE_BRIDGE_OAUTH_TOKEN环境变量
- `getBridgeBaseUrlOverride()`：ant-only的CLAUDE_BRIDGE_BASE_URL环境变量
- `getBridgeAccessToken()`：优先dev覆写，然后OAuth密钥链
- `getBridgeBaseUrl()`：优先dev覆写，然后生产OAuth配置

### envLessBridgeConfig.ts

`src/bridge/envLessBridgeConfig.ts`定义v2（env-less）桥接的配置：

- 心跳间隔（默认20秒，服务器TTL 60秒，3倍余量）
- 令牌刷新缓冲（默认5分钟）
- 连接超时（默认15秒）
- 拆卸归档超时（默认1.5秒，受gracefulShutdown 2秒限制）
- UUID去重缓冲大小（默认2000）
- 最低版本要求（独立于v1的min_version）
- Zod schema验证配置，违反约束时回退到默认值

## v1/v2传输协议与自动故障转移

### v1传输（HybridTransport）

- WebSocket读取 + HTTP POST写入到Session-Ingress
- 认证使用OAuth令牌（自动刷新）
- autoReconnect=true，指数退避重连，最多10分钟预算
- 最大连续失败50次后停止

### v2传输（SSETransport + CCRClient）

- SSE读取 + HTTP POST写入到CCR /worker/*
- 认证必须使用JWT（OAuth不含session_id声明）
- JWT过期时服务器重新分派工作项，onWorkReceived重新触发
- `v2Generation`计数器防止陈旧握手：两个并发的registerWorker竞争时，后解析的（正确epoch）胜出

### 协议选择

由服务端驱动：工作项密钥中的`use_code_sessions`字段决定使用v1还是v2。`CLAUDE_BRIDGE_USE_CCR_V2`环境变量是ant-dev覆写。

## 依赖注入模式

initBridgeCore通过BridgeCoreParams实现完全依赖注入：

- `createSession`：注入会话创建逻辑（REPL使用createBridgeSession，Daemon使用createBridgeSessionLean）
- `archiveSession`：注入归档逻辑
- `toSDKMessages`：注入消息转换（避免拉入整个命令注册表）
- `onAuth401`：注入OAuth 401处理
- `getPollIntervalConfig`：注入轮询配置（REPL使用GrowthBook，Daemon使用静态配置）
- `getCurrentTitle`：注入标题读取（REPL读sessionStorage，Daemon返回静态标题）

## FlushGate - 刷写门控

`src/bridge/flushGate.ts`实现了一个状态机，防止初始刷写期间消息交错：

- `start()`：标记刷写进行中，enqueue()开始排队
- `enqueue()`：刷写活跃时返回true并排队，否则返回false
- `end()`：结束刷写，返回排队项供排空
- `drop()`：丢弃排队项（永久传输关闭）
- `deactivate()`：清除活跃标志但保留排队项（传输替换）

## 世代计数器

`v2Generation`计数器防止陈旧握手。在以下情况递增：
- onWorkReceived（任何新传输）
- doReconnect（环境重建）
- v2 createV2ReplTransport的then()回调中检查世代号，不匹配则丢弃陈旧传输

## Bridge生命周期流程图

```mermaid
flowchart TD
    A["initReplBridge() 入口"] --> B{"Bridge启用检查"}
    B -->|否| Z["返回null"]
    B -->|是| C{"OAuth令牌检查"}
    C -->|无令牌| Z
    C -->|有令牌| D{"组织策略检查"}
    D -->|禁止| Z
    D -->|允许| E{"Env-less模式?"}
    E -->|是| F["initEnvLessBridgeCore()"]
    E -->|否| G["initBridgeCore()"]

    G --> H["读取bridgePointer.json"]
    H --> I["注册桥接环境 registerBridgeEnvironment()"]
    I -->|失败| Z
    I -->|成功| J{"存在先前环境?"}
    J -->|是| K["tryReconnectInPlace()"]
    J -->|否| L["创建新会话 createSession()"]
    K -->|成功| M["复用先前会话"]
    K -->|失败| L
    L --> N["写入bridgePointer.json"]
    M --> N

    N --> O["启动工作轮询循环 startWorkPollLoop()"]
    O --> P{"轮询获取工作项"}
    P -->|无工作| Q["等待后继续轮询"]
    P -->|收到工作项| R["解码WorkSecret"]
    R --> S["确认工作项 acknowledgeWork()"]
    S --> T{"v1还是v2?"}
    T -->|v1| U["创建HybridTransport"]
    T -->|v2| V["创建SSETransport+CCRClient"]
    U --> W["wireTransport() 连接"]
    V --> W

    W --> X["初始消息刷写"]
    X --> Y["状态变为connected"]

    Q --> P

    AA["传输关闭"] --> AB{"关闭代码"}
    AB -->|1000| AC["正常关闭 触发teardown"]
    AB -->|4001| AD["会话未找到 有限重试"]
    AB -->|4003| AE["永久拒绝 停止重连"]
    AB -->|其他| AF["重连预算耗尽"]
    AF --> AG["reconnectEnvironmentWithSession()"]
    AG --> AH{"策略1: 原地重连"}
    AH -->|成功| O
    AH -->|失败| AI{"策略2: 新建会话"}
    AI -->|成功| O
    AI -->|失败| AJ["状态变为failed"]
