# 错误恢复与容错机制

## 概述

Claude Code 实现了一个深度的多层错误恢复体系，覆盖从 API 调用到进程崩溃的完整故障谱系。整个恢复架构的核心设计原则是：**优先静默自动恢复，仅在所有恢复路径穷尽后才向用户暴露错误**。这一原则通过"暂扣-恢复"（withhold-recover）模式实现——可恢复的错误消息先被扣留，恢复成功后消息被丢弃，恢复失败时才向上层传播。

系统涵盖七层恢复机制：查询层的模型回退与 withheld 错误恢复、压缩层的 PTL 重试与熔断器、Bridge 层的崩溃恢复指针与进程挂起检测、远程会话的 WebSocket 重连策略、MCP 的指数退避重连、认证层的 stale-while-revalidate 缓存，以及各子系统统一的 fail-open/fail-closed 决策和超时层级。

## 错误恢复级联总览

```mermaid
flowchart TD
    A["API 请求发起"] --> B{"响应/错误?"}
    B -- "正常响应" --> C["继续正常流程"]
    B -- "FallbackTriggeredError" --> D["模型回退重试"]
    B -- "529/过载" --> E["withRetry 指数退避"]

    B -- "prompt-too-long" --> F{"withheld?"}
    F -- "是" --> G["暂扣错误消息"]
    G --> H{"contextCollapse<br/>drain 恢复?"}
    H -- "成功" --> I["drain 已暂存collapse<br/>重新发起请求"]
    H -- "暂存队列空" --> J{"reactiveCompact<br/>反应式压缩?"}
    J -- "成功" --> K["压缩后重试"]
    J -- "已尝试过/失败" --> L["暴露错误<br/>终止查询"]

    B -- "max-output-tokens" --> M{"withheld?"}
    M -- "是" --> N{"使用默认8k?"}
    N -- "是" --> O["升级到64k<br/>ESCALATED_MAX_TOKENS"]
    N -- "否" --> P{"恢复次数<br/>&lt; 3?"}
    P -- "是" --> Q["注入meta消息<br/>继续生成"]

    P -- "否" --> R["暴露错误"]

    B -- "media-size" --> S{"reactiveCompact<br/>启用?"}
    S -- "是" --> T["strip-retry<br/>移除媒体重试"]
    S -- "否" --> U["暴露错误"]

    V["压缩请求自身"] --> W{"命中PTL?"}
    W -- "是" --> X{"PTL重试次数<br/>&lt; 2?"}
    X -- "是" --> Y["truncateHeadForPTLRetry<br/>裁剪最旧轮次"]
    X -- "否" --> Z["压缩失败"]
    W -- "否" --> AA["压缩成功"]

    AB["自动压缩"] --> AC{"连续失败次数?"}
    AC -- "&lt; 3" --> AD["继续尝试"]
    AC -- ">= 3" --> AE["熔断器触发<br/>跳过后续自动压缩"]

    AF["Bridge 进程崩溃"] --> AG["bridgePointer.json<br/>持久化恢复指针"]
    AG --> AH{"指针过期?<br/>mtime &gt; 4h"}
    AH -- "否" --> AI["--continue 恢复会话"]
    AH -- "是" --> AJ["清除指针<br/>正常启动"]

    style C fill:"#6f6",stroke:"#333"
    style I fill:"#6f6",stroke:"#333"
    style K fill:"#6f6",stroke:"#333"
    style AE fill:"#f96",stroke:"#333"
    style L fill:"#f66",stroke:"#333",color:"#fff"
    style R fill:"#f66",stroke:"#333",color:"#fff"
    style Z fill:"#f66",stroke:"#333",color:"#fff"
    style AI fill:"#6cf",stroke:"#333"
```

## 一、查询层错误恢复：query.ts

### 1.1 FallbackTriggeredError 与模型回退

`FallbackTriggeredError` 定义在 `src/services/api/withRetry.ts` 中，当主模型过载返回 529 错误且配置了 `fallbackModel` 时由 `withRetry` 抛出。在 query.ts 的流式循环中，该错误被捕获并触发完整的模型回退流程：

**回退流程步骤：**

1. 将 `currentModel` 切换为 `fallbackModel`，设置 `attemptWithFallback = true` 以重入流式循环
2. 通过 `yieldMissingToolResultBlocks()` 为已发出的 tool_use 块生成合成错误 tool_result，确保 API 消息配对完整性
3. 清除 `assistantMessages`、`toolResults`、`toolUseBlocks` 数组
4. 如果使用流式工具执行器（`streamingToolExecutor`），调用 `discard()` 丢弃挂起结果并创建新实例
5. 对于 Ant 内部用户，调用 `stripSignatureBlocks()` 移除 thinking 签名块——因为签名与模型绑定，将受保护 thinking 块重放给未受保护的回退模型会导致 400 错误
6. 记录 `tengu_model_fallback_triggered` 遥测事件
7. 向用户发出系统消息通知模型切换，使用 `warning` 级别

**关键的孤儿消息处理：** 当流式回退在流中间触发时，已部分生成的 assistant 消息（特别是含 thinking 块的）具有无效签名。系统为这些孤儿消息 yield tombstone 消息，从 UI 和 transcript 中移除它们，避免 "thinking blocks cannot be modified" 的 API 错误。

### 1.2 yieldMissingToolResultBlocks

此函数是 query.ts 中保证 API 消息配对完整性的核心机制。API 要求每个 tool_use 都有对应的 tool_result，否则下一轮请求会被拒绝。函数在以下场景被调用：

- 模型回退时（`'Model fallback triggered'`）
- 查询异常抛出时（原始错误消息）
- 用户中断时（`'Interrupted by user'`）
- 流式执行器丢弃时（内部清理）

```typescript
function* yieldMissingToolResultBlocks(
  assistantMessages: AssistantMessage[],
  errorMessage: string,
) {
  for (const assistantMessage of assistantMessages) {
    const toolUseBlocks = assistantMessage.message.content.filter(
      content => content.type === 'tool_use',
    ) as ToolUseBlock[]
    for (const toolUse of toolUseBlocks) {
      yield createUserMessage({
        content: [{
          type: 'tool_result',
          content: errorMessage,
          is_error: true,
          tool_use_id: toolUse.id,
        }],
        toolUseResult: errorMessage,
        sourceToolAssistantUUID: assistantMessage.uuid,
      })
    }
  }
}
```

### 1.3 Withheld 可恢复错误机制

query.ts 实现了精细的"暂扣-恢复"模式，三种可恢复错误被 withhold 而非立即 yield：

**withhold 判定逻辑：**

- **prompt-too-long**：由 `reactiveCompact.isWithheldPromptTooLong()` 和 `contextCollapse.isWithheldPromptTooLong()` 双重门控，任一子系统标记即暂扣
- **max-output-tokens**：由 `isWithheldMaxOutputTokens()` 判定（`msg.apiError === 'max_output_tokens'`）
- **media-size**：由 `reactiveCompact.isWithheldMediaSizeError()` 判定，且必须 `mediaRecoveryEnabled`（在流循环前提升的门控值）

**关键设计：** `mediaRecoveryEnabled` 在流循环之前一次性提升（hoist），因为 `getFeatureValue_CACHED_MAY_BE_STALE` 可能在 5-30 秒的流过程中翻转。如果 withhold 和 recover 使用不同的门控值，就会出现"暂扣但不恢复"的死锁——消息被吞掉永远不会到达用户。

withheld 消息仍然被推入 `assistantMessages` 数组，使后续的恢复检查能发现它们，但不会通过 `yield` 暴露给 SDK 消费者（如 cowork/desktop），因为这些消费者会在任何 `error` 字段上终止会话。

### 1.4 prompt-too-long 多层恢复

当检测到被暂扣的 prompt-too-long 错误时，系统按优先级尝试三条恢复路径：

**路径一：Context Collapse Drain**

```typescript
if (feature('CONTEXT_COLLAPSE') && contextCollapse &&
    state.transition?.reason !== 'collapse_drain_retry') {
  const drained = contextCollapse.recoverFromOverflow(messagesForQuery, querySource)
  if (drained.committed > 0) {
    // 设置 transition: { reason: 'collapse_drain_retry', committed } 并 continue
  }
}
```

- 调用 `recoverFromOverflow()` 提交所有已暂存的 context collapse
- 成功 drain 后重新发起请求（`collapse_drain_retry` transition）
- **防抖动保护**：检查前一次 transition 是否已为 `collapse_drain_retry`，避免在 drain 后仍然 413 时无限循环
- 这是"廉价"的恢复路径，保留了细粒度上下文

**路径二：Reactive Compact**

```typescript
const compacted = await reactiveCompact.tryReactiveCompact({
  hasAttempted: hasAttemptedReactiveCompact,
  messages: messagesForQuery,
  cacheSafeParams: { systemPrompt, userContext, systemContext, toolUseContext, ... },
})
```

- 完整的对话压缩，从尾部剥离消息缩减上下文
- `hasAttemptedReactiveCompact` 标志防止螺旋——如果压缩已尝试过但仍然 prompt-too-long，不会再试
- 压缩成功后重建消息并继续查询循环

**路径三：暴露错误**

所有恢复路径穷尽后：
- yield 被暂扣的错误消息
- 调用 `executeStopFailureHooks()` 通知钩子系统
- 返回 `{ reason: 'prompt_too_long' }` 终止查询

**关键：不落入 stop hooks。** 在 prompt-too-long 路径上，模型从未产生有效响应，stop hooks 评估它会产生死亡螺旋：错误 → hook 阻塞 → 重试 → 错误 → ...每个循环注入更多 token。

### 1.5 max-output-tokens 升级恢复

输出 token 限制恢复采用升级式策略，分三个阶段：

**阶段一：8k → 64k 升级**

```typescript
if (capEnabled && maxOutputTokensOverride === undefined &&
    !process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS) {
  // 设置 maxOutputTokensOverride = ESCALATED_MAX_TOKENS 并 continue
}
```

- 仅在未设置自定义输出限制时触发（`maxOutputTokensOverride === undefined`）
- 升级到 64k 无需 meta 消息，直接重试同一请求
- 此升级每个 turn 最多触发一次（由 override 检查守护）
- 由 `tengu_otk_slot_v1` 功能标志控制（第三方默认关闭，因未在 Bedrock/Vertex 上验证）

**阶段二：多轮恢复**

```typescript
if (maxOutputTokensRecoveryCount < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT) {
  const recoveryMessage = createUserMessage({
    content: 'Output token limit hit. Resume directly — no apology, no recap...' ,
    isMeta: true,
  })
  // 将 recoveryMessage 追加到消息列表并 continue
}
```

- 注入 meta 消息指示模型继续生成，从截断处接续
- `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3` 限制恢复轮次
- 每轮递增 `maxOutputTokensRecoveryCount`

**阶段三：暴露错误**

- 恢复计数耗尽后 yield 被暂扣的 max-output-tokens 错误
- 与 prompt-too-long 不同，max-output-tokens 之后可能仍落入 stop hooks（但 `isApiErrorMessage` 检查会短路）

### 1.6 media-size 错误恢复

图片/PDF/多图等媒体大小错误通过 reactive compact 的 strip-retry 路径恢复：

- 与 prompt-too-long 不同，跳过 context collapse drain（collapse 不移除图片）
- `mediaRecoveryEnabled` 门控确保 withhold 和 recover 一致
- 如果超大媒体在保留的尾部中，压缩后仍会触发媒体错误；`hasAttemptedReactiveCompact` 防止螺旋
- 恢复失败时返回 `{ reason: 'image_error' }`

### 1.7 查询循环状态管理

query.ts 使用 `State` 类型在循环迭代间传递可变状态，包含九个字段：

```typescript
type State = {
  messages: Message[]
  toolUseContext: ToolUseContext
  autoCompactTracking: AutoCompactTrackingState | undefined
  maxOutputTokensRecoveryCount: number
  hasAttemptedReactiveCompact: boolean
  maxOutputTokensOverride: number | undefined
  pendingToolUseSummary: Promise<ToolUseSummaryMessage | null> | undefined
  stopHookActive: boolean | undefined
  turnCount: number
  transition: Continue | undefined  // 记录上一次迭代为何 continue
}
```

`transition` 字段特别重要，它让下游恢复路径知道前一次迭代的原因（如 `collapse_drain_retry`、`reactive_compact_retry`、`max_output_tokens_escalate`、`max_output_tokens_recovery`、`stop_hook_blocking`、`token_budget_continuation`），从而避免恢复螺旋。

## 二、压缩层错误恢复

### 2.1 truncateHeadForPTLRetry

`src/services/compact/compact.ts` 中的 `truncateHeadForPTLRetry()` 处理压缩请求自身触发 prompt-too-long 的边界情况。这是一个"简单但安全"的回退——有损地丢弃最旧上下文来解除阻塞。

**执行步骤：**

1. **去除前次标记**：如果第一条消息是 `isMeta` 且内容为 `PTL_RETRY_MARKER`，从输入中移除（否则它独占 group 0，20% 裁剪仅移除标记，零进度）
2. **API 轮次分组**：`groupMessagesByApiRound()` 将消息按 API 调用轮次分组
3. **智能裁剪量计算**：
   - 如果 PTL 响应中包含 token 差距信息（`getPromptTooLongTokenGap`），累加组直到覆盖差距
   - 否则使用 20% 比例裁剪：`Math.max(1, Math.floor(groups.length * 0.2))`
4. **安全边界**：至少保留一个组（`dropCount = Math.min(dropCount, groups.length - 1)`）
5. **合成前导**：裁剪后如果首条消息是 assistant（API 要求首条为 role=user），前置合成用户标记

**重试限制**：`MAX_PTL_RETRIES = 2`，超过后抛出 `ERROR_MESSAGE_PROMPT_TOO_LONG`。

此函数在两个路径被调用：
- **主动/手动压缩路径**：`compactConversation()` 中的主循环
- **部分压缩路径**：`compactPartialConversation()` 中的类似逻辑

### 2.2 熔断器：autoCompact

`src/services/compact/autoCompact.ts` 实现了自动压缩的熔断器机制，防止上下文不可恢复地超出限制时无意义地重复压缩尝试。

**熔断参数：**

```typescript
const MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
```

**追踪状态：**

```typescript
type AutoCompactTrackingState = {
  compacted: boolean
  turnCounter: number
  turnId: string
  consecutiveFailures?: number  // 连续失败计数，成功时重置为0
}
```

**工作流程：**

1. `autoCompactIfNeeded()` 入口检查 `tracking.consecutiveFailures >= 3`
2. 如果达到阈值，直接返回 `{ wasCompacted: false }` 跳过压缩
3. 压缩成功时 `consecutiveFailures = 0`
4. 压缩失败时 `consecutiveFailures` 递增
5. 达到阈值时记录警告：`circuit breaker tripped after N consecutive failures — skipping future attempts this session`

**设计依据**：2026-03-10 的数据显示，1,279 个会话有 50+ 次连续失败（最多 3,272 次），每天浪费约 250K 次 API 调用。熔断器不仅节省 API 调用，还避免了对用户的不必要延迟。

### 2.3 adjustIndexToPreserveAPIInvariants

`src/services/compact/sessionMemoryCompact.ts` 中的 `adjustIndexToPreserveAPIInvariants()` 确保压缩边界不破坏 API 的消息配对约束：

**保护的两个不变量：**

1. **tool_use/tool_result 配对**：如果裁剪边界切断了 tool_use 所在的 assistant 消息但其 tool_result 在保留部分，必须将边界前移以包含 tool_use
   - 收集保留范围内所有 tool_result 的 ID
   - 向前搜索这些 ID 对应的 tool_use
   - 扩展 startIndex 以包含完整的配对

2. **同消息 thinking 块**：同一 `message.id` 的 thinking/redacted_thinking 块不能被分割到压缩边界两侧
   - 收集保留范围内所有 thinking 块的 message ID
   - 向前搜索同 ID 的块
   - 确保 thinking 块的完整消息都在保留范围内

此函数在 sessionMemoryCompact 的多个决策点被调用，是压缩安全性的最后一道防线。

### 2.4 递归守护

`shouldAutoCompact()` 包含多个递归守护：

- `querySource === 'session_memory' || querySource === 'compact'`：forked agent 不会递归触发压缩
- `querySource === 'marble_origami'`：上下文代理的压缩会破坏主线程的 committed log
- reactive-only 模式（`tengu_cobalt_raccoon`）：抑制主动压缩，让 reactive compact 捕获 API 413
- context-collapse 模式：类似抑制，因为 collapse 本身就是上下文管理系统

## 三、Bridge 崩溃恢复

### 3.1 bridgePointer 崩溃恢复指针

`src/bridge/bridgePointer.ts` 实现了 Bridge 会话的崩溃恢复机制，能在 `kill -9` 级别的进程终止后恢复会话。

**指针内容：**

```typescript
type BridgePointer = {
  sessionId: string      // Bridge 会话 ID
  environmentId: string  // Bridge 环境 ID
  source: 'standalone' | 'repl'  // 来源类型
}
```

**生命周期：**

1. **写入时机**：Bridge 会话创建后立即写入
2. **定期刷新**：长会话中每小时重写一次（replBridge.ts 中的 `pointerRefreshTimer`），使用相同内容但更新 mtime
3. **正常关闭**：`clearBridgePointer()` 删除指针文件
4. **异常终止**：指针文件保留，下次启动可恢复

**过期策略：**

- `BRIDGE_POINTER_TTL_MS = 4 * 60 * 60 * 1000`（4 小时）
- 使用文件 mtime 而非内嵌时间戳进行过期检查
- 好处：相同内容的定期重写可以"刷新时钟"，5 小时以上持续轮询的 Bridge 崩溃后仍有新鲜指针
- 超过 4 小时的指针在 `readBridgePointer()` 中被自动删除

**Worktree 感知恢复：**

`readBridgePointerAcrossWorktrees()` 处理 `--continue` 场景中 REPL Bridge 的工作目录可能被 worktree 改变的问题：

1. **快速路径**：检查当前目录（一次 stat 调用），覆盖无 worktree 变化的常见情况
2. **扇出搜索**：`getWorktreePathsPortable()` 获取 git worktree 兄弟（上限 50 个）
3. **并行读取**：对每个候选目录并行调用 `readBridgePointer()`
4. **选择最新**：返回 `ageMs` 最小的指针

### 3.2 进程挂起检测

`src/bridge/replBridge.ts` 实现了进程挂起（非崩溃但无响应）的启发式检测：

```typescript
let suspensionDetected = false

// 在 at-capacity sleep 返回后
const overrun = Date.now() - sleepStart - sleepMs
if (overrun > 60_000) {
  suspensionDetected = true
  logEvent('tengu_bridge_repl_suspension_detected', { overrun_ms: overrun })
}
```

**原理：** 如果一个 sleep 超出其截止时间 60 秒以上，说明进程被挂起（笔记本合盖、SIGSTOP、VM 暂停）——即使是极端的 GC 暂停也只有秒级而非分钟级。检测到挂起后强制执行一个快速轮询周期，因为 `isAtCapacity()` 依赖 `transport !== null`，而 transport 在自动重连期间保持 true，会导致轮询循环直接回到 10 分钟的 sleep。

**辅助检测器：** WebSocketTransport 的 ping 间隔（10 秒粒度）是更短的挂起的主要检测器。sleep 超时检测是当 transport 正在重连（ping 间隔停止）时的后备。

### 3.3 重连策略

Bridge 的重连使用两阶段策略，由 `doReconnect()` 实现：

**策略一：Reconnect-in-place（原地重连）**

```typescript
async function tryReconnectInPlace(requestedEnvId: string, sessionId: string) {
  // 幂等的 re-register，使用 reuseEnvironmentId
  bridgeConfig.reuseEnvironmentId = requestedEnvId
  const reg = await api.registerBridgeEnvironment(bridgeConfig)
  // 如果后端返回相同的 env ID，调用 reconnectSession() 重排队现有会话
  if (environmentId === requestedEnvId) {
    await api.reconnectSession(environmentId, id)
    return true  // 会话 ID 不变，用户手机上的 URL 继续有效
  }
  return false
}
```

优势：currentSessionId 不变、URL 有效、previouslyFlushedUUIDs 保留（避免重复发送历史）。

**策略二：Fresh session fallback（新会话回退）**

当原地重连失败时（环境 ID 不同，意味着原始环境已过期/被回收）：
1. 归档旧会话
2. 在新注册的环境上创建全新会话
3. 重写崩溃恢复指针
4. 清空已刷新的 UUID 集合

**重入保护：** 使用 `reconnectPromise` 实现基于 Promise 的重入守护，确保并发调用者共享同一次重连尝试。

**最大重建次数：** `MAX_ENVIRONMENT_RECREATIONS = 3`，超限后放弃重连。

### 3.4 助手模式（Perpetual Mode）的特殊处理

在助手模式下，Bridge 的行为有所不同：
- 跳过 teardown 时的指针清除，使会话在正常退出后也能恢复
- 每小时 mtime 刷新确保长时间空闲的守护进程不会有过期指针
- 初始启动时读取崩溃恢复指针并尝试 `tryReconnectInPlace`

## 四、远程会话恢复：WebSocket 重连

### 4.1 实现位置

`src/remote/SessionsWebSocket.ts`

### 4.2 重连参数

```typescript
const MAX_RECONNECT_ATTEMPTS = 5
const MAX_SESSION_NOT_FOUND_RETRIES = 3
const RECONNECT_DELAY_MS = ...  // 基础重连延迟
const PING_INTERVAL_MS = 30000  // 心跳间隔
```

### 4.3 关闭码分类处理

```typescript
const PERMANENT_CLOSE_CODES = new Set([4003])  // unauthorized
```

**4001 - Session Not Found（特殊处理）：**

压缩期间服务器可能短暂认为会话过期，因此需要有限重试：

```typescript
if (code === 4001 && sessionNotFoundRetries < MAX_SESSION_NOT_FOUND_RETRIES) {
  sessionNotFoundRetries++
  this.scheduleReconnect(
    RECONNECT_DELAY_MS * this.sessionNotFoundRetries,  // 线性退避
    `4001 attempt ${sessionNotFoundRetries}/${MAX_SESSION_NOT_FOUND_RETRIES}`,
  )
  return
}
```

**4003 - Unauthorized（永久错误）：**

立即停止重连，调用 `onClose` 回调通知上层。

**其他关闭码：**

指数退避重连，最多 5 次尝试。超限后调用 `onClose` 终止。

### 4.4 心跳与连接检测

```typescript
private startPingInterval(): void {
  this.pingInterval = setInterval(() => {
    if (this.ws && this.state === 'connected') {
      try { this.ws.ping?.() } catch { /* close handler 处理 */ }
    }
  }, PING_INTERVAL_MS)
}
```

心跳超时触发 WebSocket 的 close 事件，进入重连流程。

### 4.5 强制重连能力

`forceReconnect()` 方法允许外部触发立即重连（如认证 token 刷新后），绕过正常的退避等待。

## 五、MCP 重连机制

### 5.1 实现位置

`src/services/mcp/useManageMCPConnections.ts`、`src/services/mcp/client.ts`

### 5.2 重连参数

```typescript
const MAX_RECONNECT_ATTEMPTS = 5
const INITIAL_BACKOFF_MS = 1000    // 初始退避 1 秒
const MAX_BACKOFF_MS = 30000       // 最大退避 30 秒
```

### 5.3 指数退避重连

当 MCP 服务器连接断开时，`reconnectWithBackoff()` 执行重连：

```typescript
const reconnectWithBackoff = async () => {
  for (let attempt = 1; attempt <= MAX_RECONNECT_ATTEMPTS; attempt++) {
    // 检查服务器是否在等待期间被禁用
    if (isMcpServerDisabled(client.name)) return

    updateServer({ ...client, type: 'pending', reconnectAttempt: attempt })

    const result = await reconnectMcpServerImpl(client.name, client.config)
    if (result.client.type === 'connected') {
      // 重连成功，清除计时器并更新连接
      reconnectTimersRef.current.delete(client.name)
      onConnectionAttempt(result)
      return
    }

    // 计算退避延迟：1s, 2s, 4s, 8s, 16s，上限 30s
    const backoffMs = Math.min(
      INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1),
      MAX_BACKOFF_MS,
    )
    await new Promise(resolve => {
      const timer = setTimeout(resolve, backoffMs)
      reconnectTimersRef.current.set(client.name, timer)
    })
  }
  // 5 次尝试后标记为断开
}
```

**关键设计：**
- 重连期间服务器状态显示为 `pending`，包含 `reconnectAttempt` 和 `maxReconnectAttempts`
- 重连计时器存储在 `reconnectTimersRef` 中，可在配置变更时取消
- 服务器被禁用后立即停止重连尝试

### 5.4 会话过期检测

在 `client.ts` 中，MCP 连接检测会话过期错误：

```typescript
if ((transportType === 'http' || transportType === 'claudeai-proxy') &&
    isMcpSessionExpiredError(error)) {
  closeTransportAndRejectPending('session expired')
}
```

当服务器返回 404 并包含 session-not-found 信息时，检测为会话过期，触发完整的重连周期（新会话 ID）。

### 5.5 连续错误检测

```typescript
const MAX_ERRORS_BEFORE_RECONNECT = 3
```

对于 SSE 传输，SDK 的自动重连可能不触发 `onclose`。系统追踪连续连接错误，达到 3 次后手动关闭 transport 触发完全重连。

## 六、认证容错

### 6.1 Stale-While-Revalidate 缓存

`src/utils/auth.ts` 中的 `getApiKeyFromApiKeyHelper()` 实现了 API 密钥的 stale-while-revalidate 缓存：

**缓存结构：**

```typescript
let _apiKeyHelperCache: { value: string; timestamp: number } | null = null
let _apiKeyHelperInflight: {
  promise: Promise<string | null>
  startedAt: number | null  // 仅冷启动设置，SWR 后台刷新为 null
} | null = null
```

**SWR 流程：**

1. 缓存命中且未过期：返回缓存值
2. 缓存命中但过期（TTL 已过）：立即返回过期值，启动后台刷新（`startedAt: null`）
3. 冷缓存：等待刷新完成（`startedAt: Date.now()`），并发调用共享同一 Promise

**临时失败处理：**

```typescript
// SWR 路径：临时失败不应将工作密钥替换为 ' ' 哨兵值
if (!isCold && _apiKeyHelperCache && _apiKeyHelperCache.value !== ' ') {
  _apiKeyHelperCache = { value: _apiKeyHelperCache.value, timestamp: Date.now() }
  return _apiKeyHelperCache.value
}
```

后台刷新失败时继续提供旧值，更新时间戳防止每秒重试。

### 6.2 Epoch-based 失效

`_apiKeyHelperEpoch` 是一个单调递增的计数器，防止孤立的异步执行写入过时数据：

```typescript
let _apiKeyHelperEpoch = 0

async function _runAndCache(isNonInteractiveSession, isCold, epoch) {
  try {
    const value = await _executeApiKeyHelper(isNonInteractiveSession)
    if (epoch !== _apiKeyHelperEpoch) return value  // epoch 已变，丢弃结果
    if (value !== null) {
      _apiKeyHelperCache = { value, timestamp: Date.now() }
    }
    return value
  } catch (e) {
    if (epoch !== _apiKeyHelperEpoch) return ' '  // epoch 已变，返回哨兵
    // ... 错误处理
  }
}
```

`clearApiKeyHelperCache()` 递增 epoch，使所有进行中的异步执行在完成时发现自己已过时，不会覆写新的缓存/飞行状态。这在 settings 变更或 401 重试期间尤为重要。

### 6.3 跨进程死 Token 退避

当 API 密钥被验证为无效（401 错误）时，系统不会立即重试，而是实现退避以防止使用已知的死 token 反复请求。这与 `clearApiKeyHelperCache()` 配合，确保新的认证信息能被正确获取。

## 七、通用错误处理模式

### 7.1 Fail-open vs Fail-closed 决策

每个子系统根据其影响范围选择 fail-open 或 fail-closed：

| 子系统 | 策略 | 理由 |
|--------|------|------|
| 认证缓存 | fail-open (SWR) | 过期密钥仍可尝试，比完全阻止好 |
| 自动压缩 | fail-closed (熔断) | 无效压缩浪费 API 调用 |
| MCP 重连 | fail-closed (5次后停止) | 避免对已死服务器持续请求 |
| prompt-too-long | fail-open (尝试恢复) | 恢复可能成功，不应立即放弃 |
| max-output-tokens | fail-open (升级重试) | 更高限制可能解决问题 |
| Bridge 崩溃 | fail-open (提示恢复) | 用户确认后可安全恢复 |

### 7.2 超时层级

系统实现了分层的超时结构，内层超时先于外层触发：

- **工具执行超时**：单个工具调用的执行时限
- **查询循环迭代**：单次 API 调用 + 工具执行的周期
- **心跳间隔**：Bridge/WebSocket 的 30 秒心跳
- **会话 TTL**：Bridge 指针 4 小时过期
- **压缩熔断**：3 次连续失败后停止

### 7.3 优雅降级策略

所有恢复路径穷尽后的统一降级行为：

1. **详细日志**：通过 `logError()` 和 `logAntError()` 记录完整错误上下文
2. **用户消息**：`createAssistantAPIErrorMessage()` 生成简洁错误，不含内部细节
3. **会话保持**：对话历史完整保留，用户可手动重试
4. **内容保护**：不丢失用户已输入的内容和已完成的工具执行结果
5. **遥测上报**：所有恢复路径记录遥测事件用于监控和改进

### 7.4 遥测事件清单

| 事件名 | 触发条件 |
|--------|----------|
| `tengu_model_fallback_triggered` | 模型回退触发 |
| `tengu_orphaned_messages_tombstoned` | 流式回退时孤儿消息标记 |
| `tengu_max_tokens_escalate` | 输出 token 升级 |
| `tengu_compact_failed` | 压缩失败 |
| `tengu_compact_ptl_retry` | 压缩 PTL 重试 |
| `tengu_auto_compact_succeeded` | 自动压缩成功 |
| `tengu_query_error` | 查询层未捕获错误 |
| `tengu_bridge_repl_suspension_detected` | 进程挂起检测 |
| `tengu_api_custom_529_overloaded_error` | 外部用户 529 错误 |

## 八、错误恢复策略总结

| 错误类型 | 恢复策略 | 最大重试 | 源文件 |
|----------|----------|----------|--------|
| API 529 过载 | withRetry 指数退避 | 可配置 | withRetry.ts |
| 模型不可用 | FallbackTriggered 回退 | 1 次 | query.ts |
| prompt-too-long (查询层) | collapse drain → reactiveCompact | 各1次 | query.ts |
| prompt-too-long (压缩层) | truncateHeadForPTLRetry | 2 次 | compact.ts |
| max-output-tokens | 8k→64k 升级 + 多轮 meta | 3 轮 | query.ts |
| media-size | reactiveCompact strip-retry | 1 次 | query.ts |
| 连续压缩失败 | 熔断器 | 3 次 | autoCompact.ts |
| Bridge 崩溃 | bridgePointer 恢复 | 1 次（用户确认） | bridgePointer.ts |
| Bridge 环境丢失 | reconnect-in-place → fresh session | 3 次重建 | replBridge.ts |
| 进程挂起 | sleep 超时启发式检测 | 自动 | replBridge.ts |
| WebSocket 4001 | 有限线性退避重试 | 3 次 | SessionsWebSocket.ts |
| WebSocket 通用 | 指数退避重连 | 5 次 | SessionsWebSocket.ts |
| WebSocket 4003 | 停止重连 | 0 次 | SessionsWebSocket.ts |
| MCP 断开 | 指数退避重连 | 5 次 | useManageMCPConnections.ts |
| MCP 会话过期 | 完全重连周期 | 含在5次内 | client.ts |
| 认证缓存过期 | stale-while-revalidate | 自动 | auth.ts |
| 认证 epoch 冲突 | 丢弃过时结果 | 自动 | auth.ts |

## 关键源文件

| 文件 | 职责 |
|------|------|
| `src/query.ts` | 查询层错误恢复协调、withheld 机制、多路径恢复 |
| `src/services/api/withRetry.ts` | API 重试、FallbackTriggeredError 定义 |
| `src/services/compact/compact.ts` | 压缩层 PTL 重试（truncateHeadForPTLRetry） |
| `src/services/compact/autoCompact.ts` | 自动压缩熔断器（MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES） |
| `src/services/compact/reactiveCompact.ts` | 反应式压缩恢复（413 后触发） |
| `src/services/compact/sessionMemoryCompact.ts` | adjustIndexToPreserveAPIInvariants |
| `src/services/contextCollapse/index.ts` | Context collapse drain 恢复 |
| `src/bridge/bridgePointer.ts` | Bridge 崩溃恢复指针（读写、过期检查、worktree 感知） |
| `src/bridge/replBridge.ts` | Bridge 重连策略（reconnect-in-place、进程挂起检测） |
| `src/bridge/bridgeMain.ts` | 独立 Bridge 的退避配置和心跳 |
| `src/remote/SessionsWebSocket.ts` | WebSocket 重连策略（关闭码分类、心跳） |
| `src/services/mcp/useManageMCPConnections.ts` | MCP 指数退避重连 |
| `src/services/mcp/client.ts` | MCP 会话过期检测、连续错误计数 |
| `src/utils/auth.ts` | 认证 SWR 缓存、epoch 失效 |
