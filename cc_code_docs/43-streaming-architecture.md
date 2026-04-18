# 流式响应架构

## 概述

Claude Code 的流式响应架构贯穿整个系统，从 API 调用到 UI 渲染，实现了端到端的实时数据流处理。核心思想是：API 响应不是等待完整生成后一次性返回，而是通过流式传输逐步到达，使工具可以在完整响应完成前就开始执行，UI 可以即时展示进度。这种设计显著降低了用户感知延迟，并实现了工具并发执行、思维过程可视化等高级功能。

## 流式数据流

```mermaid
flowchart TD
    A["API 流式响应<br/>Anthropic Messages API"] --> B["query.ts<br/>for-await-of 循环消费"]
    B --> C["StreamEvent 事件流"]
    C --> D["StreamingToolExecutor<br/>工具即时执行"]
    C --> E["REPL.tsx<br/>handleMessageFromStream()"]
    D --> F["getCompletedResults()<br/>非阻塞结果获取"]
    D --> G["getRemainingResults()<br/>等待剩余结果"]
    F --> H["MessageUpdate 消息更新"]
    G --> H
    H --> I["setMessages()<br/>React 状态更新"]
    E --> J["streamingToolUses<br/>流式工具使用状态"]
    E --> K["streamingThinking<br/>流式思维状态"]
    J --> I
    K --> I
    I --> L["Ink 渲染引擎<br/>渐进式 UI 更新"]
    L --> M["终端输出"]

    D --> N{"流式回退?"}
    N -- "是" --> O["discard() 丢弃结果"]
    O --> P["墓碑消息<br/>孤立消息重置"]
    P --> Q["重置执行器<br/>重新请求"]

    style D fill:"#6cf",stroke:"#333"
    style L fill:"#6f6",stroke:"#333"
    style O fill:"#f96",stroke:"#333"
```

## 一、流式 API 消费：query.ts

### 核心消费模式

`src/query.ts` 使用 `for-await-of` 循环消费流式 API 响应。这是整个流式架构的起点——API 返回的不是完整消息，而是一系列增量事件（StreamEvent），每个事件携带一部分生成内容。

### 流式事件类型

流式响应中可能产生的事件包括：

- **content_block_start**：新的内容块开始（text、tool_use、thinking 等）
- **content_block_delta**：内容块增量更新（文本片段、工具输入 JSON 片段、思维片段）
- **content_block_stop**：内容块结束
- **message_start**：消息开始，携带 usage 信息
- **message_delta**：消息级更新
- **message_stop**：消息结束

### yield 机制

`query()` 函数是一个 async generator，通过 `yield` 将流式事件逐个传递给上层消费者。每个 yield 的消息（Message）已经经过解析和格式化，包含了工具调用、思维内容或普通文本。

## 二、流式工具执行器：StreamingToolExecutor

### 设计目标

`src/services/tools/StreamingToolExecutor.ts` 是流式架构中最关键的优化组件。传统模式下，系统需要等待完整响应（包括所有 tool_use 块）后才开始执行工具。StreamingToolExecutor 允许工具在 tool_use 块流式到达时立即开始执行，显著降低延迟。

### 核心机制

```typescript
export class StreamingToolExecutor {
  private tools: TrackedTool[] = []
  private toolUseContext: ToolUseContext
  private hasErrored = false
  private siblingAbortController: AbortController
  private discarded = false
}
```

### 工具追踪

每个工具调用被包装为 `TrackedTool`：

```typescript
type TrackedTool = {
  id: string
  block: ToolUseBlock
  assistantMessage: AssistantMessage
  status: ToolStatus  // 'queued' | 'executing' | 'completed' | 'yielded'
  isConcurrencySafe: boolean
  promise?: Promise<void>
  results?: Message[]
  pendingProgress: Message[]
  contextModifiers?: Array<(context: ToolUseContext) => ToolUseContext>
}
```

### 添加工具：addTool()

当流式响应中的 `tool_use` 块到达时：

1. **查找工具定义**：`findToolByName()` 定位对应的 Tool 对象。未找到时生成 "No such tool available" 错误结果
2. **解析输入**：`toolDefinition.inputSchema.safeParse(block.input)` 验证输入格式
3. **并发安全检查**：调用 `toolDefinition.isConcurrencySafe(parsedInput.data)` 判断工具是否可并发执行
4. **入队并处理**：将工具添加到 `tools` 数组，触发 `processQueue()`

### 并发控制

`canExecuteTool()` 实现并发控制逻辑：

- **无执行中工具**：可以执行
- **并发安全工具**：如果所有当前执行中的工具都是并发安全的，可以并行执行
- **非并发安全工具**：必须独占执行（排他访问）

### 进度消息

进度消息（`type === 'progress'`）通过 `pendingProgress` 数组独立存储，可立即 yield 给消费者，无需等待工具完成。`progressAvailableResolve` 信号用于唤醒 `getRemainingResults()` 中等待的消费者。

### 结果获取

**非阻塞获取**：`getCompletedResults()` 是一个同步 generator，返回已完成且未 yield 的结果。保持工具顺序——非并发安全工具必须按序 yield。

**等待获取**：`getRemainingResults()` 是一个 async generator，在工具执行期间持续等待并 yield 结果。使用 `Promise.race` 等待执行中的工具或进度信号。

## 三、流式事件处理：handleMessageFromStream

### 实现位置

`src/utils/messages.ts` 导出的 `handleMessageFromStream()` 函数。

### 消费者

- **REPL.tsx**：`onQueryEvent` 回调调用 `handleMessageFromStream` 处理流式事件
- **useRemoteSession.ts**：远程会话中处理流式事件

### 处理逻辑

函数接收流式事件并更新 React 状态：

1. **tool_use 块到达**：更新 `streamingToolUses` 状态，显示工具调用进度
2. **thinking 块到达**：更新 `streamingThinking` 状态，显示思维过程
3. **完整消息到达**：添加到 `messages` 数组，触发 UI 重渲染
4. **压缩边界消息**：特殊处理，标记压缩点

## 四、流式状态管理

### streamingToolUses

REPL.tsx 中的 `streamingToolUses` 状态追踪当前正在流式传输的工具调用：

```typescript
const [streamingToolUses, setStreamingToolUses] = useState<StreamingToolUse[]>([])
```

每个 `StreamingToolUse` 包含工具 ID、名称、输入（增量更新）和状态。UI 组件可以实时显示工具调用进度，如 "Reading src/foo.ts..." → "Reading src/foo.ts (42 lines)"。

### streamingThinking

`streamingThinking` 状态追踪当前正在流式传输的思维块：

```typescript
const [streamingThinking, setStreamingThinking] = useState<StreamingThinking | null>(null)
```

思维块在完成后 30 秒自动隐藏（`setTimeout` 设置 30 秒计时器）。

### 流式模式

`streamMode` 状态控制流式处理行为，`streamModeRef` 确保回调中始终使用最新值。

## 五、SDK/转录回填

### tool_use 输入回填

流式传输期间，tool_use 块的输入是增量到达的。SDK 和转录系统需要完整的 tool_use 输入。回填机制在 `tool_use` 块完成时（`content_block_stop` 事件）将增量输入合并为完整 JSON，确保：

- SDK 消费者（如 VS Code 扩展）收到完整的工具调用信息
- 转录文件记录完整的工具输入
- `inProgressToolUseIDs` 在工具完成后被移除

### 消息规范化

`normalizeMessagesForAPI()` 确保消息格式符合 API 要求，处理流式传输中可能出现的边界情况（如空 content blocks）。

## 六、流式回退处理

### 回退触发

当流式响应出现错误或需要重试时（如模型回退 `FallbackTriggeredError`、连接中断），系统执行流式回退：

1. **丢弃执行器**：`StreamingToolExecutor.discard()` 标记所有待处理和执行中的工具为已丢弃
2. **生成墓碑消息**：为孤立消息（已有 tool_use 但无 tool_result 的消息）生成墓碑标记
3. **重置执行器**：创建新的 `StreamingToolExecutor` 实例
4. **重新请求**：使用备用模型或简化上下文重新发起 API 请求

### discard() 机制

```typescript
discard(): void {
  this.discarded = true
}
```

标记后：
- 排队的工具不会启动
- 执行中的工具收到合成错误消息（`streaming_fallback` 原因）
- `getCompletedResults()` 和 `getRemainingResults()` 立即返回空

### 合成错误消息

`createSyntheticErrorMessage()` 为被丢弃的工具生成错误消息，原因包括：

- `sibling_error`：并行工具调用中的一个出错
- `user_interrupted`：用户中断（使用 `REJECT_MESSAGE` 显示 "User rejected"）
- `streaming_fallback`：流式回退，工具执行被丢弃

## 七、兄弟工具错误传播

### Bash 错误级联

当 Bash 工具执行出错时，`StreamingToolExecutor` 取消所有兄弟工具：

```typescript
if (tool.block.name === BASH_TOOL_NAME) {
  this.hasErrored = true
  this.erroredToolDescription = this.getToolDescription(tool)
  this.siblingAbortController.abort('sibling_error')
}
```

设计理由：Bash 命令通常有隐式依赖链（如 `mkdir` 失败 → 后续命令无意义）。而 Read/WebFetch 等独立工具的错误不会级联——一个失败不应终止其余工具。

### 中止控制器层次

```
toolUseContext.abortController (父: 查询级)
  └── siblingAbortController (兄: 工具组级)
        └── toolAbortController (子: 单工具级)
```

- 父控制器中止时，所有工具中止
- 兄控制器中止时，所有执行中的工具中止（兄弟错误级联）
- 子控制器中止时，仅该工具中止（权限拒绝等）

## 八、Ink 渲染集成

### 渐进式 UI 更新

流式事件通过 React 状态更新触发 Ink 渲染引擎的重渲染。由于 Ink 使用虚拟 DOM diff，只有变化的部分会更新终端输出，避免全屏重绘。

### 终端标题动画

960ms 动画 tick 用于终端标题更新，显示当前查询状态（如 "Claude Code - Running query..."）。这个间隔平衡了视觉流畅性和性能开销。

### 消息渲染优化

- **消息分组**：连续的读/搜索操作被折叠显示
- **工具活动摘要**：`generateToolUseSummary()` 为完成的工具生成简洁摘要
- **进度条**：长时间运行的工具显示实时进度

## 九、与查询引擎的集成

### QueryEngine.ts

查询引擎协调流式响应和工具执行的完整生命周期：

1. **发起查询**：调用 `query()` 获取 async generator
2. **处理事件**：从 generator 读取事件，传递给 `handleMessageFromStream`
3. **工具执行**：`StreamingToolExecutor` 管理工具并发执行
4. **结果合并**：将工具执行结果合并到对话历史
5. **循环继续**：如果模型请求更多工具调用，继续循环

### 背压处理

流式系统通过 `for-await-of` 循环自然实现背压——如果 UI 渲染慢于事件到达速度，generator 会暂停，API 连接会缓冲数据。这防止了内存无限增长。

## 十、性能优化

### 延迟优化

1. **工具即时执行**：StreamingToolExecutor 在 tool_use 块到达时立即执行，不等完整响应
2. **并发安全工具并行**：多个并发安全的工具可同时运行
3. **进度即时 yield**：进度消息无需等待工具完成
4. **增量渲染**：Ink 只重绘变化部分

### 内存优化

1. **流式消费**：不缓冲完整响应，逐事件处理
2. **工具结果按序 yield**：完成后立即 yield，不累积
3. **进度消息独立队列**：不与结果消息混合

## 关键源文件

| 文件 | 职责 |
|------|------|
| `src/query.ts` | 流式 API 响应消费 |
| `src/services/tools/StreamingToolExecutor.ts` | 流式工具并发执行器 |
| `src/utils/messages.ts` | handleMessageFromStream 流式事件处理 |
| `src/screens/REPL.tsx` | 流式状态管理（streamingToolUses/streamingThinking） |
| `src/hooks/useRemoteSession.ts` | 远程会话流式处理 |
| `src/services/tools/toolOrchestration.ts` | 工具编排 |
| `src/services/api/claude.ts` | API 流式请求 |
| `src/services/api/withRetry.ts` | 流式重试与回退 |
