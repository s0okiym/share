# 守护进程与后台任务

## 概述

Claude Code 的守护进程（Daemon）和后台任务系统构成了一个完整的长时运行操作管理框架，允许耗时任务在后台执行而不阻塞主会话的交互。守护进程是一个持久运行的监督器，管理多个并发会话的生命周期；后台任务涵盖六种类型——Bash 命令、本地代理、主会话后台化、Dream 记忆巩固、MCP 监控和定时调度。整个系统通过 CLI 子命令（`ps`/`logs`/`attach`/`kill`）、任务注册表（`AppState.tasks`）、UI 组件（底部药丸标签和任务对话框）和工具接口（TaskOutputTool/ScheduleCronTool）提供统一的管理与交互能力。

## 守护进程架构

```mermaid
flowchart TD
    A["cli.tsx 入口"] --> B{"--daemon-worker?"}
    B -- "是" --> C["runDaemonWorker()<br/>工作进程注册"]
    A --> D{"daemon 子命令?"}
    D -- "是" --> E["daemonMain()<br/>守护进程监督器"]
    A --> F{"ps/logs/attach/kill<br/>或 --bg/--background?"}
    F -- "是" --> G["会话管理命令"]
    A --> H["正常交互 REPL"]

    E --> I["持久进程管理"]
    I --> J["多会话监督"]

    J --> K["LocalShellTask<br/>后台Bash命令"]
    J --> L["LocalAgentTask<br/>后台代理"]
    J --> M["LocalMainSessionTask<br/>主会话后台化"]
    J --> N["DreamTask<br/>记忆巩固"]
    J --> O["MonitorMcpTask<br/>MCP监控"]
    J --> P["ScheduleCronTool<br/>定时调度"]

    K --> Q["stall watchdog<br/>停滞看门狗"]
    K --> R["output file<br/>输出文件"]

    L --> S["ProgressTracker<br/>进度追踪"]
    S --> T["toolUseCount<br/>tokenCount<br/>recentActivities"]

    M --> U["Ctrl+B x2<br/>后台化主查询"]
    U --> V["startBackgroundSession()<br/>独立query()调用"]

    N --> W["DreamPhase<br/>starting→updating"]
    W --> X["filesTouched<br/>turns<br/>priorMtime"]

    P --> Y["CronCreateTool"]
    P --> Z["CronDeleteTool"]
    P --> AA["CronListTool"]

    Y --> BB["durable: .claude/scheduled_tasks.json"]
    Y --> CC["session-only: 内存存储"]

    G --> DD["ps: 列出活跃会话"]
    G --> EE["logs: 查看会话日志"]
    G --> FF["attach: 附加到会话"]
    G --> GG["kill: 终止会话"]

    HH["AppState.tasks"] --> II["任务注册表"]
    II --> JJ["pillLabel.ts<br/>底部药丸标签"]
    II --> KK["BackgroundTasksDialog<br/>Shift+Down对话框"]

    style E fill:"#6cf",stroke:"#333"
    style K fill:"#6f6",stroke:"#333"
    style L fill:"#6f6",stroke:"#333"
    style M fill:"#6f6",stroke:"#333"
    style N fill:"#6f6",stroke:"#333"
    style P fill:"#ff6",stroke:"#333"
```

## 一、CLI 入口与守护进程命令

### 1.1 --daemon-worker 路径

```typescript
if (feature('DAEMON') && args[0] === '--daemon-worker') {
  const { runDaemonWorker } = await import('../daemon/workerRegistry.js')
  await runDaemonWorker(args[1])
  return
}
```

这是内部标志，由守护进程监督器为每个工作进程生成。关键设计：

- **检查优先级**：必须在 `daemon` 子命令检查之前，因为工作进程是按需生成的，对启动延迟敏感
- **轻量级加载**：不调用 `enableConfigs()` 和分析接收器，减少模块加载时间
- **延迟初始化**：需要配置/认证的工作类型（如 assistant）在自己的 `run()` 函数内部调用

### 1.2 daemon 子命令

```typescript
if (feature('DAEMON') && args[0] === 'daemon') {
  const { daemonMain } = await import('../daemon/main.js')
  await daemonMain(args.slice(1))
  return
}
```

启动长运行监督器进程，接受子命令参数控制行为。守护进程作为持久进程管理多个并发会话。

### 1.3 会话管理命令

```typescript
if (feature('BG_SESSIONS') && (
  args[0] === 'ps' || args[0] === 'logs' ||
  args[0] === 'attach' || args[0] === 'kill' ||
  args.includes('--bg') || args.includes('--background'))) {
  const bg = await import('../cli/bg.js')
  switch (args[0]) {
    case 'ps':    await bg.psHandler(args.slice(1)); break
    case 'logs':  await bg.logsHandler(args[1]); break
    case 'attach':await bg.attachHandler(args[1]); break
    case 'kill':  await bg.killHandler(args[1]); break
    default:      await bg.handleBgFlag(args)
  }
}
```

- **ps**：列出所有活跃的 Claude Code 会话（PID、会话 ID、CWD、类型、状态）
- **logs**：查看指定会话的实时日志
- **attach**：附加到指定会话的 tmux 窗口
- **kill**：终止指定会话
- **--bg / --background**：以后台模式启动新会话，通过 tmux 管理生命周期

## 二、守护进程架构

### 2.1 会话注册系统

守护进程和后台会话通过 `~/.claude/sessions/` 目录中的 PID 文件进行注册和发现：

- **会话类型**：`interactive`、`bg`、`daemon`、`daemon-worker`
- **会话状态**：`busy`、`idle`、`waiting`
- **注册流程**：`registerSession()` 写入 PID 文件，包含进程 PID、会话 ID、工作目录、类型和状态
- **清理机制**：`registerCleanup()` 确保进程退出时删除 PID 文件
- **存活检测**：`isProcessRunning()` 通过信号 0 验证进程是否仍在运行
- **类型判定**：`CLAUDE_CODE_SESSION_KIND` 环境变量设置会话类型

### 2.2 守护进程作为持久监督器

守护进程的监督器模式：
- 管理多个工作进程的生命周期
- 每个工作进程通过 `--daemon-worker=<kind>` 标志生成
- `workerRegistry.ts` 分发到对应的处理函数
- 监控工作进程健康状态，处理崩溃重启
- 与 Bridge 系统集成，支持远程会话的持久化

## 三、后台任务类型

### 3.1 LocalShellTask：后台 Bash 命令

`src/tasks/LocalShellTask/LocalShellTask.tsx` 处理后台 Bash 命令的执行、监控和通知。

#### 停滞看门狗（Stall Watchdog）

后台 Bash 命令可能因等待交互式输入而无限阻塞。`startStallWatchdog()` 通过监控输出增长率检测这种情况：

**监控参数：**

```typescript
const STALL_CHECK_INTERVAL_MS = 5_000   // 每 5 秒检查一次输出文件大小
const STALL_THRESHOLD_MS = 45_000       // 45 秒无增长判定为停滞
const STALL_TAIL_BYTES = 1024           // 读取尾部 1024 字节用于提示检测
```

**检测流程：**

1. 每 5 秒 `stat()` 输出文件获取大小
2. 如果大小增长，更新 `lastSize` 和 `lastGrowth` 时间戳
3. 如果 45 秒无增长，读取文件尾部 1024 字节
4. `looksLikePrompt()` 检查最后一行是否匹配交互式提示模式
5. 匹配则发送通知：`Background command "xxx" appears to be waiting for interactive input`
6. 不匹配则重置 `lastGrowth`，继续监控（区分真正的交互式等待和慢速输出）
7. 通知发送后设置 `cancelled = true` 停止监控

**提示模式检测：**

```typescript
const PROMPT_PATTERNS = [
  /\(y\/n\)/i,        // (Y/n), (y/N)
  /\[y\/n\]/i,        // [Y/n], [y/N]
  /\(yes\/no\)/i,     // (yes/no)
  /\b(?:Do you|Would you|Shall I|Are you sure|Ready to)\b.*\? *$/i,
  /Press (any key|Enter)/i,
  /Continue\?/i,
  /Overwrite\?/i,
]
```

**通知方式：** 通过 `enqueuePendingNotification()` 将通知入队到消息队列，使用 `task-notification` 模式和 `next` 优先级，确保模型在下一轮看到通知并可以采取行动（如 kill 任务并以非交互方式重新运行）。

**类型排除：** `kind === 'monitor'` 的任务跳过看门狗（监控任务预期是长时运行的）。

#### 通知去重

`enqueueShellNotification()` 使用原子性的 `notified` 标志防止重复通知：

```typescript
updateTaskState(taskId, setAppState, task => {
  if (task.notified) return task  // 已通知，跳过
  shouldEnqueue = true
  return { ...task, notified: true }
})
```

这避免了 TaskStopTool 和任务完成回调同时发送通知的竞争条件。

### 3.2 LocalAgentTask：后台本地代理

`src/tasks/LocalAgentTask/LocalAgentTask.tsx` 处理后台代理（subagent）的执行与进度追踪。

#### 进度追踪系统

```typescript
export type ProgressTracker = {
  toolUseCount: number               // 工具使用总次数
  latestInputTokens: number          // 最新输入 token（API 累积值）
  cumulativeOutputTokens: number     // 累积输出 token
  recentActivities: ToolActivity[]   // 最近5次工具活动
}

export type ToolActivity = {
  toolName: string
  input: Record<string, unknown>
  activityDescription?: string   // 预计算的活动描述，如 "Reading src/foo.ts"
  isSearch?: boolean             // 是否搜索操作（Grep、Glob等）
  isRead?: boolean               // 是否读取操作（Read、cat等）
}
```

**关键设计：** `latestInputTokens` 使用最新值而非累加，因为 API 的 `input_tokens` 在每轮中是累积的（包含所有前序上下文），而 `output_tokens` 是每轮的，需要求和。`getTokenCountFromTracker()` 返回 `latestInputTokens + cumulativeOutputTokens`。

**活动描述预计算：** `createActivityDescriptionResolver()` 从工具列表创建解析器，在记录时通过 `tool.getActivityDescription(input)` 预计算描述，避免在 UI 渲染时重复查找。

**搜索/读取分类：** `getToolSearchOrReadInfo()` 将工具活动分类为搜索或读取操作，用于 UI 展示的视觉区分。

#### 任务状态

```typescript
export type LocalAgentTaskState = TaskStateBase & {
  type: 'local_agent'
  agentId: string
  prompt: string
  selectedAgent?: AgentDefinition
  agentType: string
  abortController?: AbortController
  progress?: AgentProgress
  isBackgrounded: boolean          // 是否在后台
  pendingMessages: string[]        // SendMessage 排队的消息
  retain: boolean                  // UI 是否持有此任务
  diskLoaded: boolean              // 是否已从磁盘加载 JSONL
  evictAfter?: number              // 面板关闭后的宽限截止时间
}
```

**面板任务判定：** `isPanelAgentTask()` 区分协调器面板管理的代理任务和主会话任务——`agentType !== 'main-session'` 的本地代理任务在面板中渲染而非后台任务药丸。

### 3.3 LocalMainSessionTask：主会话后台化

`src/tasks/LocalMainSessionTask.ts` 处理用户在查询运行期间按两次 Ctrl+B 将主会话后台化的场景。

#### 注册与启动

```typescript
export function registerMainSessionTask(
  description: string,
  setAppState: SetAppState,
  mainThreadAgentDefinition?: AgentDefinition,
  existingAbortController?: AbortController,
): { taskId: string; abortSignal: AbortSignal }
```

- 使用 `s` 前缀的任务 ID（8 个随机字母数字字符），区分代理任务的 `a` 前缀
- 复用现有的 `AbortController`（重要：确保中止任务能中止实际的查询）
- 输出链接到隔离的转录文件（`getAgentTranscriptPath`），不使用主会话文件
- 默认 `isBackgrounded: true`（注册时已在后台）

#### startBackgroundSession

`startBackgroundSession()` 启动独立的后台查询：

```typescript
export function startBackgroundSession({
  messages, queryParams, description, setAppState, agentDefinition,
}): string {
  const { taskId, abortSignal } = registerMainSessionTask(...)

  // 持久化预后台化的对话到任务的隔离转录
  void recordSidechainTranscript(messages, taskId).catch(...)

  void runWithAgentContext(agentContext, async () => {
    const bgMessages: Message[] = [...messages]
    const recentActivities: ToolActivity[] = []
    let toolCount = 0, tokenCount = 0

    for await (const event of query({ messages: bgMessages, ...queryParams })) {
      if (abortSignal.aborted) { /* 中止处理 */ return }

      bgMessages.push(event)
      // 逐条写入转录（匹配 runAgent.ts 模式）
      void recordSidechainTranscript([event], taskId, lastRecordedUuid).catch(...)

      // 更新进度追踪
      if (event.type === 'assistant') {
        // 提取工具计数、token 计数、最近活动
      }

      setAppState(prev => ({ /* 更新任务进度和消息 */ }))
    }
    completeMainSessionTask(taskId, true, setAppState)
  })

  return taskId
}
```

**关键设计：**

- **AsyncLocalStorage 隔离**：`runWithAgentContext()` 为后台查询创建独立的代理上下文，使技能调用范围限定到任务的 `agentId`，不影响前台
- **逐条转录写入**：每条消息立即写入磁盘，即使 `/clear` 在运行中重链接符号链接，转录也保持最新
- **中止安全**：检查 `abortSignal.aborted`，中止时正确设置 `notified` 标志并发出 SDK 终止事件

#### 前台恢复

`foregroundMainSessionTask()` 将后台任务恢复到前台：

```typescript
export function foregroundMainSessionTask(
  taskId: string, setAppState: SetAppState,
): Message[] | undefined {
  // 恢复先前前台任务到后台
  // 设置 foregroundedTaskId
  // 返回任务累积的消息
}
```

#### 完成通知

完成通知仅在任务仍处于后台时发送。已前台化的任务直接由用户观看，不需要 XML 通知，但仍需设置 `notified: true` 以满足驱逐守卫条件，并发出 SDK `task_terminated` 事件。

### 3.4 DreamTask：记忆巩固代理

`src/tasks/DreamTask/DreamTask.ts` 是记忆巩固代理（/dream 技能）的后台任务入口，使原本不可见的 forked 代理在 UI 中可见。

#### 状态结构

```typescript
export type DreamTaskState = TaskStateBase & {
  type: 'dream'
  phase: DreamPhase               // 'starting' | 'updating'
  sessionsReviewing: number       // 审查的会话数
  filesTouched: string[]          // 观察到被编辑/写入的路径
  turns: DreamTurn[]              // 助手文本响应（工具调用折叠为计数）
  abortController?: AbortController
  priorMtime: number              // 整合锁的先前 mtime（kill 时回滚用）
}
```

#### 阶段演进

Dream prompt 有 4 阶段结构（orient/gather/consolidate/prune），但系统不显式解析这些阶段：

- **starting**：初始状态
- **updating**：当第一个 Edit/Write tool_use 到达时切换（`newTouched.length > 0`）

#### 文件追踪

`filesTouched` 通过 `addDreamTurn()` 的 `touchedPaths` 参数从 Edit/Write tool_use 块中提取路径。这是不完整的——bash 介导的写入不会被捕获。应视为"至少触及了这些文件"。

#### Kill 与锁回滚

```typescript
async kill(taskId, setAppState) {
  let priorMtime: number | undefined
  updateTaskState<DreamTaskState>(taskId, setAppState, task => {
    if (task.status !== 'running') return task
    task.abortController?.abort()
    priorMtime = task.priorMtime
    return { ...task, status: 'killed', notified: true }
  })
  // 回滚锁 mtime 使下次会话可以重试
  if (priorMtime !== undefined) {
    await rollbackConsolidationLock(priorMtime)
  }
}
```

`priorMtime` 存储整合锁（consolidation lock）的先前修改时间，kill 时回滚使下次会话可以重新尝试整合。与 `autoDream.ts` 中的 fork 失败路径相同。

#### Turn 限制

`MAX_TURNS = 30` 保留最近的 30 轮用于实时显示，使用 `slice(-(MAX_TURNS - 1)).concat(turn)` 滑动窗口实现。

### 3.5 MonitorMcpTask：MCP 监控

`src/tasks/MonitorMcpTask/MonitorMcpTask.ts`（通过 feature flag `MONITOR_TOOL` 启用）追踪 MCP 连接的健康状态，在服务断开或异常时通知用户。作为后台任务注册到 `AppState.tasks`，与其他任务类型共享统一的生命周期管理。

### 3.6 ScheduleCronTool：定时调度

详见第六节。

## 四、任务输出与通信

### 4.1 TaskOutputTool

`src/tools/TaskOutputTool/TaskOutputTool.tsx` 是获取后台任务结果的标准工具接口。

**输入参数：**

```typescript
z.strictObject({
  task_id: z.string().describe('要获取输出的任务ID'),
  block: z.boolean().default(true).describe('是否等待完成'),
  timeout: z.number().min(0).max(600000).default(30000).describe('最大等待时间(ms)')
})
```

**输出格式（统一）：**

```typescript
type TaskOutput = {
  task_id: string
  task_type: TaskType    // 'local_bash' | 'local_agent' | 'remote_agent' | ...
  status: string         // 'running' | 'completed' | 'failed'
  description: string
  output: string
  exitCode?: number | null
  error?: string
  prompt?: string        // 仅代理
  result?: string        // 仅代理
}
```

**按类型输出获取：**

- **local_bash**：优先使用 `shellCommand.taskOutput`（实时对象），回退到 `getTaskOutput(taskId)`（磁盘读取）
- **其他类型**：统一使用 `getTaskOutput(taskId)` 读取磁盘上的输出文件

**进度流式传输：** `block: true` 时，工具在等待期间提供增量进度更新，显示任务当前状态。

### 4.2 后台任务状态存储

所有后台任务状态存储在 `AppState.tasks` 中：

```typescript
type AppState = {
  tasks: Record<string, TaskState>  // 任务ID → 任务状态
  foregroundedTaskId?: string       // 当前前台化的任务ID
  // ...
}
```

**任务类型联合：**

```typescript
type TaskState =
  | LocalShellTaskState
  | LocalAgentTaskState
  | RemoteAgentTaskState
  | InProcessTeammateTaskState
  | LocalWorkflowTaskState
  | MonitorMcpTaskState
  | DreamTaskState
```

**后台任务判定：**

```typescript
function isBackgroundTask(task: TaskState): task is BackgroundTaskState {
  if (task.status !== 'running' && task.status !== 'pending') return false
  if ('isBackgrounded' in task && task.isBackgrounded === false) return false
  return true
}
```

前台运行的任务（`isBackgrounded === false`）不算"后台任务"，避免在药丸标签中显示。

### 4.3 PillLabel 系统

`src/tasks/pillLabel.ts` 实现了底部状态栏的药丸标签，提供后台任务的即时概览。

**getPillLabel()：**

根据任务类型和数量生成紧凑的标签文本：

| 条件 | 标签 |
|------|------|
| 单个 local_bash | "1 shell" |
| 单个 monitor 类型 | "1 monitor" |
| 单个 local_agent | "1 local agent" |
| 单个 remote_agent (ultraplan ready) | "◆ ultraplan ready" |
| 单个 remote_agent (needs input) | "◇ ultraplan needs your input" |
| 单个 remote_agent (普通) | "◇ 1 cloud session" |
| 单个 dream | "dreaming" |
| 单个 local_workflow | "1 background workflow" |
| 混合类型 | "N background tasks" |

**pillNeedsCta()：**

判定药丸是否应显示 "· ↓ to view" 的行动号召。仅在单个远程代理任务处于 `needs_input` 或 `plan_ready` 状态时返回 true——这些是需要用户注意的"注意力状态"，普通运行状态仅显示标签。

**设计原则：** 同类型任务合并为复数形式（"3 shells"），不同类型分别计数（"1 shell, 2 monitors"），极简显示不干扰主交互。

## 五、Dream 代理

### 5.1 DreamPhase 进展

DreamTask 使用简单的两阶段模型：

1. **starting**：代理刚启动，尚未执行任何编辑操作
2. **updating**：第一个 Edit/Write tool_use 到达时切换，表示记忆正在被修改

实际的 Dream prompt 内部有 4 阶段（orient → gather → consolidate → prune），但这是提示工程层面的结构，系统不解析它。两阶段模型足以驱动 UI 展示——用户只关心"正在思考"和"正在修改记忆"的区别。

### 5.2 文件触及追踪

`filesTouched` 数组追踪 Dream 代理观察到的文件变更：

```typescript
export function addDreamTurn(
  taskId: string, turn: DreamTurn, touchedPaths: string[], setAppState,
): void {
  updateTaskState<DreamTaskState>(taskId, setAppState, task => {
    const seen = new Set(task.filesTouched)
    const newTouched = touchedPaths.filter(p => !seen.has(p) && seen.add(p))
    // 空轮次且无新文件 → 跳过更新（避免无意义重渲染）
    if (turn.text === '' && turn.toolUseCount === 0 && newTouched.length === 0) {
      return task
    }
    return {
      ...task,
      phase: newTouched.length > 0 ? 'updating' : task.phase,
      filesTouched: newTouched.length > 0 ? [...task.filesTouched, ...newTouched] : task.filesTouched,
      turns: task.turns.slice(-(MAX_TURNS - 1)).concat(turn),
    }
  })
}
```

**去重机制**：使用 `Set` 确保同一文件路径不会重复添加。

**空轮次优化**：纯文本轮次（无工具调用、无新文件）跳过状态更新，避免频繁的 React 重渲染。

### 5.3 完成与失败处理

- **完成**：`completeDreamTask()` 设置 `status: 'completed'`、`notified: true`、清除 `abortController`。Dream 没有面向模型的通知路径（纯 UI），`notified: true` 立即设置以满足驱逐条件
- **失败**：`failDreamTask()` 同样设置终端状态和 `notified: true`

## 六、定时调度：ScheduleCronTool

### 6.1 CronCreateTool

`src/tools/ScheduleCronTool/CronCreateTool.ts` 创建定时任务。

**输入模式：**

```typescript
z.strictObject({
  cron: z.string().describe(
    '标准5字段cron表达式（本地时间）：M H DoM Mon DoW'
  ),
  prompt: z.string().describe('每次触发时入队的提示'),
  recurring: z.boolean().optional().describe(
    'true（默认）= 每次cron匹配时触发直到删除或7天后过期。' +
    'false = 下次匹配时触发一次后自动删除'
  ),
  durable: z.boolean().optional().describe(
    'true = 持久化到.claude/scheduled_tasks.json，跨重启存活。' +
    'false（默认）= 仅内存存储，会话结束时消失'
  ),
})
```

**验证规则：**

1. `parseCronExpression(input.cron)` 验证 cron 表达式语法
2. `nextCronRunMs(input.cron, Date.now()) !== null` 验证未来一年内有匹配日期
3. `tasks.length < MAX_JOBS`（50 个）限制任务数量
4. Teammate 不允许 durable（teammate 不跨会话存活，durable cron 会孤立）

**输出：**

```typescript
type CreateOutput = {
  id: string             // 8 字符十六进制短 ID
  humanSchedule: string  // cronToHuman() 转换的人类可读格式
  recurring: boolean
  durable?: boolean
}
```

### 6.2 CronDeleteTool

删除指定的定时任务。调用 `removeCronTasks([id])` 同时清理内存和磁盘存储。

### 6.3 Cron 任务存储

`src/utils/cronTasks.ts` 管理任务的持久化和内存存储。

**CronTask 类型：**

```typescript
type CronTask = {
  id: string
  cron: string
  prompt: string
  createdAt: number        // 创建时间（epoch ms）
  lastFiredAt?: number     // 最近触发时间（仅重复任务）
  recurring?: boolean      // 是否重复
  permanent?: boolean      // 是否永不过期（系统任务）
  durable?: boolean        // 运行时标记，不写入磁盘
  agentId?: string         // Teammate 关联的 agentId，不写入磁盘
}
```

**双存储模型：**

- **磁盘存储**：`.claude/scheduled_tasks.json`，仅存储 `durable: true` 的任务
- **内存存储**：`bootstrap/state.ts` 中的 `addSessionCronTask()`，会话结束时消失
- **合并视图**：`listAllCronTasks()` 合并两种存储，内存任务标记 `durable: false`

**添加流程：**

```typescript
async function addCronTask(cron, prompt, recurring, durable, agentId?): Promise<string> {
  const id = randomUUID().slice(0, 8)  // 8 字符十六进制短 ID
  if (!durable) {
    addSessionCronTask({ ...task, ...(agentId ? { agentId } : {}) })
    return id
  }
  const tasks = await readCronTasks()
  tasks.push(task)
  await writeCronTasks(tasks)
  return id
}
```

**删除流程：**

1. 先扫描内存存储（`removeSessionCronTasks`）
2. 如果所有 ID 都在内存中找到，跳过文件 I/O
3. 否则读写磁盘文件移除剩余 ID

**触发标记：** `markCronTasksFired()` 在每次重复任务触发后写入 `lastFiredAt`，使调度器在进程重启后能从正确的时间点恢复。批量写入（N 次触发 = 1 次读写）。

### 6.4 抖动与过期配置

`CronJitterConfig` 控制调度器的抖动和过期行为：

```typescript
type CronJitterConfig = {
  recurringFrac: number         // 重复任务前向延迟比例（默认 0.1）
  recurringCapMs: number        // 重复延迟上限（默认 15 分钟）
  oneShotMaxMs: number          // 一次性任务最大提前量（默认 90 秒）
  oneShotFloorMs: number        // 一次性任务最小提前量（默认 0）
  oneShotMinuteMod: number      // 抖动分钟取模（默认 30，即 :00/:30）
  recurringMaxAgeMs: number     // 重复任务过期时间（默认 7 天）
}
```

**重复任务抖动（`jitteredNextCronRunMs`）：**

- 延迟 = `jitterFrac(taskId) * recurringFrac * (t2 - t1)`，上限 `recurringCapMs`
- `jitterFrac` 基于任务 ID 的 8 字符十六进制前缀解析为 u32，确定性且均匀分布
- 目的：避免同一 cron 字符串的大量会话在相同时刻命中推理服务（如 `0 * * * *` → 所有人在 :00 请求）

**一次性任务抖动（`oneShotJitteredNextCronRunMs`）：**

- 用户指定的时间点（如"3pm提醒我"）不能延迟，但可以稍微提前
- 仅在 `minute % oneShotMinuteMod === 0` 时应用（默认仅 :00 和 :30）
- 提前量 = `oneShotFloorMs + jitterFrac * (oneShotMaxMs - oneShotFloorMs)`
- 不能早于任务创建时间（`Math.max(t1 - lead, fromMs)`）

**过期机制：**

- 重复任务在 `recurringMaxAgeMs`（默认 7 天）后自动过期
- 标记 `permanent: true` 的任务永不过期（助手模式的内置任务）
- 安装器的 `writeIfMissing()` 跳过已存在的文件，所以 permanent 任务不能被重建

### 6.5 遗漏任务检测

`findMissedTasks()` 检测在 Claude 离线期间应该触发但未触发的任务：

```typescript
function findMissedTasks(tasks: CronTask[], nowMs: number): CronTask[] {
  return tasks.filter(t => {
    const next = nextCronRunMs(t.cron, t.createdAt)
    return next !== null && next < nowMs
  })
}
```

这些遗漏任务在启动时向用户报告，让他们知道有计划的操作未被执行。

### 6.6 调度器核心

`src/utils/cronScheduler.ts` 实现调度器的核心循环：

- **检查间隔**：`CHECK_INTERVAL_MS = 1000`（每秒检查）
- **文件稳定性**：`FILE_STABILITY_MS = 300`（文件变更后等待 300ms 稳定）
- **锁探测**：`LOCK_PROBE_INTERVAL_MS = 5000`（非拥有会话的锁探测间隔）
- **锁机制**：`tryAcquireSchedulerLock()` 确保同一时间只有一个会话执行调度
- **文件监视**：chokidar 监视 `scheduled_tasks.json` 的变更，触发重新加载
- **错过任务处理**：启动时检测遗漏任务，通过 `findMissedTasks()` 识别

## 七、后台会话管理

### 7.1 会话生命周期

1. **启动**：`claude --bg` 或 `claude --background` 通过 `bg.handleBgFlag()` 启动
2. **注册**：`registerSession()` 写入 PID 文件，`CLAUDE_CODE_SESSION_KIND=bg`
3. **运行**：在 tmux 会话中运行，用户可以断开终端
4. **管理**：`ps`/`logs`/`attach`/`kill` 命令管理活跃会话
5. **断开**：在 BG 会话中 Ctrl+C/Ctrl+D 是 detach 而非终止
6. **清理**：`registerCleanup()` 确保退出时删除 PID 文件

### 7.2 tmux 集成

后台会话通过 tmux 管理进程生命周期：
- `createTmuxSessionForWorktree()` 为 worktree 创建 tmux 会话
- `generateTmuxSessionName()` 生成唯一会话名
- 用户可以通过 `claude attach <session>` 或直接 `tmux attach` 重新连接
- tmux 提供了进程隔离、断开重连和输出持久化

### 7.3 进程信号处理

| 信号 | 行为 |
|------|------|
| SIGTERM | 优雅关闭，执行清理操作 |
| SIGINT | BG 会话中 detach 而非终止 |
| SIGHUP | 终端关闭时保持运行（tmux 保护） |

## 八、任务框架

### 8.1 任务注册表

`src/utils/task/framework.ts` 提供任务注册和管理的核心基础设施：

- `registerTask(task, setAppState)`：注册新任务到 `AppState.tasks`
- `updateTaskState<T>(taskId, setAppState, updater)`：类型安全的状态更新
- `PANEL_GRACE_MS`：面板关闭前的宽限时间（任务完成后短暂保留在 UI 中）

### 8.2 任务状态基础

```typescript
type TaskStateBase = {
  id: string
  type: string
  description: string
  status: 'running' | 'completed' | 'failed' | 'killed' | 'pending'
  startTime: number
  endTime?: number
  toolUseId?: string
  notified?: boolean           // 是否已发送通知（防重复）
}
```

所有任务类型扩展此基础，添加特定字段。

### 8.3 任务停止

`src/tasks/stopTask.ts` 提供统一的任务停止逻辑：

```typescript
async function stopTask(taskId: string, context: StopTaskContext): Promise<StopTaskResult> {
  const task = appState.tasks?.[taskId]
  if (!task) throw new StopTaskError('No task found', 'not_found')
  if (task.status !== 'running') throw new StopTaskError('Not running', 'not_running')

  const taskImpl = getTaskByType(task.type)
  await taskImpl.kill(taskId, setAppState)

  // Bash: 抑制 "exit code 137" 通知（噪声）
  // Agent: 不抑制——AbortError catch 发送有用信息
}
```

### 8.4 清理注册

`registerCleanup()` 确保进程退出时执行清理：
- 删除 PID 文件
- 关闭连接
- 释放锁
- Dream 代理：回滚整合锁的 mtime

## 关键源文件

| 文件 | 职责 |
|------|------|
| `src/entrypoints/cli.tsx` | CLI 入口：daemon/bg 命令分发 |
| `src/tasks/LocalShellTask/LocalShellTask.tsx` | 后台 Bash 任务（含 stall watchdog） |
| `src/tasks/LocalAgentTask/LocalAgentTask.tsx` | 后台代理任务（含 ProgressTracker） |
| `src/tasks/LocalMainSessionTask.ts` | 主会话后台化（startBackgroundSession） |
| `src/tasks/DreamTask/DreamTask.ts` | 记忆巩固任务 |
| `src/tasks/pillLabel.ts` | 底部药丸标签（getPillLabel/pillNeedsCta） |
| `src/tasks/types.ts` | 任务类型联合与判定函数 |
| `src/tasks/stopTask.ts` | 统一任务停止逻辑 |
| `src/tools/TaskOutputTool/TaskOutputTool.tsx` | 任务结果获取工具 |
| `src/tools/ScheduleCronTool/CronCreateTool.ts` | 定时任务创建 |
| `src/tools/ScheduleCronTool/CronDeleteTool.ts` | 定时任务删除 |
| `src/utils/cronTasks.ts` | 定时任务存储（双模型、抖动、过期） |
| `src/utils/cronScheduler.ts` | 定时调度核心循环 |
| `src/utils/task/framework.ts` | 任务框架基础设施 |
| `src/utils/task/diskOutput.ts` | 任务输出磁盘管理 |
