# 多Agent架构与协调

## 概述

Claude Code的多Agent架构位于`src/tasks/`目录和相关工具模块，实现了丰富的任务类型层次结构，支持从本地Shell命令到云端远程Agent、从单线程协作到群体（swarm）并行执行的多种工作模式。系统采用任务注册-状态更新模式，通过统一的Task接口管理所有异步执行单元的生命周期，并在coordinator模式下实现Agent间的协调与通信。

## 任务类型层次结构

### Task基础类型

`src/Task.ts`定义了任务系统的核心类型：

```typescript
type TaskType =
  | 'local_bash'        // 本地Shell命令
  | 'local_agent'       // 本地Agent
  | 'remote_agent'      // 远程Agent（云端/传送）
  | 'in_process_teammate' // 进程内队友（swarm）
  | 'local_workflow'    // 工作流脚本
  | 'monitor_mcp'       // MCP监控
  | 'dream'             // 记忆巩固

type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'killed'
```

TaskStateBase包含所有任务类型共享的基础字段：

- `id`：任务ID（带类型前缀）
- `type`：任务类型
- `status`：任务状态
- `description`：任务描述
- `toolUseId`：关联的工具使用ID
- `startTime` / `endTime`：时间戳
- `outputFile` / `outputOffset`：输出文件路径和偏移
- `notified`：是否已发送完成通知

Task接口定义了多态调度入口：

```typescript
type Task = {
  name: string
  type: TaskType
  kill(taskId: string, setAppState: SetAppState): Promise<void>
}
```

### TaskState联合类型

`src/tasks/types.ts`将所有具体任务状态类型汇总为联合类型：

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

### LocalShellTask - 本地Shell任务

`src/tasks/LocalShellTask/guards.ts`定义了本地Shell命令任务的状态类型：

```typescript
type LocalShellTaskState = TaskStateBase & {
  type: 'local_bash'
  command: string
  result?: { code: number; interrupted: boolean }
  completionStatusSentInAttachment: boolean
  shellCommand: ShellCommand | null
  lastReportedTotalLines: number
  isBackgrounded: boolean
  agentId?: AgentId
  kind?: 'bash' | 'monitor'
}
```

**失速看门狗（Stall Watchdog）**：LocalShellTask实现了失速检测机制，监控子进程的输出活动。如果长时间无输出，判定为失速并通知用户。

**Agent关联**：`agentId`字段记录生成此任务的Agent，当Agent退出时，关联的Shell任务会被自动清理（`killShellTasksForAgent`）。

**UI变体**：`kind`字段区分显示变体——`bash`显示命令，`monitor`显示描述文本和专用状态栏标识。

### LocalAgentTask - 本地Agent任务

LocalAgentTask是最常用的后台任务类型，执行自主Agent的完整工作流。

**进度追踪**：
- `toolUseCount`：工具使用次数
- `tokenCount`：token使用量估算
- `recentActivities`：最近工具活动列表（最多5条）
- `lastReportedToolCount` / `lastReportedTokenCount`：用于计算增量通知

**状态结构**：
- `agentId`：Agent标识
- `selectedAgent`：Agent定义（可选）
- `agentType`：Agent类型标识
- `abortController`：中止控制器
- `isBackgrounded`：是否已后台化
- `pendingMessages`：待发送消息队列
- `messages`：对话历史
- `diskLoaded`：是否从磁盘加载

### RemoteAgentTask - 远程Agent任务

RemoteAgentTask在云端CCR容器上执行Agent，本地通过轮询获取进度。

**轮询与心跳解析**：
- 定期轮询远程会话状态
- 解析心跳中的进度信息（如run_hunt.sh的progress.json）
- 远程会话输出通过`<remote-review-progress>` XML标签传递进度

**传送模式**：
- 将本地任务"传送"到云端执行
- 支持SSH隧道和代理配置
- 结果通过`<remote-review>` XML标签返回

### InProcessTeammateTask - 进程内队友任务

`src/tasks/InProcessTeammateTask/types.ts`定义了swarm模式下的队友任务。

**队友身份**：
```typescript
type TeammateIdentity = {
  agentId: string       // 如 "researcher@my-team"
  agentName: string     // 如 "researcher"
  teamName: string
  color?: string
  planModeRequired: boolean
  parentSessionId: string  // 领导者的会话ID
}
```

**消息上限**：`TEAMMATE_MESSAGES_UI_CAP = 50`，task.messages仅保留最近50条消息用于缩放视图。完整对话保存在本地allMessages数组和磁盘转录文件中。BQ分析显示，500+轮会话中每个Agent约20MB RSS，292个并发Agent可达36.8GB，主要成本是消息数组的第二份完整副本。

**消息追加与截断**：
```typescript
function appendCappedMessage<T>(prev: readonly T[] | undefined, item: T): T[]
```
超过50条时丢弃最旧的消息，始终返回新数组（AppState不可变性）。

**生命周期管理**：
- `isIdle`：队友是否空闲
- `shutdownRequested`：是否请求关闭
- `onIdleCallbacks`：空闲时通知领导者的回调数组
- `awaitingPlanApproval`：等待计划审批
- `permissionMode`：独立的权限模式

**双级中止**：
- `abortController`：终止整个队友
- `currentWorkAbortController`：仅中止当前轮次

### LocalMainSessionTask - 后台主会话任务

`src/tasks/LocalMainSessionTask.ts`处理主会话查询的后台化。

**触发条件**：用户在查询期间按Ctrl+B两次。

**行为**：
- 查询继续在后台运行
- UI清空为新的提示符
- 查询完成时发送通知

复用LocalAgentTaskState结构，`agentType`设为`'main-session'`，`isBackgrounded`初始为true。

**任务ID前缀**：使用`'s'`前缀（vs Agent的`'a'`前缀）。

**前台化**：`foregroundMainSessionTask()`将后台任务切换回前台，恢复之前的显示状态。

### LocalWorkflowTask - 工作流脚本任务

执行预定义的工作流脚本，以结构化方式运行多步骤操作。

### MonitorMcpTask - MCP监控任务

监控MCP（Model Context Protocol）服务器的连接状态，检测断连和重连事件。

### DreamTask - 记忆巩固任务

`src/tasks/DreamTask/DreamTask.ts`实现记忆巩固功能，在空闲期间处理和整合会话记忆，优化长期存储的检索效率。

## 任务注册与状态更新

### registerTask() - 任务注册

通过`registerTask(taskState, setAppState)`将任务注册到AppState：

```typescript
function registerTask(taskState: TaskState, setAppState: SetAppState): void
```

注册后任务出现在AppState.tasks映射中，UI开始显示进度。

### updateTaskState() - 状态更新

通过`updateTaskState(taskId, setAppState, updater)`更新任务状态：

```typescript
function updateTaskState<T extends TaskState>(
  taskId: string,
  setAppState: SetAppState,
  updater: (task: T) => T | T
): void
```

updater函数接收当前状态，返回新状态。如果任务不存在或状态未变化，跳过更新。

### 后台任务判断

```typescript
function isBackgroundTask(task: TaskState): task is BackgroundTaskState {
  if (task.status !== 'running' && task.status !== 'pending') return false
  if ('isBackgrounded' in task && task.isBackgrounded === false) return false
  return true
}
```

仅running/pending且未被前台化的任务才视为后台任务。

## 团队管理工具

### TeamCreateTool

创建新的Agent团队。支持配置：
- 团队名称
- 成员Agent列表
- 各成员的角色和工具权限

### TeamDeleteTool

解散现有团队。清理：
- 终止所有运行中的成员Agent
- 释放团队资源
- 从AppState中移除团队记录

## Agent间通信

### SendMessageTool

向指定Agent发送消息，实现Agent间通信：

```typescript
// 继续现有Agent的执行
SendMessageTool({ to: "agent-a1b", message: "Fix the null pointer..." })
```

关键特性：
- `to`字段使用Agent ID（来自AgentTool的启动结果或task-notification）
- 消息在Agent的当前上下文中执行
- 被停止的Agent仍可通过SendMessage继续

### ListPeersTool

列出当前会话中的所有对等Agent，显示：
- Agent ID
- Agent名称
- 当前状态（运行中/空闲/已停止）
- 工具使用统计

## 协调器模式

`src/coordinator/coordinatorMode.ts`实现了协调器（coordinator）模式，允许一个主Agent协调多个工作Agent的执行。

### 启用条件

```typescript
function isCoordinatorMode(): boolean {
  if (feature('COORDINATOR_MODE')) {
    return isEnvTruthy(process.env.CLAUDE_CODE_COORDINATOR_MODE)
  }
  return false
}
```

需要同时满足特性标志和环境变量。

### 会话模式匹配

`matchSessionMode()`确保恢复的会话使用正确的模式：

- 如果会话存储的模式与当前环境变量不匹配，自动翻转环境变量
- 返回警告消息（如"Entered coordinator mode to match resumed session."）
- 防止模式不匹配导致的工具访问错误

### 协调器用户上下文

`getCoordinatorUserContext()`注入工作Agent的工具上下文信息：

- 列出工作Agent可用的工具（排除内部工具：TeamCreate、TeamDelete、SendMessage、SyntheticOutput）
- MCP服务器名称列表
- Scratchpad目录路径（如果启用tengu_scratch门控）

### COORDINATOR_MODE_ALLOWED_TOOLS

协调器模式下允许的工具集定义在`src/constants/tools.ts`的`ASYNC_AGENT_ALLOWED_TOOLS`中，包括AgentTool、SendMessageTool、TaskStopTool等协调工具，以及subscribe_pr_activity/unsubscribe_pr_activity等PR事件工具。

### 协调器系统提示

`getCoordinatorSystemPrompt()`生成完整的协调器角色定义，包含以下核心指导：

**1. 角色定义**：协调器，不直接执行任务，而是指导worker进行研究、实现和验证。

**2. 工具使用**：
- AgentTool：生成新worker
- SendMessageTool：继续现有worker
- TaskStopTool：停止错误方向的worker
- 不用worker检查另一个worker
- 不设置model参数

**3. 任务通知格式**：
```xml
<task-notification>
<task-id>{agentId}</task-id>
<status>completed|failed|killed</status>
<summary>{human-readable status summary}</summary>
<result>{agent's final text response}</result>
<usage>
  <total_tokens>N</total_tokens>
  <tool_uses>N</tool_uses>
  <duration_ms>N</duration_ms>
</usage>
</task-notification>
```

**4. 任务工作流**：

| 阶段 | 执行者 | 目的 |
|---|---|---|
| 研究 | Workers（并行） | 调查代码库、查找文件、理解问题 |
| 综合 | 协调器 | 阅读发现、理解问题、制定实现规范 |
| 实现 | Workers | 按规范进行定向修改 |
| 验证 | Workers | 测试变更是否有效 |

**5. Worker提示编写原则**：
- Worker看不到协调器的对话，每个提示必须自包含
- 必须综合研究发现后再编写实现规范
- 禁止"基于你的发现"等懒惰委托表达
- 包含目的声明，让Worker校准深度
- 高上下文重叠时继续现有Worker，低重叠时生成新Worker

**6. 并发管理**：
- 只读任务自由并行
- 写密集任务按文件集合串行化
- 验证可与实现并行（不同文件区域）

**7. 验证要求**：
- 运行启用功能的测试
- 调查类型检查错误
- 独立测试，不橡皮图章
- 对可疑结果深入调查

## 多Agent协调流程图

```mermaid
flowchart TD
    A["用户请求"] --> B{"isCoordinatorMode()?"}
    B -->|否| C["单Agent直接执行"]
    B -->|是| D["协调器模式"]

    D --> E["分析任务 分解子任务"]
    E --> F["并行启动研究Workers"]

    F --> G1["AgentTool: 研究Worker1"]
    F --> G2["AgentTool: 研究Worker2"]
    F --> G3["AgentTool: 研究Worker3"]

    G1 --> H["task-notification 到达"]
    G2 --> H
    G3 --> H

    H --> I["协调器综合研究发现"]
    I --> J["编写实现规范"]

    J --> K{"上下文重叠度?"}
    K -->|高| L["SendMessageTool 继续Worker"]
    K -->|低| M["AgentTool 生成新Worker"]

    L --> N["实现Worker执行"]
    M --> N

    N --> O["task-notification: 完成或失败"]

    O --> P{"实现成功?"}
    P -->|是| Q["AgentTool: 验证Worker"]
    P -->|否| R{"需要修正?"}
    R -->|是| S["SendMessageTool 修正指令"]
    R -->|否| T["报告用户失败"]

    S --> N
    Q --> U["验证Worker独立测试"]
    U --> V{"验证通过?"}
    V -->|是| W["报告用户成功"]
    V -->|否| R

    X["ListPeersTool"] --> Y["列出所有对等Agent"]
    Z["TeamCreateTool"] --> AA["创建团队配置"]
    AB["TeamDeleteTool"] --> AC["解散团队 清理资源"]
