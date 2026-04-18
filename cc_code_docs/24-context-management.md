# 上下文管理

## 概述

上下文管理模块负责组装Claude Code对话中注入的系统上下文和用户上下文。该模块采用薄层记忆化（memoized）设计，将Git状态、CLAUDE.md文件内容、当前日期等信息组装为结构化字典，供系统提示构建器使用。核心设计原则是"计算一次，缓存整个对话"，通过lodash-es/memoize实现缓存，并在需要时通过`.cache.clear()`精确失效。

## context.ts - 核心模块

`src/context.ts`是上下文管理的核心入口，提供两个主要的记忆化函数：`getSystemContext()`和`getUserContext()`。

### getSystemContext() - 系统上下文

```typescript
export const getSystemContext = memoize(
  async (): Promise<{ [k: string]: string }> => { ... }
)
```

返回一个以上下文节名称为键的字符串字典，每个会话计算一次。

**组装内容**：

1. **gitStatus**：Git仓库状态快照
   - 在CCR远程模式（`CLAUDE_CODE_REMOTE`环境变量）下跳过（不必要的开销）
   - 当Git指令被禁用时跳过（`shouldIncludeGitInstructions()`）
   - 内容包括当前分支、主分支、Git用户名、状态（截断到2000字符）、最近5条提交

2. **cacheBreaker**：缓存破坏标记（ant-only）
   - 仅在BREAK_CACHE_COMMAND特性启用且`systemPromptInjection`被设置时包含
   - 格式：`[CACHE_BREAKER: ${injection}]`
   - 用于强制系统提示缓存失效

**设计决策**：返回字典而非拼接字符串，允许系统提示构建器灵活地安排各节的位置和格式。

### getUserContext() - 用户上下文

```typescript
export const getUserContext = memoize(
  async (): Promise<{ [k: string]: string }> => { ... }
)
```

返回用户特定的上下文字典。

**组装内容**：

1. **claudeMd**：从CLAUDE.md文件读取的项目指令
   - 通过`getMemoryFiles()`获取所有内存文件（目录遍历）
   - 通过`filterInjectedMemoryFiles()`过滤注入的内存文件
   - 通过`getClaudeMds()`解析CLAUDE.md内容
   - 结果缓存到`setCachedClaudeMdContent()`供自动模式分类器使用

2. **currentDate**：当前日期
   - 格式：`Today's date is ${getLocalISODate()}.`

**Bare模式处理**：
- `CLAUDE_CODE_DISABLE_CLAUDE_MDS`环境变量：硬关闭，始终跳过
- `--bare`模式：跳过自动发现（cwd遍历），但遵循显式的`--add-dir`
- bare模式意味着"跳过我没要求的"，而非"忽略我要求的"

### getGitStatus() - Git状态

```typescript
export const getGitStatus = memoize(async (): Promise<string | null> => { ... })
```

独立的记忆化函数，执行并行的Git命令获取状态：

**并行执行**：
- `getBranch()`：当前分支
- `getDefaultBranch()`：默认分支（通常是main/master）
- `git status --short`：简短状态
- `git log --oneline -n 5`：最近5条提交
- `git config user.name`：用户名

**截断处理**：
- 最大2000字符（MAX_STATUS_CHARS）
- 超出时截断并附加提示消息："truncated because it exceeds 2k characters. If you need more information, run 'git status' using BashTool"

**跳过条件**：
- 测试环境（`NODE_ENV === 'test'`）
- 非Git目录（`getIsGit()`返回false）

**输出格式**：
```
This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.

Current branch: feature/auth
Main branch (you will usually use this for PRs): main
Git user: developer
Status:
M src/auth.ts
?? new-file.ts

Recent commits:
abc1234 Fix auth bug
def5678 Add tests
```

### getSystemPromptInjection() / setSystemPromptInjection()

ant-only的缓存破坏机制：

```typescript
let systemPromptInjection: string | null = null

export function getSystemPromptInjection(): string | null {
  return systemPromptInjection
}

export function setSystemPromptInjection(value: string | null): void {
  systemPromptInjection = value
  // 立即清除上下文缓存
  getUserContext.cache.clear?.()
  getSystemContext.cache.clear?.()
}
```

设置新的注入值时，立即清除两个上下文缓存，确保下次读取时重新计算。

### 记忆化与缓存失效

两个核心函数都使用lodash-es的`memoize`，支持`.cache.clear()`精确失效：

**自动失效时机**：
- `setSystemPromptInjection()`被调用时
- `getUserContext.cache.clear?.()`
- `getSystemContext.cache.clear?.()`

**手动失效**：系统提示节清除时（`clearSystemPromptSections()`），也会触发相关缓存失效。

### 循环依赖避免

上下文模块通过`cachedClaudeMdContent`在引导状态（bootstrap state）中打破循环依赖：

- `yoloClassifier.ts`需要读取CLAUDE.md内容
- 但直接导入`claudemd.ts`会创建循环：permissions/filesystem -> permissions -> yoloClassifier
- 解决方案：getUserContext()计算claudeMd后，通过`setCachedClaudeMdContent()`写入引导状态缓存
- yoloClassifier读取缓存值而非直接导入

这种模式避免了模块图中常见的循环依赖问题，同时保持了延迟计算和记忆化的效率。

## constants/prompts.ts - 系统提示构建

`src/constants/prompts.ts`是系统提示的构建中心，使用`systemPromptSection()`和`DANGEROUS_uncachedSystemPromptSection()`框架组织提示节。

### systemPromptSection() - 记忆化提示节

```typescript
function systemPromptSection(
  name: string,
  compute: ComputeFn
): SystemPromptSection
```

- 计算一次，缓存直到`/clear`或`/compact`
- `cacheBreak`标志为false
- 适用于相对稳定的内容（工具列表、环境信息等）

### DANGEROUS_uncachedSystemPromptSection() - 非缓存提示节

```typescript
function DANGEROUS_uncachedSystemPromptSection(
  name: string,
  compute: ComputeFn,
  _reason: string  // 必须说明为什么需要破坏缓存
): SystemPromptSection
```

- 每轮重新计算
- 当值变化时会破坏提示缓存
- `_reason`参数强制开发者解释缓存破坏的必要性
- 适用于每轮变化的动态内容

### 特性门控提示节

系统提示构建大量使用特性门控（feature gates）来条件性包含提示节：

- **Dead code elimination**：条件导入使用`require()`，确保未启用的模块不进入bundle
- **KAIROS**：assistant模式的提示节
- **EXPERIMENTAL_SKILL_SEARCH**：技能搜索工具提示
- **CACHED_MICROCOMPACT**：缓存微压缩配置
- **PROACTIVE**：主动式Agent提示
- **KAIROS_BRIEF**：Brief工具提示

这些门控确保只有当前特性所需的代码被加载和执行，减少bundle大小和运行时开销。

### 提示节解析

```typescript
async function resolveSystemPromptSections(
  sections: SystemPromptSection[]
): Promise<(string | null)[]>
```

解析所有提示节，返回提示字符串数组：

1. 对于非缓存破坏节：检查缓存，命中则直接返回
2. 未命中或缓存破坏节：执行compute函数，存入缓存
3. 缓存键为节名称，存储在`getSystemPromptSectionCache()`中

### 提示节清除

```typescript
function clearSystemPromptSections(): void {
  clearSystemPromptSectionState()
  clearBetaHeaderLatches()
}
```

在`/clear`和`/compact`时调用，重置所有提示节状态，同时重置beta头锁存器，确保新对话获得新鲜的AFK/fast-mode/cache-editing头评估。

## constants/systemPromptSections.ts - 提示节框架

`src/constants/systemPromptSections.ts`定义了系统提示节的框架抽象：

### SystemPromptSection类型

```typescript
type SystemPromptSection = {
  name: string
  compute: ComputeFn
  cacheBreak: boolean
}
```

- `name`：节名称，用作缓存键
- `compute`：计算函数，返回string | null | Promise<string | null>
- `cacheBreak`：是否破坏缓存

### 缓存机制

提示节缓存与对话生命周期绑定：

- **创建时机**：新对话开始时（`/clear`后首次调用）
- **失效时机**：`/clear`或`/compact`时
- **存储位置**：引导状态的`systemPromptSectionCache`映射
- **键**：节名称字符串
- **值**：compute函数的返回值（可为null）

### 记忆化变体与易失性变体

两种变体的核心区别在于缓存行为：

| 特性 | 记忆化变体 | 易失性变体 |
|---|---|---|
| 计算频率 | 每个对话一次 | 每轮一次 |
| 缓存破坏 | 否 | 是 |
| 适用场景 | 静态内容 | 动态内容 |
| 命名约定 | systemPromptSection() | DANGEROUS_uncachedSystemPromptSection() |
| 安全性 | 缓存友好 | 谨慎使用 |

## constants/xml.ts - XML标签名

`src/constants/xml.ts`定义了结构化通信中使用的XML标签名：

### 终端输出标签
- `bash-input` / `bash-stdout` / `bash-stderr`
- `local-command-stdout` / `local-command-stderr` / `local-command-caveat`

### 任务通知标签
- `task-notification` / `task-id` / `tool-use-id`
- `task-type` / `output-file` / `status` / `summary` / `reason`

### 协作标签
- `teammate-message`：Agent间通信
- `channel-message` / `channel`：外部通道消息
- `cross-session-message`：跨会话消息

### 特殊标签
- `tick`：时间标记
- `ultraplan`：远程并行规划
- `remote-review` / `remote-review-progress`：远程审查结果
- `worktree` / `worktreePath` / `worktreeBranch`：工作树信息
- `fork-boilerplate` / `fork-directive-prefix`：Fork子进程样板

这些标签用于在消息中嵌入结构化元数据，使解析器能够提取和处理特定类型的信息，而不影响消息的主体文本。

## 上下文组装流程图

```mermaid
flowchart TD
    A["对话开始"] --> B["getSystemContext()"]
    A --> C["getUserContext()"]

    B --> D{"CCR远程模式?"}
    D -->|是| E["跳过Git状态"]
    D -->|否| F["getGitStatus()"]

    F --> G["并行执行Git命令"]
    G --> G1["getBranch()"]
    G --> G2["getDefaultBranch()"]
    G --> G3["git status --short"]
    G --> G4["git log --oneline -n 5"]
    G --> G5["git config user.name"]

    G1 --> H["组装gitStatus字符串"]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H

    H --> I{"BREAK_CACHE_COMMAND?"}
    I -->|是且injection非null| J["添加cacheBreaker节"]
    I -->|否| K["仅包含gitStatus"]

    C --> L{"禁用CLAUDE.md?"}
    L -->|是| M["跳过claudeMd"]
    L -->|否| N["getMemoryFiles()"]

    N --> O["目录遍历查找CLAUDE.md"]
    O --> P["filterInjectedMemoryFiles()"]
    P --> Q["getClaudeMds()"]
    Q --> R["setCachedClaudeMdContent()"]
    R --> S["返回claudeMd"]

    S --> T["添加currentDate"]

    K --> U["系统上下文字典"]
    T --> V["用户上下文字典"]

    U --> W["systemPromptSection()"]
    V --> W

    W --> X["resolveSystemPromptSections()"]
    X --> Y["检查缓存"]
    Y -->|命中| Z["返回缓存值"]
    Y -->|未命中| AA["执行compute()"]
    AA --> AB["存入缓存"]
    AB --> Z

    AC["setSystemPromptInjection()"] --> AD["清除所有上下文缓存"]
    AE["/clear 或 /compact"] --> AF["clearSystemPromptSections()"]
    AF --> AG["清除提示节状态"]
    AG --> AH["清除beta头锁存器"]
