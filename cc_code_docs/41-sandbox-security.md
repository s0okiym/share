# 沙箱与安全隔离

## 概述

Claude Code 实现了多层安全隔离体系，从命令沙箱化、权限绕过防护、安全验证到路径校验，构建了一套纵深防御（Defense in Depth）架构。核心目标是：即使单一防线被突破，其他层仍能有效阻止不安全操作。整个安全模型围绕"最小权限"和"失败即拒绝"（fail-closed）原则设计。

## 安全验证流程总览

```mermaid
flowchart TD
    A["用户提交Bash命令"] --> B{"沙箱是否启用?<br/>SandboxManager.isSandboxingEnabled()"}
    B -- "否" --> C["直接执行"]
    B -- "是" --> D{"dangerouslyDisableSandbox<br/>且允许非沙箱命令?"}
    D -- "是" --> C
    D -- "否" --> E{"包含排除命令?<br/>containsExcludedCommand()"}
    E -- "是" --> C
    E -- "否" --> F["在沙箱中执行"]
    F --> G["bashSecurity.ts 验证"]
    G --> H{"检测到危险模式?"}
    H -- "是" --> I["拒绝或提示用户"]
    H -- "否" --> J["AST 解析验证<br/>tree-sitter-bash"]
    J --> K{"解析结果类型?"}
    K -- "simple" --> L["匹配权限规则"]
    K -- "too-complex" --> M["提示用户确认"]
    K -- "parse-unavailable" --> M
    L --> N{"匹配允许规则?"}
    N -- "是" --> O["自动执行"]
    N -- "否" --> M

    T["--dangerously-skip-permissions"] --> U{"Root权限?"}
    U -- "是且非沙箱" --> V["拒绝并退出"]
    U -- "否" --> W{"Ant用户?"}
    W -- "否" --> Y["允许"]
    W -- "是" --> X{"豁免入口点?"}
    X -- "local-agent或claude-desktop" --> Y
    X -- "其他" --> Z{"Docker或Bubblewrap或沙箱?"}
    Z -- "否" --> V
    Z -- "是" --> AA{"有互联网访问?"}
    AA -- "是" --> V
    AA -- "否" --> Y

    style V fill:"#f66",stroke:"#333",color:"#fff"
    style I fill:"#f96",stroke:"#333"
    style O fill:"#6f6",stroke:"#333"
    style Y fill:"#6f6",stroke:"#333"
```

## 一、沙箱判定：shouldUseSandbox.ts

### 核心逻辑

`shouldUseSandbox.ts` 位于 `src/tools/BashTool/`，是沙箱系统的入口关卡，负责判断 Bash 命令是否需要在沙箱环境中执行。

```typescript
type SandboxInput = {
  command?: string
  dangerouslyDisableSandbox?: boolean
}
```

### 判定流程

函数 `shouldUseSandbox(input)` 按以下优先级依次检查：

1. **沙箱是否启用**：调用 `SandboxManager.isSandboxingEnabled()`，如果系统不支持沙箱，直接返回 `false`
2. **显式禁用检查**：如果 `input.dangerouslyDisableSandbox` 为 `true` 且 `SandboxManager.areUnsandboxedCommandsAllowed()` 也为 `true`，则不沙箱化
3. **空命令检查**：无命令时返回 `false`
4. **排除命令检查**：调用 `containsExcludedCommand(input.command)`，匹配用户配置的排除命令时跳过沙箱

### 排除命令系统

`containsExcludedCommand()` 实现了两层排除机制：

**Ant 用户动态配置**：通过 GrowthBook 特性标志 `tengu_sandbox_disabled_commands` 获取禁用命令和子串列表。检查方式包括子串匹配和命令起始匹配。解析失败的畸形 Bash 语法被视为未排除，允许后续验证检查处理。

**用户自定义排除命令**：从 `settings.sandbox.excludedCommands` 读取用户配置。系统会将复合命令（如 `docker ps && curl evil.com`）拆分为子命令逐一检查，防止首个子命令匹配排除规则而后续命令逃逸。

关键安全设计：使用迭代式固定点算法（fixed-point）剥离环境变量前缀和安全包装器。对每个子命令生成候选列表，不断应用 `stripAllLeadingEnvVars` 和 `stripSafeWrappers` 直到不再产生新候选，处理 `timeout 300 FOO=bar bazel run` 这类交错模式——单次组合无法正确匹配的情况。每个候选与排除模式进行 `prefix`/`exact`/`wildcard` 三种规则匹配。

### 安全定位

源码注释明确指出：`excludedCommands` 是用户便利功能，**不是安全边界**。真正的安全控制是沙箱权限系统（会提示用户确认）。排除命令的绕过不是安全漏洞。

## 二、权限绕过防护：--dangerously-skip-permissions

### setup.ts 中的强制验证

`--dangerously-skip-permissions` 标志跳过所有权限提示，具有极高的安全风险。`setup.ts` 在函数 `setup()` 中对此标志进行严格的环境验证，确保其仅在安全环境中使用。

#### Root 用户检查

```typescript
if (process.platform !== 'win32' &&
    typeof process.getuid === 'function' &&
    process.getuid() === 0 &&
    process.env.IS_SANDBOX !== '1' &&
    !isEnvTruthy(process.env.CLAUDE_CODE_BUBBLEWRAP)) {
  console.error('--dangerously-skip-permissions cannot be used with root/sudo')
  process.exit(1)
}
```

以 root/sudo 身份运行时直接退出，除非处于沙箱环境（TPU 开发空间等需要 root 的沙箱场景）。

#### Ant 用户的额外验证

对于 Ant 用户类型（`USER_TYPE === 'ant'`），系统执行更严格的环境检查：

1. **豁免场景**：
   - `CLAUDE_CODE_ENTRYPOINT === 'local-agent'`：Desktop 的本地代理模式，信任模型与 CCR/BYOC 相同——Anthropic 管理的启动器有意识地预批准所有操作
   - `CLAUDE_CODE_ENTRYPOINT === 'claude-desktop'`：CCD（Claude Code in Desktop），Desktop 应用无条件传递此标志以解锁会话内绕过切换

2. **环境验证**（非豁免场景）：并行检查 Docker、互联网和 Bubblewrap 状态
   ```typescript
   const [isDocker, hasInternet] = await Promise.all([
     envDynamic.getIsDocker(),
     env.hasInternetAccess(),
   ])
   const isSandboxed = isDocker || isBubblewrap || isSandbox
   if (!isSandboxed || hasInternet) {
     process.exit(1)
   }
   ```

核心约束：**必须在 Docker 或沙箱容器中且无互联网访问**才允许使用。双条件同时满足——沙箱化环境 + 无网络——确保即使权限全部放开，也无法泄露数据或接触外部系统。

## 三、紧急关断：bypassPermissionsKillswitch

### 机制设计

`bypassPermissionsKillswitch.ts` 实现了远程紧急关断能力，允许在运行时通过 Statsig 特性门控动态禁用绕过权限模式，无需重启应用。

### 工作原理

1. `checkAndDisableBypassPermissionsIfNeeded()` 在首次查询前运行一次（`bypassPermissionsCheckRan` 标志保证只执行一次）
2. 调用 `shouldDisableBypassPermissions()` 查询 Statsig 特性门控
3. 如果门控返回 `true`，通过 `createDisabledBypassPermissionsContext()` 创建禁用上下文并更新 `AppState` 中的 `toolPermissionContext`
4. 如果 `toolPermissionContext.isBypassPermissionsModeAvailable` 已经为 `false`，则直接跳过检查

### 集成点

- **REPL.tsx**：组件挂载时通过 `useKickOffCheckAndDisableBypassPermissionsIfNeeded()` 触发检查
- **login.tsx**：`/login` 后调用 `resetBypassPermissionsCheck()` 重置 `bypassPermissionsCheckRan` 标志，使新组织的门控检查重新运行
- 远程模式下跳过检查（`getIsRemoteMode()` 返回 `true` 时）

### AutoMode 关断

类似的 `checkAndDisableAutoModeIfNeeded()` 针对 AutoMode 实现了相同的远程关断机制，通过 `verifyAutoModeGateAccess()` 验证。该检查不仅运行于组件挂载时，还监听 `mainLoopModel`、`mainLoopModelForSession` 和 `fastMode` 变化，在模型切换（`/model`、Cmd+P 选择器）或模式变更（`/fast`）时重新验证。AutoMode 的通知消息通过 `notifications.queue` 推送，颜色为 `warning`，优先级为 `high`。

## 四、命令安全验证：bashSecurity.ts

### 命令替换模式检测

`bashSecurity.ts` 位于 `src/tools/BashTool/`，定义了 `COMMAND_SUBSTITUTION_PATTERNS`，检测所有已知的命令替换和注入向量：

| 模式 | 说明 |
|------|------|
| `<(` | 进程替换（输入） |
| `>(` | 进程替换（输出） |
| `=(` | Zsh 进程替换 |
| `=(cmd)` | Zsh EQUALS 展开，`=curl evil.com` 绕过 Bash(curl:*) 规则 |
| `$(` | 命令替换 |
| `${` | 参数替换 |
| `$[` | 旧式算术展开 |
| `~[` | Zsh 参数展开 |
| `(e:` | Zsh glob 限定符 |
| `(+` | Zsh glob 限定符（命令执行） |
| `} always {` | Zsh always 块（try/always 构造） |
| `<#` | PowerShell 注释语法（纵深防御） |

### Zsh 危险命令

`ZSH_DANGEROUS_COMMANDS` 集合包含所有已知可绕过安全检查的 Zsh 内建命令，按攻击向量分类：

- **zmodload**：模块加载网关，可加载 `zsh/mapfile`（隐形文件 I/O，通过数组赋值）、`zsh/system`（sysopen/syswrite 两步文件访问）、`zsh/zpty`（伪终端命令执行）、`zsh/net/tcp`（TCP 外泄）、`zsh/files`（绕过二进制安全检查的内建 rm/mv/ln/chmod）
- **emulate**：`-c` 标志等效于 eval，执行任意代码
- **zsh/system 内建**：`sysopen`（细粒度文件打开）、`sysread`（文件描述符读取）、`syswrite`（文件描述符写入）、`sysseek`（文件描述符定位）
- **zsh/zpty**：伪终端命令执行
- **zsh/net**：`ztcp`（TCP 连接）、`zsocket`（Unix/TCP 套接字）
- **zsh/files 内建**：`zf_rm`、`zf_mv`、`zf_ln`、`zf_chmod`、`zf_chown`、`zf_mkdir`、`zf_rmdir`、`zf_chgrp`——这些内建命令绕过二进制安全检查，因为它们不调用外部程序

### Here Document 分析

通过 `extractHeredocs()` 函数解析 heredoc 结构。`HEREDOC_IN_SUBSTITUTION` 模式（`$\(.*<<`）检测命令替换中的 heredoc——一种常见的注入手法。安全 heredoc 需满足分隔符单引号引用、关闭分隔符独占一行、不含嵌套匹配等条件。

### 安全检查编号系统

`BASH_SECURITY_CHECK_IDS` 为每种检查类型分配了数字 ID，避免在日志中泄露敏感字符串。共 23 个检查类别：

1. 不完整命令  2. jq 系统函数  3. jq 文件参数  4. 混淆标志  5. Shell 元字符  6. 危险变量  7. 换行符  8. 命令替换  9. 输入重定向  10. 输出重定向  11. IFS 注入  12. Git commit 替换  13. /proc/environ 访问  14. 畸形令牌注入  15. 反斜杠转义空白  16. 花括号展开  17. 控制字符  18. Unicode 空白  19. 词中 Hash  20. Zsh 危险命令  21. 反斜杠转义运算符  22. 注释引号失同步  23. 引用换行

### AST 解析验证

`src/utils/bash/ast.ts` 使用 tree-sitter-bash 解析器替代传统的 shell-quote + 手写字符遍历方法。关键设计属性是**失败即关闭**（fail-closed）：不在显式白名单中的节点类型导致整个命令被分类为 `too-complex`，强制走权限提示流程。解析结果有三种类型：`simple`（可提取可信 argv）、`too-complex`（需用户确认）、`parse-unavailable`（解析器不可用）。

## 五、危险模式列表：dangerousPatterns.ts

### 跨平台代码执行入口

`CROSS_PLATFORM_CODE_EXEC` 列出了 Unix 和 Windows 上均存在的代码执行入口点：

- **解释器**：python、python3、python2、node、deno、tsx、ruby、perl、php、lua
- **包运行器**：npx、bunx、npm run、yarn run、pnpm run、bun run
- **Shell**：bash、sh
- **远程命令**：ssh（Git Bash/WSL 在 Windows，原生在 Unix）

### Bash 危险模式

`DANGEROUS_BASH_PATTERNS` 在跨平台列表基础上增加了 Unix 专有条目：zsh、fish、eval、exec、env、xargs、sudo。

**Ant 专用扩展**：基于沙箱使用数据的风险评估，增加 `fa run`（集群代码启动器）、`coo`、`gh`/`gh api`（GitHub API 任意 HTTP）、`curl`/`wget`（网络外泄）、`git`（hooks 安装等于任意代码执行）、`kubectl`/`aws`/`gcloud`/`gsutil`（云资源写入）。这些模式在外部构建中被 DCE 消除。

### 用途

这些模式列表被 `permissionSetup.ts` 中的 `isDangerousBashPermission`/`isDangerousPowerShellPermission` 谓词使用，在 Auto 模式入口时剥离此类规则。例如 `Bash(python:*)` 或 `PowerShell(node:*)` 允许规则会让模型通过解释器运行任意代码，绕过自动模式分类器。匹配器处理各种规则形状变体（exact、`:*`、尾随 `*`、` *`、` -...*`）。

## 六、安全模型：工作区信任与受管钩子

### 工作区信任（Workspace Trust）

所有钩子执行都要求工作区信任。在交互模式下，`src/utils/hooks.ts` 的安全检查逻辑覆盖所有钩子类型：

- **PreToolUse / PostToolUse 钩子**：未信任时跳过执行并记录调试日志
- **StatusLine 命令**：跳过执行
- **FileSuggestion 命令**：跳过执行
- **SessionStart 钩子**：跳过执行

MCP 服务器头信息辅助器在信任确认前执行时会记录安全警告，提醒开发者注意异常调用时序。

### 受管钩子模式（Managed Hooks Only）

`hooksSettings.ts` 和 `hooksConfigSnapshot.ts` 实现了 `allowManagedHooksOnly` 策略：

- 当 `policySettings.allowManagedHooksOnly === true` 时，`getAllHooks()` 只返回受管钩子
- 用户/项目/本地设置的钩子在 UI 中被隐藏和阻止执行
- 受管钩子本身也故意不在 UI 中显示（防止信息泄露）

这防止了仓库中的 `.claude/settings.json` 定义恶意钩子（如 `pre-commit` 执行任意命令），因为项目来源的钩子被完全排除。`hooksConfigSnapshot.ts` 进一步确保：如果 `disableAllHooks` 在非受管设置中设置，仍然返回受管钩子——非受管设置不能禁用受管钩子。

### 项目设置排除策略

安全关键配置故意排除项目级设置来源：

- `autoMemoryDirectory`：只从 `policySettings`、`flagSettings`、`localSettings`、`userSettings` 读取，**不读 `projectSettings`**——防止恶意仓库设置 `autoMemoryDirectory: "~/.ssh"` 获得静默写入敏感目录的能力
- `allowManagedHooksOnly`：同样只从 `policySettings` 读取
- 钩子配置：项目钩子在受管模式下被排除

## 七、路径校验：validateMemoryPath

### 实现位置

`src/memdir/paths.ts` 中的 `validateMemoryPath(raw, expandTilde)` 函数。

### 校验规则

函数对所有候选自动记忆目录路径执行以下安全检查，任一不通过即返回 `undefined`：

| 检查项 | 原因 | 示例 |
|--------|------|------|
| 非绝对路径 | 相对路径基于 CWD 解释，不可控 | `../foo` |
| 路径长度 < 3 | 根路径/近根路径过于宽泛 | `/` 变为空, `/a` 太短 |
| Windows 驱动器根 | 会匹配整个驱动器 | `C:\` 变为 `C:` |
| UNC 路径 | 网络路径是不透明信任边界 | `\\server\share` |
| 空字节 | `normalize()` 无法消除，可截断系统调用 | `foo\0bar` |

### Tilde 展开安全

- `~/` 和 `~\` 前缀的路径支持展开，但裸 `~`、`~/`、`~/.`、`~/..` 被拒绝——这些会展开到 `$HOME` 或其父目录
- `normalize()` 后的剩余部分如果是 `.` 或 `..` 也被拒绝
- 环境变量覆盖（`CLAUDE_COWORK_MEMORY_PATH_OVERRIDE`）不支持 tilde 展开——由 Cowork/SDK 以编程方式设置，应始终传递绝对路径

### 路径标准化

验证通过后返回的路径：去除尾部分隔符后添加恰好一个 `sep`，并执行 NFC Unicode 标准化，确保路径比较的一致性。

### 记忆路径安全写入豁免

`isAutoMemPath()` 检查路径是否在自动记忆目录内。当 `CLAUDE_COWORK_MEMORY_PATH_OVERRIDE` 设置时，该函数匹配覆盖目录，但**不获得文件系统写入豁免**——只有用户通过 settings.json 选择的路径才获得写入豁免，因为那是用户从可信来源的明确选择。

## 八、打包技能文件安全提取

### 安全写入机制

`bundledSkills.ts` 在首次调用时将技能引用文件提取到磁盘。提取过程采用多层安全机制：

1. **O_NOFOLLOW**：不跟随符号链接，防止符号链接攻击
2. **O_EXCL**：排他创建，文件已存在则失败，防止覆盖攻击
3. **严格文件模式**：目录 `0o700`、文件 `0o600`
4. **路径遍历检测**：拒绝包含 `..` 的路径和绝对路径
5. **不 unlink+重试**：遇到 EEXIST 时不删除重试，因为 `unlink()` 会跟随中间符号链接

提取是闭包级记忆化（per-process 单次），并发调用共享同一个 Promise 避免竞态条件。

## 九、网络安全指令：cyberRiskInstruction.ts

### 内容

`CYBER_RISK_INSTRUCTION` 定义了 Claude 在处理安全相关请求时的行为边界：

> 协助授权的安全测试、防御性安全、CTF 挑战和教育场景。拒绝破坏性技术、DoS 攻击、大规模目标攻击、供应链妥协或用于恶意目的的检测规避。双用途安全工具（C2 框架、凭据测试、漏洞利用开发）需要明确的授权上下文：渗透测试项目、CTF 竞赛、安全研究或防御性用例。

### 所有权与修改规则

- 该指令由 Safeguards 团队拥有（David Forsythe, Kyla Guru）
- 修改前必须联系 Safeguards 团队
- 必须通过正式评估并获得明确批准
- Claude 不得自行编辑此文件

## 十、纵深防御架构总结

Claude Code 的安全体系遵循纵深防御原则，从外到内形成多层防线：

1. **环境层**：沙箱容器 + 无网络约束（`--dangerously-skip-permissions` 强制）
2. **会话层**：工作区信任确认 + 远程关断开关（bypassPermissionsKillswitch）
3. **配置层**：项目设置排除 + 受管钩子模式（allowManagedHooksOnly）
4. **命令层**：bashSecurity 模式检测 + AST 解析验证（tree-sitter fail-closed）
5. **权限层**：危险模式列表 + 自动模式剥离（dangerousPatterns）
6. **路径层**：validateMemoryPath 全方位校验（相对/根/UNC/null 字节）
7. **文件层**：O_NOFOLLOW|O_EXCL 安全写入 + 路径遍历检测
8. **行为层**：cyberRiskInstruction 约束模型输出

每一层都独立工作，即使其他层被突破，仍能提供有效防护。这种架构确保了单个安全漏洞不会导致系统完全失守。

## 关键源文件

| 文件 | 职责 |
|------|------|
| `src/tools/BashTool/shouldUseSandbox.ts` | 沙箱启用决策 |
| `src/tools/BashTool/bashSecurity.ts` | Bash 命令安全验证流水线 |
| `src/utils/bash/ast.ts` | AST 解析安全验证（tree-sitter） |
| `src/utils/permissions/dangerousPatterns.ts` | 危险权限模式定义 |
| `src/constants/cyberRiskInstruction.ts` | 网络安全行为指令 |
| `src/memdir/paths.ts` | 内存路径验证与安全 |
| `src/skills/bundledSkills.ts` | 技能文件安全提取 |
| `src/setup.ts` | --dangerously-skip-permissions 验证 |
| `src/utils/permissions/bypassPermissionsKillswitch.ts` | 权限绕过紧急关闭 |
| `src/utils/hooks/hooksSettings.ts` | Hook 托管模式与工作区信任 |
| `src/utils/hooks/hooksConfigSnapshot.ts` | Hook 配置快照与受管钩子过滤 |
