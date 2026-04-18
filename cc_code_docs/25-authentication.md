# 认证与密钥管理

## 概述

Claude Code的认证与密钥管理系统位于`src/utils/auth.ts`（约2000+行），实现了复杂的多源认证解析、OAuth令牌管理、API密钥辅助器缓存和跨平台密钥链集成。该模块是Claude Code安全架构的核心，负责确定使用何种认证方式（1P OAuth、3P API密钥、apiKeyHelper等），并以安全优先的方式管理密钥的生命周期。

## isAnthropicAuthEnabled() - 认证启用判定

```typescript
function isAnthropicAuthEnabled(): boolean
```

判定是否激活1P（第一方）Anthropic认证。返回false的情况：

1. **Bare模式**（`--bare`）：仅API密钥模式，永不使用OAuth
2. **3P服务**：使用Bedrock/Vertex/Foundry时，认证由云服务商处理
3. **外部API密钥**：
   - 存在`ANTHROPIC_AUTH_TOKEN`环境变量（非托管上下文）
   - 存在`apiKeyHelper`配置（非托管上下文）
   - `ANTHROPIC_API_KEY`来自外部源（非托管上下文）

**托管上下文例外**：CCR和Claude Desktop生成的会话（`CLAUDE_CODE_REMOTE`或`claude-desktop`入口点）绕过用户设置，即使配置了外部API密钥也使用OAuth。这防止了用户的终端CLI设置影响托管会话。

**SSH隧道特殊处理**：当`ANTHROPIC_UNIX_SOCKET`设置时，只有同时存在`CLAUDE_CODE_OAUTH_TOKEN`才启用1P认证。远程端的`~/.claude`设置绝不能翻转此判断，否则会导致与本地代理的头部不匹配。

## getAuthTokenSource() - 令牌来源解析

```typescript
function getAuthTokenSource(): { source: string, hasToken: boolean }
```

按优先级确定认证令牌来源：

| 优先级 | 来源 | 条件 |
|---|---|---|
| 1 | `apiKeyHelper` | bare模式下仅检查--settings源 |
| 2 | `ANTHROPIC_AUTH_TOKEN` | 非托管上下文 |
| 3 | `CLAUDE_CODE_OAUTH_TOKEN` | 环境变量存在 |
| 4 | `CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR` | 文件描述符OAuth |
| 5 | `CCR_OAUTH_TOKEN_FILE` | CCR磁盘回退 |
| 6 | `apiKeyHelper` | 非托管上下文 |
| 7 | `claude.ai` | OAuth令牌具有正确scope |
| 8 | `none` | 无可用令牌 |

注意优先级1和6都是apiKeyHelper，但上下文不同：bare模式仅允许--settings标志提供的apiKeyHelper，而完整模式还允许用户设置和项目设置。

## getAnthropicApiKeyWithSource() - API密钥解析

```typescript
function getAnthropicApiKeyWithSource(
  opts?: { skipRetrievingKeyFromApiKeyHelper?: boolean }
): { key: string | null, source: ApiKeySource }
```

返回`{key, source}`元组，source类型为`ApiKeySource`：

### 解析优先级

1. **Bare模式**：
   - `ANTHROPIC_API_KEY`环境变量
   - `apiKeyHelper`（仅来自--settings标志）

2. **Homespace**：跳过`ANTHROPIC_API_KEY`，使用Console密钥

3. **CI模式**（`CI=true`或`NODE_ENV=test`）：
   - 文件描述符API密钥（`CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR`）
   - `ANTHROPIC_API_KEY`（若存在）
   - 若两者都无且无OAuth令牌，抛出错误

4. **已批准的ANTHROPIC_API_KEY**：
   - 用户通过信任对话框批准的API密钥
   - 存储在`customApiKeyResponses.approved`列表中
   - 通过`normalizeApiKeyForConfig()`规范化后匹配

5. **文件描述符API密钥**：`getApiKeyFromFileDescriptor()`

6. **apiKeyHelper**：
   - 使用同步缓存（`getApiKeyFromApiKeyHelperCached()`）
   - 缓存为冷时返回null和source='apiKeyHelper'（不阻塞）
   - 需要真实密钥的调用者必须先await `getApiKeyFromApiKeyHelper()`

7. **配置文件或macOS密钥链**：`getApiKeyFromConfigOrMacOSKeychain()`

8. **无密钥**：`{ key: null, source: 'none' }`

### 认证来源解析顺序总结

```
ANTHROPIC_API_KEY -> 文件描述符 -> apiKeyHelper -> /login托管密钥
```

在非bare模式下，`ANTHROPIC_API_KEY`需要用户批准才被接受（安全措施），防止无意中使用环境变量中的密钥。

## getApiKeyFromApiKeyHelper() - 异步密钥辅助器

```typescript
async function getApiKeyFromApiKeyHelper(
  isNonInteractiveSession: boolean
): Promise<string | null>
```

实现了stale-while-revalidate（SWR）缓存模式，平衡响应速度和密钥新鲜度。

### 缓存策略

**热缓存（缓存存在且未过期）**：
- 直接返回缓存值，不执行辅助器
- TTL默认5分钟（DEFAULT_API_KEY_HELPER_TTL = 300,000ms）
- 可通过`CLAUDE_CODE_API_KEY_HELPER_TTL_MS`环境变量自定义

**过期缓存（缓存存在但TTL已过）**：
- 立即返回过期值（不阻塞）
- 在后台启动刷新（`_runAndCache`）
- `startedAt`设为null（标记为后台刷新）

**冷缓存（缓存不存在）**：
- 阻塞等待辅助器执行完成
- `startedAt`设为当前时间（用于超时监控）
- 去重并发调用（同一时刻只有一个inflight请求）

### Epoch-based失效

```typescript
let _apiKeyHelperEpoch = 0

function clearApiKeyHelperCache(): void {
  _apiKeyHelperEpoch++
  _apiKeyHelperCache = null
  _apiKeyHelperInflight = null
}
```

每次缓存清除时递增epoch。正在执行的辅助器在完成时检查captured epoch是否匹配当前epoch，不匹配则不写入缓存，防止过期的异步操作覆盖新状态。

### 执行流程

```typescript
async function _runAndCache(isCold, epoch): Promise<string | null> {
  try {
    const value = await _executeApiKeyHelper(isNonInteractiveSession)
    if (epoch !== _apiKeyHelperEpoch) return value  // epoch不匹配，不写缓存
    if (value !== null) {
      _apiKeyHelperCache = { value, timestamp: Date.now() }
    }
    return value
  } catch (e) {
    if (epoch !== _apiKeyHelperEpoch) return ' '  // epoch不匹配
    // SWR路径：瞬态失败不应替换工作密钥
    if (!isCold && _apiKeyHelperCache && _apiKeyHelperCache.value !== ' ') {
      _apiKeyHelperCache = { ..._apiKeyHelperCache, timestamp: Date.now() }
      return _apiKeyHelperCache.value
    }
    // 冷缓存：缓存' '（空格哨兵），防止调用者回退到OAuth
    _apiKeyHelperCache = { value: ' ', timestamp: Date.now() }
    return ' '
  } finally {
    if (epoch === _apiKeyHelperEpoch) {
      _apiKeyHelperInflight = null
    }
  }
}
```

**哨兵值' '**：当apiKeyHelper执行失败且无先前缓存时，存储空格字符作为哨兵值，防止调用者因key=null而回退到OAuth认证路径。这是安全优先的设计——配置了apiKeyHelper的用户明确不希望使用OAuth。

### 安全检查

```typescript
async function _executeApiKeyHelper(isNonInteractiveSession): Promise<string | null> {
  // 来自项目/本地设置的apiKeyHelper需要信任确认
  if (isApiKeyHelperFromProjectOrLocalSettings()) {
    const hasTrust = checkHasTrustDialogAccepted()
    if (!hasTrust && !isNonInteractiveSession) {
      // 安全：apiKeyHelper在工作区信任确认前执行
      logEvent('tengu_apiKeyHelper_missing_trust11', {})
      return null
    }
  }
  // 执行辅助器命令
  const result = await execa(apiKeyHelper, {
    shell: true,
    timeout: 10 * 60 * 1000,  // 10分钟超时
    reject: false
  })
  // ...
}
```

项目设置中的apiKeyHelper在工作区信任确认前不会被自动执行，防止恶意仓库在用户不知情的情况下执行任意命令。

## 托管与非托管上下文

```typescript
function isManagedOAuthContext(): boolean {
  return (
    isEnvTruthy(process.env.CLAUDE_CODE_REMOTE) ||
    process.env.CLAUDE_CODE_ENTRYPOINT === 'claude-desktop'
  )
}
```

**托管上下文**（CCR、Claude Desktop）：
- 绕过用户设置中的apiKeyHelper、ANTHROPIC_API_KEY、ANTHROPIC_AUTH_TOKEN
- 始终使用OAuth令牌
- 防止用户的终端CLI设置影响托管会话

**非托管上下文**（终端CLI）：
- 尊重用户配置
- 遵循标准优先级解析

## 跨平台密钥链

### macOS Keychain

在macOS上，API密钥存储在系统Keychain中：

- 服务名：`getMacOsKeychainStorageServiceName()`返回的服务标识
- 账户名：`getUsername()`获取当前用户名
- 通过`getSecureStorage()`抽象访问

### 配置文件回退

在非macOS平台上，API密钥存储在配置文件中：

- 路径：`~/.claude/`目录下
- `getApiKeyFromConfigOrMacOSKeychain()`自动选择适当的存储

### 密钥链缓存

- `clearKeychainCache()`：清除密钥链缓存
- `getMacOsKeychainStorageServiceName()`：获取macOS密钥链服务名
- `getUsername()`：获取当前系统用户名

### 预取优化

```typescript
function prefetchApiKeyFromApiKeyHelperIfSafe(isNonInteractiveSession): void {
  // 项目设置的apiKeyHelper需要信任确认，未确认时跳过
  if (isApiKeyHelperFromProjectOrLocalSettings() &&
      !checkHasTrustDialogAccepted()) {
    return
  }
  void getApiKeyFromApiKeyHelper(isNonInteractiveSession)
}
```

在安全条件满足时预取apiKeyHelper的值，减少首次API调用的延迟。异步执行，不阻塞启动。

## Bare模式隔离

Bare模式（`--bare`标志）实现了完全的认证隔离：

- **仅允许的来源**：`ANTHROPIC_API_KEY`环境变量和`apiKeyHelper`（仅来自--settings标志）
- **永不使用**：OAuth、密钥链、配置文件API密钥、用户设置
- **3P服务**：使用云服务商凭据，不经过此路径
- **设计意图**：CI/CD和自动化场景的简洁认证路径

## OAuth令牌管理

### 令牌检查与刷新

```typescript
function checkAndRefreshOAuthTokenIfNeeded(): Promise<void>
```

- 检查OAuth令牌是否即将过期
- 主动刷新过期的令牌
- 清除自己的缓存（在所有接触密钥链的路径中）
- 避免在有效令牌路径上强制阻塞式密钥链生成

### 令牌过期处理

```typescript
function handleOAuth401Error(staleAccessToken: string): Promise<boolean>
```

- 401错误时尝试刷新OAuth令牌
- 成功返回true，失败返回false
- 桥接系统通过此函数实现withOAuthRetry模式

### OAuth来源

`src/services/oauth/`目录实现了OAuth流程：

- **client.ts**：OAuth客户端，包括令牌获取、刷新和scope检查
- **getOauthProfile.ts**：获取OAuth配置
- **auth-code-listener.ts**：认证码监听器（本地回调服务器）
- **crypto.ts**：PKCE加密工具

### 文件描述符OAuth

`getOAuthTokenFromFileDescriptor()`从文件描述符获取OAuth令牌：

- CCR子进程无法继承管道FD时，回退到磁盘文件
- 通过`CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR`环境变量指定
- 来源区分：FD来源 vs CCR磁盘回退

## AWS认证支持

auth.ts还支持AWS认证模式：

- **awsAuthRefresh**：AWS SSO登录刷新命令
- **awsCredentialExport**：AWS凭据导出
- **STS验证**：`checkStsCallerIdentity()`验证当前AWS身份
- **项目设置安全检查**：来自项目设置的awsAuthRefresh需要工作区信任确认
- **状态管理**：`AwsAuthStatusManager`跟踪认证流程的状态和输出
- **超时**：3分钟（AWS_AUTH_REFRESH_TIMEOUT_MS），覆盖浏览器SSO流程

## 认证解析流程图

```mermaid
flowchart TD
    A["认证请求"] --> B{"isBareMode()?"}

    B -->|是| C["仅检查ANTHROPIC_API_KEY和apiKeyHelper"]
    C --> D{"ANTHROPIC_API_KEY?"}
    D -->|是| E["返回 ANTHROPIC_API_KEY"]
    D -->|否| F{"apiKeyHelper --settings?"}
    F -->|是| G["getApiKeyFromApiKeyHelperCached()"]
    F -->|否| H["返回 none"]

    B -->|否| I{"isManagedOAuthContext()?"}

    I -->|是| J["跳过用户设置 使用OAuth"]
    I -->|否| K["标准解析路径"]

    K --> L{"ANTHROPIC_AUTH_TOKEN?"}
    L -->|是 且非托管| M["返回 ANTHROPIC_AUTH_TOKEN"]
    L -->|否| N{"CLAUDE_CODE_OAUTH_TOKEN?"}
    N -->|是| O["返回 CLAUDE_CODE_OAUTH_TOKEN"]
    N -->|否| P{"FD OAuth令牌?"}
    P -->|是| Q["返回 FD_OAUTH_TOKEN"]
    P -->|否| R{"apiKeyHelper?"}

    R -->|是| S["getApiKeyFromApiKeyHelperCached()"]
    S --> T{"缓存热?"}
    T -->|是| U["返回缓存值"]
    T -->|否| V{"缓存冷?"}
    V -->|是| W["阻塞等待辅助器执行"]
    V -->|否| X["返回过期值 后台刷新"]

    R -->|否| Y{"claude.ai OAuth?"}
    Y -->|是| Z["返回 claude.ai"]
    Y -->|否| AA["返回 none"]

    W --> BB["_executeApiKeyHelper()"]
    BB --> CC{"项目设置apiKeyHelper?"}
    CC -->|是| DD{"信任已确认?"}
    DD -->|否| EE["返回null 安全阻止"]
    DD -->|是| FF["执行辅助器命令"]
    CC -->|否| FF
    FF --> GG{"执行成功?"}
    GG -->|是| HH["缓存值 返回"]
    GG -->|否| II{"SWR: 有旧缓存?"}
    II -->|是| JJ["返回旧值 续期时间戳"]
    II -->|否| KK["缓存' '哨兵 阻止OAuth回退"]

    subgraph "密钥链存储"
        LL["macOS Keychain"]
        MM["配置文件回退"]
    end

    LL --> NN["getApiKeyFromConfigOrMacOSKeychain()"]
    MM --> NN
