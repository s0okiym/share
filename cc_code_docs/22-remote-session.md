# 远程会话管理

## 概述

远程会话管理模块位于`src/remote/`目录，负责管理与CCR（Claude Code Remote）后端的远程会话连接。该模块协调WebSocket订阅接收消息、HTTP POST发送消息、权限请求/响应流程，以及会话的连接、断开和重连。核心设计采用观察者/参与者（viewer/actor）分离模式，支持纯查看模式和完整交互模式。

## RemoteSessionManager - 远程会话管理器

`src/remote/RemoteSessionManager.ts`是远程会话的核心协调器，管理WebSocket订阅、HTTP POST消息发送和权限请求流程。

### 配置类型

```typescript
type RemoteSessionConfig = {
  sessionId: string
  getAccessToken: () => string
  orgUuid: string
  hasInitialPrompt?: boolean
  viewerOnly?: boolean
}
```

- `sessionId`：CCR会话ID
- `getAccessToken`：获取访问令牌的函数
- `orgUuid`：组织UUID
- `hasInitialPrompt`：会话是否已有初始提示正在处理
- `viewerOnly`：纯查看模式标志

### viewerOnly模式

当`viewerOnly`为true时，远程会话进入纯查看模式，具有以下特征：

- **不发送中断**：Ctrl+C/Escape不会向远程Agent发送中断信号
- **禁用60秒重连超时**：不会因超时放弃重连
- **不更新会话标题**：永远不会修改远程会话的标题
- 用于`claude assistant`场景，用户仅需观察远程Agent的执行过程

### 回调接口

```typescript
type RemoteSessionCallbacks = {
  onMessage: (message: SDKMessage) => void
  onPermissionRequest: (request: SDKControlPermissionRequest, requestId: string) => void
  onPermissionCancelled?: (requestId: string, toolUseId: string | undefined) => void
  onConnected?: () => void
  onDisconnected?: () => void
  onReconnecting?: () => void
  onError?: (error: Error) => void
}
```

### 核心方法

#### connect() - 建立连接

创建SessionsWebSocket实例并连接。将WebSocket的各类事件映射到回调：

- `onMessage` -> `handleMessage()` -> 根据消息类型分发
- `onConnected` -> 回调通知连接成功
- `onClose` -> 回调通知断开连接
- `onReconnecting` -> 回调通知正在重连
- `onError` -> 回调通知错误

#### handleMessage() - 消息处理

处理从WebSocket接收的消息，根据类型分发：

1. **control_request**：权限请求，存储到`pendingPermissionRequests` Map中，触发`onPermissionRequest`回调
2. **control_cancel_request**：服务端取消挂起的权限提示，从Map中删除，触发`onPermissionCancelled`回调
3. **control_response**：确认响应，仅记录日志
4. **SDKMessage**：转发到`onMessage`回调（使用`isSDKMessage()`类型守卫过滤）

#### sendMessage() - 发送消息

通过HTTP POST向远程会话发送用户消息：

- 调用`sendEventToRemoteSession()`发送
- 支持自定义UUID（opts.uuid）
- 返回boolean表示发送是否成功

#### respondToPermissionRequest() - 响应权限请求

处理权限请求的响应流程：

1. 从`pendingPermissionRequests` Map中查找对应的请求
2. 删除Map中的条目
3. 构造SDKControlResponse：
   - `behavior: 'allow'` + `updatedInput`
   - `behavior: 'deny'` + `message`
4. 通过WebSocket发送control_response

#### cancelSession() - 取消会话

发送中断信号（`{ subtype: 'interrupt' }`）到远程会话，取消当前请求。注意在viewerOnly模式下不应调用此方法。

#### disconnect() - 断开连接

关闭WebSocket连接，清空`pendingPermissionRequests` Map。

#### reconnect() - 强制重连

调用WebSocket的`reconnect()`方法，在容器关闭后订阅变得陈旧时使用。

### 权限请求管理

`pendingPermissionRequests`是一个Map<string, SDKControlPermissionRequest>，存储所有待处理的权限请求：

- 键为request_id
- 值为SDKControlPermissionRequest（包含tool_name、tool_use_id、input等）
- 当权限请求被响应或被服务端取消时，从Map中删除
- 断开连接时清空整个Map

### 权限响应构建

权限响应遵循SDK协议格式：

```typescript
{
  type: 'control_response',
  response: {
    subtype: 'success',
    request_id: string,
    response: {
      behavior: 'allow' | 'deny',
      // allow时包含 updatedInput
      // deny时包含 message
    }
  }
}
```

对于不识别的control_request子类型（非`can_use_tool`），返回错误响应：

```typescript
{
  type: 'control_response',
  response: {
    subtype: 'error',
    request_id: string,
    error: string
  }
}
```

## SessionsWebSocket - WebSocket客户端

`src/remote/SessionsWebSocket.ts`实现了连接CCR会话的WebSocket客户端，支持Bun原生WebSocket和Node.js的ws包。

### 连接协议

1. 连接到`wss://api.anthropic.com/v1/sessions/ws/{sessionId}/subscribe?organization_uuid=...`
2. 通过Authorization头发送OAuth令牌认证
3. 接收SDKMessage流

### 双运行时WebSocket抽象

SessionsWebSocket根据运行时环境选择WebSocket实现：

**Bun运行时**：
- 使用`globalThis.WebSocket`
- 支持headers、proxy、tls选项
- 通过addEventListener绑定事件

**Node.js运行时**：
- 动态导入ws包（`import('ws')`）
- 支持headers、agent（代理）、TLS选项
- 通过.on()方法绑定事件

两种实现共享`WebSocketLike`接口：`close()`、`send(data)`、`ping?()`。

### 认证方式

通过HTTP头进行认证，无需发送单独的auth消息：

```typescript
headers = {
  Authorization: `Bearer ${accessToken}`,
  'anthropic-version': '2023-06-01'
}
```

每次连接尝试获取最新令牌（`this.getAccessToken()`），确保使用有效的认证凭据。

### Ping/Pong保活

- 间隔：30秒（PING_INTERVAL_MS）
- 连接成功后启动定时ping
- 收到pong时记录日志
- 断开连接时停止ping定时器
- ping错误被静默忽略（close处理器会处理连接问题）

### 指数退避重连

当WebSocket意外关闭时：

- **基础延迟**：2000毫秒（RECONNECT_DELAY_MS）
- **最大尝试次数**：5次（MAX_RECONNECT_ATTEMPTS）
- 仅在先前状态为connected时尝试重连
- 每次重连递增reconnectAttempts计数器
- 超过最大尝试次数后触发`onClose`回调

### 特殊关闭代码处理

**4001 - 会话未找到**：
- 在压缩期间可能是瞬态的（服务端短暂认为会话陈旧）
- 最多重试3次（MAX_SESSION_NOT_FOUND_RETRIES）
- 延迟递增：2000ms * retryCount
- 超过重试预算后触发`onClose`

**4003 - 未授权**：
- 永久关闭代码，立即停止重连
- 触发`onClose`回调

**其他关闭代码**：
- 尝试标准重连流程（最多5次）

### 消息解析与分发

接收到的消息通过`isSessionsMessage()`类型守卫验证：

```typescript
function isSessionsMessage(value: unknown): value is SessionsMessage {
  // 接受任何具有字符串type字段的消息
  // 下游处理器决定如何处理未知类型
  // 硬编码白名单会静默丢弃后端新增的消息类型
}
```

消息类型包括：SDKMessage | SDKControlRequest | SDKControlResponse | SDKControlCancelRequest

### 强制重连

`reconnect()`方法重置所有重连计数器，关闭现有连接，500ms延迟后建立新连接。用于容器关闭后订阅变得陈旧的场景。

## sdkMessageAdapter - 消息适配器

`src/remote/sdkMessageAdapter.ts`负责将CCR后端发送的SDK格式消息转换为REPL内部Message类型。

### 转换规则

| SDK消息类型 | REPL消息类型 | 说明 |
|---|---|---|
| SDKAssistantMessage | AssistantMessage | 直接转换message、uuid、error |
| SDKPartialAssistantMessage | StreamEvent | 流式事件，转换event字段 |
| SDKResultMessage | SystemMessage | 仅错误结果转为warning级别 |
| SDKSystemMessage(init) | SystemMessage | 显示模型信息 |
| SDKStatusMessage | SystemMessage | compacting等状态 |
| SDKToolProgressMessage | SystemMessage | 工具运行进度 |
| SDKCompactBoundaryMessage | SystemMessage | 压缩边界标记 |
| SDKUserMessage | 取决于选项 | 可转换tool_result和用户文本 |

### 转换选项

```typescript
type ConvertOptions = {
  convertToolResults?: boolean    // 转换包含tool_result的用户消息
  convertUserTextMessages?: boolean  // 转换用户文本消息（历史事件）
}
```

- **CCR模式**：默认忽略所有用户消息（由REPL本地处理）
- **直连模式**：启用convertToolResults，渲染远程工具结果
- **历史事件转换**：启用convertUserTextMessages，渲染用户输入的消息

### 忽略的消息类型

以下消息类型在适配器中被忽略：

- **auth_status**：由单独的逻辑处理
- **tool_use_summary**：SDK-only事件
- **rate_limit_event**：SDK-only事件
- **未知类型**：优雅忽略，记录日志但不崩溃

### 辅助函数

- `isSessionEndMessage(msg)`：判断消息是否为会话结束（type === 'result'）
- `isSuccessResult(msg)`：判断结果消息是否成功（subtype === 'success'）
- `getResultText(msg)`：提取成功结果的文本

## remotePermissionBridge - 远程权限桥接

`src/remote/remotePermissionBridge.ts`为远程权限请求创建合成消息，使本地UI能够渲染它们。

### 合成AssistantMessage

当权限请求到达时，本地没有真实的AssistantMessage（工具使用在CCR容器上运行）。`createSyntheticAssistantMessage()`构造一个合成的AssistantMessage：

- 包含tool_use内容块（id、name、input来自权限请求）
- message.id格式为`remote-{requestId}`
- 模型设为空字符串
- token使用量全部为0

### 工具存根

当远程CCR具有本地CLI不知道的工具（如MCP工具）时，`createToolStub()`创建最小化的Tool存根：

- `name`：工具名称
- `isEnabled()`：始终返回true
- `userFacingName()`：返回工具名称
- `renderToolUseMessage()`：显示前3个输入条目
- `isReadOnly()`：返回false
- `needsPermissions()`：返回true
- 路由到FallbackPermissionRequest进行权限处理

## 观察者/参与者分离设计

远程会话管理采用viewer/actor分离设计：

### Viewer（观察者）

- 仅接收和显示消息
- 不发送中断信号
- 不响应权限请求
- 不更新会话标题
- 用于assistant模式（`claude assistant`）

### Actor（参与者）

- 完整的双向交互
- 发送用户消息
- 响应权限请求
- 可发送中断
- 可更新会话标题
- 用于标准远程控制场景

## 远程会话流程图

```mermaid
flowchart TD
    A["RemoteSessionManager.connect()"] --> B["创建SessionsWebSocket"]
    B --> C["选择WebSocket实现"]
    C --> D{"Bun运行时?"}
    D -->|是| E["globalThis.WebSocket"]
    D -->|否| F["动态导入ws包"]
    E --> G["连接到 /v1/sessions/ws/{id}/subscribe"]
    F --> G

    G --> H{"连接结果"}
    H -->|成功| I["启动ping定时器 30秒间隔"]
    I --> J["触发onConnected回调"]
    J --> K["等待消息"]

    K --> L{"收到消息"}
    L -->|SDKMessage| M["触发onMessage回调"]
    L -->|control_request| N["存入pendingPermissionRequests"]
    N --> O["触发onPermissionRequest回调"]
    L -->|control_cancel_request| P["从Map中删除"]
    P --> Q["触发onPermissionCancelled回调"]
    L -->|control_response| R["记录日志"]

    M --> K
    R --> K

    S["用户发送消息"] --> T["sendMessage()"]
    T --> U["HTTP POST sendEventToRemoteSession()"]

    V["用户响应权限"] --> W["respondToPermissionRequest()"]
    W --> X["构建SDKControlResponse"]
    X --> Y["WebSocket.send()"]

    Z["WebSocket关闭"] --> AA{"关闭代码"}
    AA -->|4003| AB["永久关闭 触发onClose"]
    AA -->|4001| AC{"重试次数<3?"}
    AC -->|是| AD["指数退避重连"]
    AC -->|否| AB
    AA -->|其他| AE{"先前connected且重试<5?"}
    AE -->|是| AF["2秒延迟后重连"]
    AE -->|否| AB

    AD --> G
    AF --> G
