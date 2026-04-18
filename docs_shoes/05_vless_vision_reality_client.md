# VLESS + VISION + REALITY 客户端支持说明

## 结论

**`shoes` 完全支持作为客户端连接远程 xray 的 `VLESS + VISION + REALITY` 服务。**

该组合在 `shoes` 中是一个一等公民特性：配置校验会强制验证协议层关系，Handler 工厂有专门的 `new_vision_vless()` 构造路径，且仓库自带可直接运行的示例配置。

---

## 协议栈层次

```
┌─────────────────────────────────────────────┐
│  Application Data (HTTP/HTTPS/...)           │
├─────────────────────────────────────────────┤
│  VLESS Protocol                              │  ← user_id 认证 + 目标地址
├─────────────────────────────────────────────┤
│  XTLS VISION                                │  ← TLS-in-TLS 零拷贝流控 + 填充
├─────────────────────────────────────────────┤
│  REALITY Handshake                          │  ← X25519 / HKDF / AES-256-GCM
├─────────────────────────────────────────────┤
│  TLS 1.3 Record Layer (mimicry)             │  ← 外观像正常 TLS 到 www.google.com
├─────────────────────────────────────────────┤
│  TCP (or QUIC if layered)                   │
└─────────────────────────────────────────────┘
```

- **REALITY**：提供审查抵抗，TLS 1.3 握手伪装，使用 X25519 公钥和 `short_id` 做服务端认证。
- **VISION**（XTLS-RPRX-Vision）：针对 TLS-in-TLS 场景的零拷贝优化流控，带填充以模仿真实 TLS 流量模式。
- **VLESS**：内层代理协议，通过 `user_id` 做用户认证，携带最终目标地址。

---

## 配置结构

在 `client_chain`（或 `client_config_group` 中引用的链）中，协议层采用**嵌套结构**：外层是 `reality`，其 `protocol` 字段内嵌 `vless`。

### 字段说明

| 字段路径 | 类型 | 必填 | 说明 |
|---------|------|------|------|
| `address` | `string` | 是 | REALITY 服务端地址，格式 `host:port`，例如 `your-server.example.com:443` |
| `protocol.type` | `string` | 是 | 固定为 `reality` |
| `protocol.public_key` | `string` | 是 | 服务端 X25519 公钥，Base64URL 编码 |
| `protocol.short_id` | `string` | 是 | 服务端 short ID，十六进制，最长 16 字符 |
| `protocol.sni_hostname` | `string` | 是 | TLS 握手使用的 SNI，例如 `www.google.com` |
| `protocol.vision` | `bool` | 否 | 是否启用 XTLS-Vision。**启用时必须内层为 VLESS** |
| `protocol.protocol.type` | `string` | 是（vision=true 时） | 固定为 `vless` |
| `protocol.protocol.user_id` | `string` | 是（vision=true 时） | VLESS UUID，例如 `b85798ef-e9dc-46a4-9a87-8da4499d36d0` |
| `protocol.protocol.udp_enabled` | `bool` | 否 | VLESS 是否支持 UDP（默认取决于实现） |

---

## 完整配置示例

### 示例 1：本地 SOCKS5 代理出站走 VLESS+VISION+REALITY

```yaml
# 在本地 1080 开启 SOCKS5 入站
# 所有流量通过 VLESS + VISION + REALITY 连接到远程 xray 服务端
- address: "127.0.0.1:1080"
  protocol:
    type: socks
  rules:
    - masks: "0.0.0.0/0"
      action: allow
      client_chain:
        - address: "your-server.example.com:443"
          protocol:
            type: reality
            public_key: "SERVER_PUBLIC_KEY_HERE"
            short_id: "0123456789abcdef"
            sni_hostname: "www.google.com"
            vision: true
            protocol:
              type: vless
              user_id: "b85798ef-e9dc-46a4-9a87-8da4499d36d0"
```

### 示例 2：配合规则分流使用

```yaml
- address: "127.0.0.1:1080"
  protocol:
    type: mixed   # 自动探测 HTTP / SOCKS5
  dns:
    servers: default
  rules:
    # 国内直连
    - masks:
        - "geoip:cn"
      action: allow
      client_chain:
        - address: direct

    # 其他全部走 VLESS+VISION+REALITY
    - masks: "0.0.0.0/0"
      action: allow
      client_chain:
        - address: "your-server.example.com:443"
          protocol:
            type: reality
            public_key: "SERVER_PUBLIC_KEY_HERE"
            short_id: "0123456789abcdef"
            sni_hostname: "www.google.com"
            vision: true
            protocol:
              type: vless
              user_id: "b85798ef-e9dc-46a4-9a87-8da4499d36d0"
```

### 示例 3：多跳链（先 Shadowsocks 中继，再 VLESS+VISION+REALITY）

```yaml
rules:
  - masks: "0.0.0.0/0"
    action: allow
    client_chain:
      # 第一跳：Shadowsocks 中继
      - address: "relay.example.com:1080"
        protocol:
          type: shadowsocks
          method: "aes-256-gcm"
          password: "relay-password"
      # 第二跳：VLESS+VISION+REALITY
      - address: "your-server.example.com:443"
        protocol:
          type: reality
          public_key: "SERVER_PUBLIC_KEY_HERE"
          short_id: "0123456789abcdef"
          sni_hostname: "www.google.com"
          vision: true
          protocol:
            type: vless
            user_id: "b85798ef-e9dc-46a4-9a87-8da4499d36d0"
```

> **注意**：多跳链中 `vision: true` 只能出现在最后一跳或承载最终 TLS 的跳上，且其内层必须仍为 VLESS。

---

## 配置校验行为

`shoes` 在启动时会**严格校验**以下约束，不满足则直接报错退出：

1. **`vision: true` 时内层必须是 VLESS**  
   校验位置：`src/config/validate.rs` → `validate_client_vision_protocol()`  
   错误示例：
   ```
   Reality client config has vision=true but inner protocol is Shadowsocks (not VLESS).
   Vision (XTLS-RPRX-Vision) requires VLESS as the inner protocol.
   ```

2. **`reality` 配置必须包含 `public_key`、`short_id`、`sni_hostname`**  
   校验位置：`src/config/validate.rs` → `validate_client_proxy_structure()`

3. **`short_id` 长度不超过 16 个十六进制字符**

---

## 代码实现路径

若需从源码验证或调试，以下是关键文件与函数：

| 文件 | 关键函数/类型 | 作用 |
|------|--------------|------|
| `src/config/types/client.rs` | `ClientProxyConfig::Reality` | 配置结构，`vision: bool` 字段 |
| `src/config/validate.rs` | `validate_client_vision_protocol()` | 强制 vision 必须搭配 VLESS |
| `src/tcp/tcp_client_handler_factory.rs` | `create_tcp_client_handler()` | 工厂：Reality + vision → `RealityClientHandler::new_vision_vless()` |
| `src/reality_client_handler.rs` | `new_vision_vless()` | REALITY 客户端 handler，持有 VLESS UUID |
| `src/reality_client_handler.rs` | `setup_client_tcp_stream()` | REALITY 握手后调用 VISION 初始化 |
| `src/vless/vless_client_handler.rs` | `setup_custom_tls_vision_vless_client_stream()` | 写 VLESS header + 包装 `VisionStream::new_client()` |
| `src/vless/vision_stream.rs` | `VisionStream::new_client()` | VISION 流控与填充 |
| `examples/reality_vision_client.yaml` | 完整示例 | 可直接复制修改使用 |

---

## 服务端兼容性说明

该客户端配置与 **xray-core** 的以下服务端配置兼容：

```json
{
  "protocol": "vless",
  "settings": {
    "clients": [{
      "id": "b85798ef-e9dc-46a4-9a87-8da4499d36d0",
      "flow": "xtls-rprx-vision"
    }]
  },
  "streamSettings": {
    "network": "tcp",
    "security": "reality",
    "realitySettings": {
      "dest": "www.google.com:443",
      "serverNames": ["www.google.com"],
      "privateKey": "...",
      "shortIds": ["0123456789abcdef"]
    }
  }
}
```

关键点：
- xray 服务端 `flow` 需为 `"xtls-rprx-vision"`
- xray 服务端 `security` 需为 `"reality"`
- `user_id` / `public_key` / `short_id` / `sni_hostname` 必须匹配

---

## 已知限制与注意事项

1. **VISION 仅支持 TCP**：VISION 是为 TLS-in-TLS 零拷贝设计的，UDP 流量不会经过 VISION 流控逻辑。
2. **REALITY 外层为 TLS 1.3 mimicry，不提供额外加密**：数据保密性仍依赖内层 VLESS + TLS（如果最终目标是 HTTPS）。
3. **`vision: true` 不支持与 `Vmess`、`Trojan`、`Shadowsocks` 等作为内层协议**：配置校验会阻止这种组合。
4. 若服务端更换了 `public_key` 或 `short_id`，客户端配置必须同步更新，否则 REALITY 握手会失败。
