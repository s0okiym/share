# 最小 Xray 客户端实现方案：VLESS + XHTTP + REALITY

> 目标：基于 `xtls/xray-core` 构建一个仅做客户端的最小代理程序。
> 入站支持 SOCKS5 与 HTTP；出站仅支持 VLESS + XHTTP(SplitHTTP) + REALITY，用于抗信息审查。

---

## 1. 需求拆解

| 层级 | 需求 | 对应 Xray 包 |
|------|------|-------------|
| 入站协议 | SOCKS5 (可含 UDP) | `proxy/socks` |
| 入站协议 | HTTP Proxy | `proxy/http` |
| 出站协议 | VLESS | `proxy/vless/outbound` |
| 传输层 | XHTTP (SplitHTTP) | `transport/internet/splithttp` |
| 安全层 | REALITY | `transport/internet/reality` |
| 底层网络 | TCP / UDP Dial | `transport/internet/tcp` / `udp` |
| 核心调度 | Dispatcher + Inbound/Outbound Manager | `app/dispatcher`, `app/proxyman/*` |

---

## 2. 实现方案总览

Xray-core 的架构天然支持"按需裁剪"。所有协议/传输/功能都通过各包的 `init()` 函数向全局注册表注册。因此，**只要控制 import 列表，就能精确控制二进制体积和功能范围**。

推荐两条实现路径：

### 路径 A：自定义 Distro（推荐）
- 仿照 `main/distro/all/all.go`，新建一个最小 distro 文件，只 blank-import 需要的包。
- 保留 `main/main.go` 的 CLI 框架（配置加载、信号处理、生命周期管理）。
- 配置格式继续使用 JSON/TOML/YAML，与标准 Xray 兼容。
- **优点**：开发量最小，能直接复用成熟的命令行和配置系统；服务器端配置无需改动即可客户端使用。
- **缺点**：仍然带上了 `main` 包的 CLI 框架，体积不是理论最小。

### 路径 B：纯库 API（嵌入式/无 CLI）
- 不依赖 `main` 包，直接在用户代码里调用 `core.New(config)` 创建实例。
- 配置完全通过 Go 代码拼装 `core.Config` protobuf 结构体。
- **优点**：体积可压到极限；适合嵌入其他应用、移动 App、或作为库使用。
- **缺点**：需要自行处理配置加载、日志、信号、生命周期；对 Xray 内部 protobuf 结构熟悉度要求高。

---

## 3. 最小模块依赖清单（精确到包）

下面列出**必须** blank-import 的包，按功能分组。

```go
package minimal

import (
    // ========== 核心基础设施（缺一不可） ==========
    _ "github.com/xtls/xray-core/app/dispatcher"           // 流量调度中枢
    _ "github.com/xtls/xray-core/app/proxyman/inbound"     // 入站 Handler 生命周期管理
    _ "github.com/xtls/xray-core/app/proxyman/outbound"    // 出站 Handler 生命周期管理

    // ========== 可选但强烈建议保留 ==========
    _ "github.com/xtls/xray-core/app/log"                  // 日志输出（调试必需）
    _ "github.com/xtls/xray-core/app/policy"               // 连接策略/超时（默认会自动注入）
    _ "github.com/xtls/xray-core/app/router"               // 路由（默认会自动注入）
    _ "github.com/xtls/xray-core/app/dns"                  // DNS 解析（默认会自动注入）

    // ========== 入站协议 ==========
    _ "github.com/xtls/xray-core/proxy/socks"              // SOCKS4/4a/5 入站
    _ "github.com/xtls/xray-core/proxy/http"               // HTTP 入站

    // ========== 出站协议 ==========
    _ "github.com/xtls/xray-core/proxy/vless/outbound"     // VLESS 客户端
    // ⚠️ 注意：不要导入 proxy/vless/inbound，那是服务端用的

    // ========== 传输层 ==========
    _ "github.com/xtls/xray-core/transport/internet/tcp"   // TCP 底层 Dial
    _ "github.com/xtls/xray-core/transport/internet/udp"   // UDP 底层 Dial
    _ "github.com/xtls/xray-core/transport/internet/splithttp" // XHTTP 传输
    _ "github.com/xtls/xray-core/transport/internet/reality"   // REALITY 安全层
    _ "github.com/xtls/xray-core/transport/internet/tls"   // TLS 指纹库依赖（uTLS）

    // ========== 配置加载器（若走路径 A） ==========
    _ "github.com/xtls/xray-core/main/json"                // JSON 配置支持
    // _ "github.com/xtls/xray-core/main/toml"              // 如需 TOML
    // _ "github.com/xtls/xray-core/main/yaml"              // 如需 YAML
    _ "github.com/xtls/xray-core/main/confloader/external" // 从文件/HTTP 加载配置
)
```

### 关于若干包的说明

| 包 | 是否必须 | 说明 |
|---|---------|------|
| `app/log` | 强烈建议 | 没有它，初始化失败或运行时错误将完全静默，调试极其困难。 |
| `app/router` | 建议 | `core.New` 会自动注入一个默认 Router；显式导入可确保行为一致，并支持自定义路由规则。 |
| `app/dns` | 建议 | 同上，自动注入默认 DNS。若省略，系统会走 Go 默认 DNS，通常够用。 |
| `app/policy` | 建议 | 控制连接超时、缓冲等。自动注入默认值。 |
| `transport/internet/tls` | **必须** | REALITY 内部依赖 `tls.GetFingerprint()` 获取 uTLS 指纹模板。不导入会导致运行时 panic。 |
| `transport/internet/reality` | **必须** | 虽然 REALITY 不是通过 TransportDialer 注册，而是通过 `SecurityType` 字符串在 StreamConfig 中解析，但此包的 `init()` 会将其 protobuf 类型注册到全局 protoregistry，否则 `serial.GetInstance()` 会失败。 |

---

## 4. 核心配置构建详解

### 4.1 入站：SOCKS5

SOCKS5 Server 在 Xray 中有个特殊设计：**它内嵌了一个 HTTP Server**。当首个字节不是 SOCKS 协议 magic (`0x04`/`0x05`) 时，会自动 fallback 到 HTTP 代理模式。这意味着，如果你只开一个 SOCKS inbound，它实际上可以同时服务 SOCKS 和 HTTP 客户端（但行为并不完全等同于独立的 HTTP inbound，比如认证方式不同）。

若要求独立、标准的 HTTP 代理，建议**同时**配置一个独立的 HTTP inbound。

**SOCKS5 `ServerConfig` 关键字段：**

```protobuf
message ServerConfig {
  AuthType auth_type = 1;          // NO_AUTH (0) 或 PASSWORD (1)
  map<string, string> accounts = 2; // auth_type=PASSWORD 时生效，key=username, value=password
  IPOrDomain address = 3;           // 绑定地址，通常留空或 127.0.0.1
  bool udp_enabled = 4;             // 是否开启 UDP Associate
  uint32 user_level = 6;            // 用户等级，通常 0
}
```

**HTTP `ServerConfig` 关键字段：**

```protobuf
message ServerConfig {
  map<string, string> accounts = 2; // Basic Auth，key=username, value=password
  bool allow_transparent = 3;       // 透明代理开关，客户端用保持 false
  uint32 user_level = 4;
}
```

### 4.2 出站：VLESS + XHTTP + REALITY

出站配置由三层嵌套组成：`OutboundHandlerConfig` → `ProxySettings` (VLESS) + `SenderSettings` (StreamConfig)。

#### VLESS Outbound Config

```protobuf
// proxy/vless/outbound/config.proto
message Config {
  xray.common.protocol.ServerEndpoint vnext = 1;
}
```

`ServerEndpoint` 包含地址、端口、用户列表。每个用户的 `Account` 必须是 `vless.Account`：

```protobuf
// proxy/vless/account.proto
message Account {
  string id = 1;            // UUID，如 "66ad4540-b58c-4ad2-9926-ea63445a9b57"
  string flow = 2;          // 客户端通常留空；若服务端要求 XTLS Vision 则填 "xtls-rprx-vision"
  string encryption = 3;    // ML-KEM/X25519 公钥，通常留空
  // ... 其余字段一般保持默认
}
```

#### StreamConfig（传输 + 安全）

```protobuf
// transport/internet/config.proto
message StreamConfig {
  string protocol_name = 5;                    // "splithttp"
  repeated TransportConfig transport_settings = 2;
  string security_type = 3;                    // "xray.transport.internet.reality.Config"
  repeated serial.TypedMessage security_settings = 4;
  SocketConfig socket_settings = 6;
}
```

#### SplitHTTP Config

```protobuf
// transport/internet/splithttp/config.proto
message Config {
  string host = 1;           // HTTP Host 头（伪装域名）
  string path = 2;           // URL 路径，如 "/xhttp"
  string mode = 3;           // "auto", "stream-one", "stream-up", "packet-up"
  map<string, string> headers = 4;
  XmuxConfig xmux = 12;      // 连接复用配置
  // ... 其余为高级调优项，最小实现可忽略
}
```

**模式选择建议（客户端）：**

| 模式 | 行为 | 与 REALITY 兼容性 | 建议 |
|------|------|------------------|------|
| `auto` | 默认 `packet-up`（无 REALITY）或 `stream-one`（有 REALITY） | 好 | **推荐**，让代码自动选择 |
| `stream-one` | 单条双向 HTTP 流（POST 上行，响应体下行） | **最佳** | REALITY 场景下的首选，延迟低 |
| `packet-up` | 上行分片 POST，下行 GET 长连接 | 一般 | 无 REALITY 时抗检测更好 |
| `stream-up` | 上行 POST + 下行 GET 分离 | 支持 | 需配合 `downloadSettings` |

**注意**：REALITY 会强制 HTTP/2，因此 `stream-one` 在 REALITY 下工作最稳定。

#### REALITY Config（客户端字段）

```protobuf
// transport/internet/reality/config.proto
message Config {
  string Fingerprint = 21;     // TLS 指纹，如 "chrome", "firefox", "safari", "ios"
  string server_name = 22;     // SNI，必须与服务器证书/CAMO 域名匹配
  bytes public_key = 23;       // 服务器 X25519 公钥（Base64 解码后的原始字节）
  bytes short_id = 24;         // Short ID（通常 8 字节十六进制字符串的解码值）
  bytes mldsa65_verify = 25;   // ML-DSA-65 验证公钥，通常留空
  string spider_x = 26;        // 蜘蛛初始路径，通常留空
  repeated int64 spider_y = 27;// 蜘蛛参数，通常留空
}
```

**关键安全参数说明：**

- **`Fingerprint`**：决定发送的 TLS ClientHello 指纹。必须使用一个真实存在的浏览器指纹（如 `"chrome"`），否则审查方可通过指纹识别出这是 Xray 客户端。
- **`server_name`**：SNI 域名。REALITY 的核心原理是向中间人展示一个"真实存在的高可信度网站"的 TLS 握手。此域名必须是服务器配置中 `server_names` 允许的值，且该域名在公网上有真实的 TLS 证书和正常网站内容。
- **`public_key`**：服务器的长期 X25519 公钥。用于客户端与服务器进行密钥交换和证书真实性验证。**必须准确**，否则 REALITY 验证失败会触发 spider fallback 或直接报错。
- **`short_id`**：服务器分配的 short ID，用于在 TLS Session ID 中标识该客户端/配置。

---

## 5. 代码示例

### 5.1 路径 A：自定义 Distro + JSON 配置

**文件：`main/distro/minimal/minimal.go`**

```go
package minimal

import (
    _ "github.com/xtls/xray-core/app/dispatcher"
    _ "github.com/xtls/xray-core/app/dns"
    _ "github.com/xtls/xray-core/app/log"
    _ "github.com/xtls/xray-core/app/policy"
    _ "github.com/xtls/xray-core/app/proxyman/inbound"
    _ "github.com/xtls/xray-core/app/proxyman/outbound"
    _ "github.com/xtls/xray-core/app/router"

    _ "github.com/xtls/xray-core/proxy/http"
    _ "github.com/xtls/xray-core/proxy/socks"
    _ "github.com/xtls/xray-core/proxy/vless/outbound"

    _ "github.com/xtls/xray-core/transport/internet/reality"
    _ "github.com/xtls/xray-core/transport/internet/splithttp"
    _ "github.com/xtls/xray-core/transport/internet/tcp"
    _ "github.com/xtls/xray-core/transport/internet/tls"
    _ "github.com/xtls/xray-core/transport/internet/udp"

    _ "github.com/xtls/xray-core/main/confloader/external"
    _ "github.com/xtls/xray-core/main/json"
)
```

**文件：`main/main.go`（修改 import）**

将原来的：
```go
_ "github.com/xtls/xray-core/main/distro/all"
```
替换为：
```go
_ "github.com/xtls/xray-core/main/distro/minimal"
```

然后正常编译即可：
```bash
CGO_ENABLED=0 go build -o xray-minimal -trimpath -ldflags="-s -w" -v ./main
```

**对应的 `config.json` 示例：**

```json
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "tag": "socks-in",
      "port": 10808,
      "listen": "127.0.0.1",
      "protocol": "socks",
      "settings": {
        "auth": "noauth",
        "udp": true,
        "ip": "127.0.0.1"
      }
    },
    {
      "tag": "http-in",
      "port": 10809,
      "listen": "127.0.0.1",
      "protocol": "http",
      "settings": {
        "allowTransparent": false
      }
    }
  ],
  "outbounds": [
    {
      "tag": "vless-out",
      "protocol": "vless",
      "settings": {
        "vnext": [
          {
            "address": "your.server.com",
            "port": 443,
            "users": [
              {
                "id": "your-uuid-here",
                "flow": "",
                "encryption": "none"
              }
            ]
          }
        ]
      },
      "streamSettings": {
        "network": "splithttp",
        "security": "reality",
        "splithttpSettings": {
          "host": "www.example.com",
          "path": "/xhttp",
          "mode": "auto"
        },
        "realitySettings": {
          "fingerprint": "chrome",
          "serverName": "www.example.com",
          "publicKey": "YOUR_BASE64_PUBLIC_KEY",
          "shortId": "0123456789abcdef",
          "spiderX": ""
        }
      }
    }
  ]
}
```

### 5.2 路径 B：纯库调用（Programmatic Config）

适合不想维护 JSON 文件、想把配置硬编码或从自己的配置格式转换的场景。

```go
package main

import (
    "context"
    "encoding/base64"
    "log"
    "os"
    "os/signal"
    "syscall"

    "github.com/xtls/xray-core/app/dispatcher"
    "github.com/xtls/xray-core/app/dns"
    "github.com/xtls/xray-core/app/log"
    "github.com/xtls/xray-core/app/policy"
    "github.com/xtls/xray-core/app/proxyman"
    "github.com/xtls/xray-core/app/router"
    "github.com/xtls/xray-core/common/net"
    "github.com/xtls/xray-core/common/protocol"
    "github.com/xtls/xray-core/common/serial"
    "github.com/xtls/xray-core/core"
    httproxy "github.com/xtls/xray-core/proxy/http"
    "github.com/xtls/xray-core/proxy/socks"
    vlessout "github.com/xtls/xray-core/proxy/vless/outbound"
    vlessacct "github.com/xtls/xray-core/proxy/vless"
    "github.com/xtls/xray-core/transport/internet"
    "github.com/xtls/xray-core/transport/internet/reality"
    "github.com/xtls/xray-core/transport/internet/splithttp"

    // 以下 blank imports 用于触发 init() 注册
    _ "github.com/xtls/xray-core/app/proxyman/inbound"
    _ "github.com/xtls/xray-core/app/proxyman/outbound"
    _ "github.com/xtls/xray-core/proxy/http"
    _ "github.com/xtls/xray-core/proxy/socks"
    _ "github.com/xtls/xray-core/proxy/vless/outbound"
    _ "github.com/xtls/xray-core/transport/internet/reality"
    _ "github.com/xtls/xray-core/transport/internet/splithttp"
    _ "github.com/xtls/xray-core/transport/internet/tcp"
    _ "github.com/xtls/xray-core/transport/internet/tls"
    _ "github.com/xtls/xray-core/transport/internet/udp"
)

func mustDecodeB64(s string) []byte {
    b, err := base64.StdEncoding.DecodeString(s)
    if err != nil {
        panic(err)
    }
    return b
}

func buildConfig() *core.Config {
    // ---------- 入站：SOCKS5 ----------
    socksInbound := &core.InboundHandlerConfig{
        Tag: "socks-in",
        ReceiverSettings: serial.ToTypedMessage(&proxyman.ReceiverConfig{
            PortList: &net.PortList{Range: []*net.PortRange{
                net.SinglePortRange(net.Port(10808)),
            }},
            Listen: net.NewIPOrDomain(net.LocalHostIP),
        }),
        ProxySettings: serial.ToTypedMessage(&socks.ServerConfig{
            AuthType:   socks.AuthType_NO_AUTH,
            UdpEnabled: true,
            Address:    net.NewIPOrDomain(net.LocalHostIP),
        }),
    }

    // ---------- 入站：HTTP ----------
    httpInbound := &core.InboundHandlerConfig{
        Tag: "http-in",
        ReceiverSettings: serial.ToTypedMessage(&proxyman.ReceiverConfig{
            PortList: &net.PortList{Range: []*net.PortRange{
                net.SinglePortRange(net.Port(10809)),
            }},
            Listen: net.NewIPOrDomain(net.LocalHostIP),
        }),
        ProxySettings: serial.ToTypedMessage(&httproxy.ServerConfig{}),
    }

    // ---------- 出站：VLESS ----------
    vlessOutbound := &core.OutboundHandlerConfig{
        Tag: "vless-out",
        ProxySettings: serial.ToTypedMessage(&vlessout.Config{
            Vnext: []*protocol.ServerEndpoint{{
                Address: net.NewIPOrDomain(net.DomainAddress("your.server.com")),
                Port:    443,
                User: []*protocol.User{{
                    Account: serial.ToTypedMessage(&vlessacct.Account{
                        Id: "your-uuid-here",
                    }),
                }},
            }},
        }),
        SenderSettings: serial.ToTypedMessage(&proxyman.SenderConfig{
            StreamSettings: &internet.StreamConfig{
                ProtocolName: "splithttp",
                TransportSettings: []*internet.TransportConfig{{
                    ProtocolName: "splithttp",
                    Settings: serial.ToTypedMessage(&splithttp.Config{
                        Host: "www.example.com",
                        Path: "/xhttp",
                        Mode: "auto", // REALITY 下自动转 stream-one
                    }),
                }},
                SecurityType: "xray.transport.internet.reality.Config",
                SecuritySettings: []*serial.TypedMessage{
                    serial.ToTypedMessage(&reality.Config{
                        Fingerprint: "chrome",
                        ServerName:  "www.example.com",
                        PublicKey:   mustDecodeB64("YOUR_BASE64_PUBLIC_KEY"),
                        ShortId:     mustDecodeB64("ABCD1234"), // 注意：这里实际是 raw bytes，需确认服务端生成方式
                    }),
                },
            },
        }),
    }

    // ---------- 核心 App ----------
    return &core.Config{
        App: []*serial.TypedMessage{
            serial.ToTypedMessage(&log.Config{}),
            serial.ToTypedMessage(&dispatcher.Config{}),
            serial.ToTypedMessage(&proxyman.InboundConfig{}),
            serial.ToTypedMessage(&proxyman.OutboundConfig{}),
            serial.ToTypedMessage(&policy.Config{}),
            serial.ToTypedMessage(&dns.Config{}),
            serial.ToTypedMessage(&router.Config{}),
        },
        Inbound:  []*core.InboundHandlerConfig{socksInbound, httpInbound},
        Outbound: []*core.OutboundHandlerConfig{vlessOutbound},
    }
}

func main() {
    config := buildConfig()

    instance, err := core.New(config)
    if err != nil {
        log.Fatalf("failed to create instance: %v", err)
    }

    if err := instance.Start(); err != nil {
        log.Fatalf("failed to start instance: %v", err)
    }
    defer instance.Close()

    log.Println("minimal client started: socks=127.0.0.1:10808 http=127.0.0.1:10809")

    // 阻塞等待信号
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    <-sigCh
    log.Println("shutting down...")
}
```

**几点代码注意事项：**

1. `serial.ToTypedMessage(...)` 是 Xray 中将具体 protobuf message 包装成 `*serial.TypedMessage` 的常用方法。它内部通过 `protoregistry.GlobalTypes` 查找类型的全名，因此对应的包必须被 import（即使 blank import）以触发 `init()` 注册 protobuf 描述符。
2. `net.SinglePortRange` 和 `net.Port` 在 `common/net/port.go` 中定义。
3. `proxyman.SenderConfig` 是出站连接的发送端配置，包含 `StreamSettings`。它的 protobuf 定义在 `app/proxyman/config.proto` 中，生成的 Go 文件里需要确认字段名。如果找不到，可以直接引用 `app/proxyman/config.pb.go`。

---

## 6. 开发难度评估

| 维度 | 难度 | 说明 |
|------|------|------|
| 理解 Xray 注册机制 | ⭐⭐ 中等 | 需要理解 `common.RegisterConfig` + `init()` 的全局注册模式，但一旦掌握，裁剪逻辑非常清晰。 |
| 配置拼装（Programmatic） | ⭐⭐⭐ 较难 | protobuf 嵌套层级深，字段多，需要反复查阅 `.pb.go` 文件；缺乏 IDE 自动补全会很痛苦。 |
| REALITY 参数获取 | ⭐⭐ 中等 | 需要理解 `public_key`、`short_id`、`server_name` 的生成和对应关系，建议先用标准 Xray 服务端跑通再移植。 |
| xhttp 模式调优 | ⭐⭐⭐ 较难 | `auto` 模式足够起步，但想针对特定网络环境（如高延迟、丢包）优化，需要理解 `stream-one` / `packet-up` / `xmux` 的区别。 |
| 编译与体积优化 | ⭐ 简单 | Go 的编译本身简单；若追求极致体积，可配合 `-ldflags="-s -w"` 和 `upx`。 |
| 整体工期估算 | | 有 Go 基础 + 熟悉 Xray 配置：1~2 天；从零开始：3~5 天。 |

---

## 7. 权衡与注意事项

### 7.1 体积 vs 功能

- 即使只做最小裁剪，由于 Xray-core 引入了 `utls`、`quic-go`、`gvisor` 等重型依赖，最终二进制体积仍在 **30~50 MB** 左右（未经压缩）。
- 若体积敏感（如路由器、嵌入式），可考虑：
  - 使用 `upx` 压缩（可减至 10~15 MB，但启动时延增加）。
  - 进一步裁剪：去掉 `app/dns`、`app/router`、`app/log` 的显式导入，依赖 `core.New` 的自动注入（节省的是代码量，不是依赖库体积）。
  - 使用 Go 的 `//go:build` 标签配合自定义构建脚本，彻底剥离不需要的依赖（**高阶操作**，需要修改大量文件）。

### 7.2 SOCKS5 的 HTTP Fallback 行为

- `proxy/socks` 的 `Server` 会读取连接首字节判断协议版本。
- 如果客户端发送的是 HTTP 请求（首字节不是 `0x04`/`0x05`），它会将连接交给内嵌的 `http.Server` 处理。
- 但这个 fallback HTTP Server **共享 SOCKS 的端口**，且认证逻辑与独立 HTTP inbound 不同（SOCKS 的 `accounts` map 不会自动同步到 HTTP Basic Auth）。
- **建议**：生产环境同时开启独立的 SOCKS 和 HTTP inbound，不要依赖 fallback。

### 7.3 REALITY 的"前置条件"

REALITY 不是简单的 TLS 伪装，它要求：

1. **服务器端必须正确配置 REALITY**，并生成匹配的 `private_key` / `public_key` 对。
2. `server_name` 必须是公网上真实存在的、有健康 HTTPS 服务的域名，且该域名在服务端 `server_names` 白名单内。
3. 客户端时间必须基本准确（服务端可配置 `max_time_diff` 容忍偏差，但偏差过大会导致握手失败）。
4. 如果 REALITY 验证失败，客户端会进入 **spider fallback** 模式：向目标域名发起真实的 HTTP/2 爬虫请求，模拟正常浏览器行为，然后返回错误。这意味着：
   - 配置错误时不会立即崩溃，但代理功能不可用。
   - 会有额外的、看似正常的 HTTPS 流量产生（这是设计特性，用于迷惑审查方）。

### 7.4 UDP 支持

- SOCKS5 inbound 开启 `udp_enabled: true` 后，支持 UDP Associate。
- XHTTP (SplitHTTP) 底层基于 HTTP，**原生是 TCP 传输**。UDP 流量会被封装在 VLESS 的 UDP-over-TCP 机制中传输（VLESS Protocol 的 `RequestCommandUDP`）。
- 如果审查方对 UDP 有严格 QoS 或阻断，XHTTP 的 TCP 封装反而是一种优势；但如果需要原生 UDP 性能（如游戏、VoIP），延迟会比直连高。

### 7.5 DNS 与路由的默认行为

- 最小客户端若不显式配置 DNS 和路由，Xray 会自动注入默认实现：
  - DNS：使用系统默认 DNS 解析器。
  - Router：无规则，所有流量走默认 outbound（即你配置的 VLESS）。
- 这对"最小客户端"通常是期望行为。若想实现分流（如大陆直连、海外走代理），需要引入 `app/router` 规则并配置 `geoip.dat` / `geosite.dat`，这会显著增加复杂度。

### 7.6 日志与可观测性

- 最小裁剪后若去掉 `app/log`，错误信息将完全丢失。
- 建议保留 `app/log` 并在 `core.Config` 中配置 `log.Config`：
  ```go
  &log.Config{
      ErrorLogType:  log.LogType_Console,
      ErrorLogLevel: log.LogLevel_Warning,
      AccessLogType: log.LogType_None,
  }
  ```

### 7.7 升级维护

- Xray-core 更新频繁（尤其是 REALITY 和 XHTTP 仍在快速迭代）。
- 最小客户端因为 import 列表是手写的，每次上游升级后可能需要检查：
  - 是否有新的强制依赖包被引入。
  - protobuf 定义是否有破坏性变更。
  - REALITY / xhttp 的 config 字段是否有新增必填项。
- **建议**：保留标准 Xray 作为对比基准，每次升级先在标准版本验证配置正确性，再同步到最小客户端。

### 7.8 关于 `proxy/vless/outbound` 的 Flow 字段

- `flow` 字段控制是否启用 XTLS Vision 等高级特性。
- 如果服务器端未启用 XTLS Vision，客户端必须将 `flow` 留空（`""`），否则握手会失败。
- 如果服务器端要求 `"xtls-rprx-vision"`，则客户端必须填写相同值，且外层 TLS 必须是 TLS 1.3。
- **最小客户端建议**：先不启用 XTLS Vision，只使用普通 VLESS + REALITY + XHTTP。这已具备极强的抗检测能力，且复杂度更低。

---

## 8. 安全与审查对抗要点

1. **TLS 指纹必须真实**：`fingerprint` 字段不可乱填。常用值 `"chrome"` `"firefox"` `"safari"` `"ios"` `"android"` `"edge"`。不要填 `random` 或自定义字符串，否则等于主动暴露身份。

2. **SNI 与目标站点的合理性**：`server_name` 应该是一个真实、热门、长期在线的 HTTPS 网站域名（如 CDN 域名、大站域名）。避免使用小众或已下线的域名，因为审查方可能通过 SNI 黑名单或主动探测发现异常。

3. **XHTTP 路径的自然性**：`path` 建议模仿真实 API 路径或随机字符串，避免使用 `/vless`、`/proxy`、`/xray` 等具有明显协议特征的路径。

4. **Host 头与 SNI 一致性**：SplitHTTP 的 `host` 字段通常应与 REALITY 的 `server_name` 保持一致，或者至少属于同一可信域名体系，防止 HTTP 层与 TLS 层的域名信息矛盾。

5. **不要开启 AllowInsecure**：REALITY 的设计目的之一就是消除对 `allowInsecure` 的依赖。如果配置中混入了普通 TLS 的 `allowInsecure: true`，会彻底破坏安全模型。

6. **Short ID 的随机性**：`short_id` 是服务端从 `short_ids` 池中分配的一个值，用于多用户/多配置隔离。客户端应使用服务端分配的具体值，不应自行编造。

7. **时间同步**：REALITY 握手嵌入 Unix 时间戳。客户端系统时间若偏差过大（默认超过数十秒），服务端会拒绝握手。确保运行环境有 NTP 同步。

---

## 9. 总结

实现一个仅支持 SOCKS5 + HTTP 入站、VLESS + XHTTP + REALITY 出站的最小 Xray 客户端，**技术上完全可行**，核心思路是：

1. **控制 import**：不导入 `main/distro/all`，而是自建最小 distro，只 blank-import 需要的包。
2. **保留核心 App**：`dispatcher`、`proxyman/inbound`、`proxyman/outbound` 是运行时的基础设施，不可省略。
3. **传输层链**：`tcp` → `reality` (utls指纹) → `splithttp` (HTTP/2)。`tls` 包必须导入以提供指纹模板。
4. **配置方式**：可以复用 JSON 配置（路径 A），也可以纯代码拼装 protobuf（路径 B）。

**开发难度总体为中等**。最大挑战不在于代码量（实际新增代码很少），而在于理解 Xray 的模块注册机制、protobuf 配置结构、以及 REALITY / XHTTP 各参数的安全含义。建议在实施前，先用标准 Xray 客户端 + 服务端完整跑通 VLESS+XHTTP+REALITY 链路，确认配置参数无误后，再迁移到最小客户端。
