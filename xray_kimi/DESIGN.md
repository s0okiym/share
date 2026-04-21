# Rust VLESS + REALITY + Vision 客户端实现方案

## 1. 项目目标

实现一个轻量级、高性能的 Rust 代理客户端程序：
- **入站 (Inbound)**：支持 SOCKS5 和 HTTP Mixed（自动协议识别）代理
- **出站 (Outbound)**：支持 VLESS 协议，配合 REALITY 传输层和 XTLS Vision 流控
- **平台**：Linux / macOS / Windows

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           客户端程序 (Rust)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  入站层 (Inbound)                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ SOCKS5 Server│  │ HTTP Server  │  │ Mixed Server │                   │
│  │  (TCP/UDP)   │  │  (CONNECT)   │  │ 首字节嗅探   │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         └─────────────────┴─────────────────┘                           │
│                           │                                             │
│  会话管理层               ▼                                             │
│  ┌─────────────────────────────────────────┐                            │
│  │     Session Manager (目标地址路由)        │                            │
│  │  - 从 SOCKS5/HTTP 提取 target_addr       │                            │
│  │  - 管理连接生命周期                       │                            │
│  └─────────────────────────────────────────┘                            │
│                           │                                             │
│  出站层 (Outbound)        ▼                                             │
│  ┌─────────────────────────────────────────┐                            │
│  │         VLESS Outbound Handler          │                            │
│  │  ┌───────────────────────────────────┐  │                            │
│  │  │  VLESS Protocol Encoder/Decoder   │  │                            │
│  │  │  - Request Header 编码             │  │                            │
│  │  │  - Response Header 解码            │  │                            │
│  │  │  - Address (IPv4/IPv6/Domain) 编解码│  │                            │
│  │  └───────────────────────────────────┘  │                            │
│  │  ┌───────────────────────────────────┐  │                            │
│  │  │      XTLS Vision Flow             │  │                            │
│  │  │  - TrafficState 状态机             │  │                            │
│  │  │  - TLS 流量识别 (XtlsFilterTls)    │  │                            │
│  │  │  - Padding / Unpadding             │  │                            │
│  │  │  - DirectCopy 切换                 │  │                            │
│  │  └───────────────────────────────────┘  │                            │
│  └─────────────────────────────────────────┘                            │
│                           │                                             │
│  传输层 (Transport)       ▼                                             │
│  ┌─────────────────────────────────────────┐                            │
│  │           REALITY Client                │                            │
│  │  - utls 指纹伪装 (craftls)              │                            │
│  │  - ClientHello SessionId 嵌入身份       │                            │
│  │  - X25519 AuthKey 计算                  │                            │
│  │  - 证书自定义验证 (HMAC-SHA512)         │                            │
│  │  - Spider Fallback (可选)               │                            │
│  └─────────────────────────────────────────┘                            │
│                           │                                             │
│  网络层                   ▼                                             │
│  ┌─────────────────────────────────────────┐                            │
│  │          TCP Connection                 │                            │
│  └─────────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 入站层设计 (SOCKS5 + HTTP Mixed)

### 3.1 Mixed 协议自动识别

参考 Xray-core 的 `proxy/socks/server.go` 实现：

1. TCP 连接建立后，读取 **第一个字节**
2. 根据首字节判断协议类型：
   - `0x05` → SOCKS5
   - `0x04` → SOCKS4/4a
   - 其他 (`G`/`P`/`H`/`C`/`D` 等 ASCII 字符) → HTTP
3. 将首字节放回 reader，交给对应的协议处理器

### 3.2 SOCKS5 协议实现

**认证协商阶段** (`auth5`)：
```
Client -> Server: [VER=5, NMETHODS, METHODS...]
Server -> Client: [VER=5, SELECTED_METHOD]
```
- 支持 `0x00` (No Auth) 和 `0x02` (Username/Password)
- 若配置要求密码认证但客户端不支持，返回 `0xFF`

**密码认证阶段** (`ReadUsernamePassword`, RFC 1929)：
```
Client -> Server: [VER=1, ULEN, UNAME, PLEN, PASSWD]
Server -> Client: [VER=1, STATUS]
```

**请求阶段** (`handshake5`)：
```
Client -> Server: [VER=5, CMD, RSV=0, ATYP, DST.ADDR, DST.PORT]
Server -> Client: [VER=5, REP, RSV=0, BND.ADDR, BND.PORT]
```
- `CMD=0x01` (CONNECT) → 建立 TCP 代理隧道
- `CMD=0x03` (UDP ASSOCIATE) → 返回 UDP relay 地址，保持 TCP 控制连接
- `CMD=0x02` (BIND) → 返回错误（不支持）

**地址类型编码** (`ATYP`)：
- `0x01` → IPv4 (4 bytes)
- `0x04` → IPv6 (16 bytes)
- `0x03` → Domain (1 byte length + N bytes域名)

**SOCKS5 UDP ASSOCIATE**：
1. 响应中包含服务器 UDP relay 的监听地址和端口
2. 保持 TCP 控制连接（客户端断开 TCP 则 UDP relay 终止）
3. UDP 数据包格式：`[RSV=0x0000, FRAG=0x00, ATYP, DST.ADDR, DST.PORT, DATA...]`
4. 需要实现 UDP 白名单过滤（记录 TCP ASSOCIATE 时的客户端 IP）

### 3.3 HTTP CONNECT 代理实现

**处理流程**：
1. 使用 `bufio` 读取 HTTP 请求行
2. 解析 `CONNECT target:port HTTP/1.1`
3. 若配置了认证，检查 `Proxy-Authorization: Basic base64(user:pass)`
4. 返回 `HTTP/1.1 200 Connection established\r\n\r\n`
5. 此后连接变为双向隧道，直接转发数据

**普通 HTTP 代理** (可选)：
- 处理非 CONNECT 请求 (GET/POST/PUT 等)
- 移除 Hop-by-hop 头部
- 重组请求并转发

---

## 4. 出站层设计 (VLESS Protocol)

### 4.1 VLESS 请求头格式

参考 `proxy/vless/encoding/encoding.go`：

```
+-----------+----------+----------------+------------------+----------+-------------------+
|  Version  |   UUID   | Addons Length  |  Addons Protobuf | Command  |  Address + Port   |
|  (1 byte) | (16 bytes)|   (1 byte)    |  (variable)      | (1 byte) |    (variable)     |
+-----------+----------+----------------+------------------+----------+-------------------+
```

**字段详解**：
- **Version**：固定为 `0`
- **UUID**：16 字节原始 UUID。注意 Xray 的 `ProcessUUID` 会将第 6、7 字节清零，认证时只比对前 14 字节 + 清零后的 2 字节
- **Addons Length**：1 字节，Addons protobuf 的长度
- **Addons**：若 flow 为 `xtls-rprx-vision`，protobuf 编码 `Addons{Flow: "xtls-rprx-vision"}`；否则长度为 0
- **Command**：
  - `0x01` = TCP
  - `0x02` = UDP
  - `0x03` = Mux (地址自动为 `v1.mux.cool`)
- **Address + Port**：使用 SOCKS5 风格的地址编码 (1 byte type + data)，Port 为 2 bytes Big Endian

### 4.2 VLESS 响应头格式

```
+-----------+----------------+------------------+
|  Version  | Addons Length  |   Addons Bytes   |
|  (1 byte) |   (1 byte)     |  (Length bytes)  |
+-----------+----------------+------------------+
```

- Version 必须与请求一致（当前为 0）
- 当前版本入站端响应 Addons 通常为空（Flow 字段被注释掉）

### 4.3 数据流传输模式

**无 XTLS (plain VLESS)**：
- 上行：客户端请求头后直接跟 payload
- 下行：响应头后直接跟 payload
- UDP：每个包前加 2 字节大端长度

**有 XTLS Vision**：
- 上行通过 `VisionWriter` 包装，添加 padding
- 下行通过 `VisionReader` 包装，去除 padding

---

## 5. XTLS Vision Flow 实现

### 5.1 核心目标

1. **消除 TLS in TLS 特征**：通过 padding 隐藏 VLESS 协议本身的长度特征
2. **在确认外层为 TLS 1.3 后，直接透传原始 TLS 数据**，避免二次加解密开销

### 5.2 TrafficState 状态机

```rust
pub struct TrafficState {
    pub user_uuid: [u8; 16],
    pub number_of_packet_to_filter: i32,  // 初始为 8
    pub enable_xtls: bool,                // 是否为 TLS 1.3 + 现代 CipherSuite
    pub is_tls_12_or_above: bool,
    pub is_tls: bool,
    pub cipher: u16,                      // TLS Cipher Suite
    pub remaining_server_hello: i32,      // 剩余待解析的 ServerHello 长度
    pub inbound: InboundState,
    pub outbound: OutboundState,
}

pub struct OutboundState {
    pub within_padding_buffers: bool,
    pub direct_copy: bool,
    pub remaining_command: i32,
    pub remaining_content: i32,
    pub remaining_padding: i32,
}
```

### 5.3 TLS 流量识别 (XtlsFilterTls)

**ClientHello 检测**：
- 以 `0x16 0x03` 开头且第 6 字节为 `0x01`

**ServerHello 检测**：
- 以 `0x16 0x03 0x03` 开头且第 6 字节为 `0x02`
- 解析长度字段
- 查找 `supported_versions` 扩展 (`0x00 0x2b 0x00 0x02 0x03 0x04`) 确认 TLS 1.3
- 若 CipherSuite 为 TLS 1.3 套件（非 `TLS_AES_128_CCM_8_SHA256`），则 `enable_xtls = true`

### 5.4 Padding 格式

```
[UserUUID: 16 bytes, 仅首包] [Command: 1 byte] [ContentLen: 2 bytes] [PaddingLen: 2 bytes] [Content: N bytes] [Padding: M bytes]
```

**Command 类型**：
- `0x00` = Continue (继续 padding 模式)
- `0x01` = End (退出 padding 模式)
- `0x02` = Direct (退出 padding 并启用 DirectCopy)

**Padding 策略**：
- 非 TLS 流量或 TLS 握手阶段：生成较长的随机 padding（默认 seed: `{900, 500, 900, 256}`）
- 每个 buffer 大小限制为 ≤ `buf_size - 21`，确保有足够空间附加 padding 头

### 5.5 DirectCopy 切换

**上行 (客户端 → 服务器)**：
1. `VisionWriter` 检测 TLS Application Data Record (`0x17 0x03 0x03`)
2. 若 `enable_xtls == true`，发送 `CommandPaddingDirect`，随后切换到直接透传模式
3. 后续数据直接写入底层 TLS 连接，不再 padding

**下行 (服务器 → 客户端)**：
1. `VisionReader` 读取并解析 padding
2. 收到 `CommandPaddingDirect` 后，设置 `direct_copy = true`
3. 提取 TLS 库中已缓冲但未处理的原始数据（`input` / `rawInput`）
4. 后续直接从底层 TCP 连接读取原始加密数据，绕过 TLS 解析

> **Rust 实现注意**：Go 中使用 `unsafe` + `reflect` 直接访问 `crypto/tls.Conn` 的内部 `input` 和 `rawInput` 字段。在 Rust 中，若使用 `rustls`，需要通过其 Session API 或考虑在适当时机直接切换到底层 TCP stream 进行 splice 操作。由于 Rust 中无 unsafe 反射访问 TLS 内部的机制，可考虑：
> 1. 在 Rustls 中利用其内部状态获取已缓冲数据
> 2. 或采用更保守的方案：在 DirectCopy 信号后，关闭当前 TLS reader，新建一个直接从 TCP socket 读取的 reader，依赖操作系统缓冲区保证数据不丢失

---

## 6. REALITY 传输层实现

### 6.1 核心设计理念

REALITY 的核心是：**服务器不持有 TLS 证书，而是"偷取"伪装目标的 TLS 握手数据，随后用动态生成的 Ed25519 证书完成握手。客户端通过 Pre-Shared Key (AuthKey) 验证这张"假证书"的真实性。**

### 6.2 客户端握手流程

```
1. 建立到 VLESS 服务器的原始 TCP 连接
2. 使用 utls/craftls 构建 ClientHello，应用 TLS 指纹伪装 (Chrome/Firefox/Safari)
3. 在 ClientHello.SessionId 中嵌入加密身份令牌
4. 发送 ClientHello，完成 TLS 1.3 握手
5. 通过 VerifyPeerCertificate 验证服务器返回的证书签名
```

### 6.3 ClientHello SessionId 构造

```rust
let mut session_id = [0u8; 32];
session_id[0] = version_x;   // Xray 版本号 x
session_id[1] = version_y;   // Xray 版本号 y
session_id[2] = version_z;   // Xray 版本号 z
session_id[3] = 0;           // reserved

// [4:8] = Unix 时间戳 (Big Endian u32)
big_endian::write_u32(&mut session_id[4..8], now_timestamp());

// [8:16] = ShortId (8 bytes)
session_id[8..16].copy_from_slice(config.short_id);

// 用 AES-GCM(AuthKey) 加密前 16 字节，生成 32 字节密文覆盖整个 SessionId
let nonce = &client_hello_random[20..];  // 12 bytes
let aad = &client_hello_payload;
let ciphertext = aes_gcm_encrypt(auth_key, nonce, &session_id[..16], aad);
session_id.copy_from_slice(&ciphertext); // 16 bytes plaintext + 16 bytes tag = 32 bytes
```

### 6.4 AuthKey 计算

```rust
// 1. 从 utls KeyShare 中获取客户端 ephemeral X25519 私钥对应的公钥
// 2. 计算 X25519 共享密钥
let shared_secret = x25519(client_private_key, server_public_key);

// 3. HKDF-SHA256 派生
let auth_key = hkdf_sha256(
    salt: &client_hello_random[..20],
    ikm: &shared_secret,
    info: b"REALITY",
    length: 32
);
```

### 6.5 服务器证书验证 (VerifyPeerCertificate)

在标准 X.509 验证之前，注入自定义验证：

```rust
fn verify_peer_certificate(raw_certs: &[&[u8]], _chain: &mut [Certificate]) -> Result<(), Error> {
    // 1. 解析服务器证书，提取公钥（必须是 Ed25519）
    let cert = parse_certificate(raw_certs[0])?;
    let server_pubkey = cert.public_key_ed25519()?;
    
    // 2. 计算 HMAC-SHA512(AuthKey, pubKey)
    let expected_signature = hmac_sha512(auth_key, server_pubkey);
    
    // 3. 与证书中的 Signature 字段比对
    if !constant_time_eq(&expected_signature[..], &cert.signature[..]) {
        return Err("certificate signature mismatch");
    }
    
    // 4. (可选) 若配置了 ML-DSA-65，进一步验证额外签名
    if let Some(mldsa65_pubkey) = config.mldsa65_verify {
        verify_mldsa65(mldsa65_pubkey, auth_key, client_hello, server_hello)?;
    }
    
    Ok(())
}
```

### 6.6 未验证时的 Fallback (Spider)

如果 `Verified == false`，说明连接到了真实目标（或遭遇 MITM）：
- 启动 HTTP/2 "Spider" 模拟正常浏览器行为
- 访问随机路径，递归爬取 HTML 中的链接
- 参数 `SpiderX` 为初始路径，`SpiderY` 数组控制并发、Cookie padding、请求间隔等
- 最后返回错误 `REALITY: processed invalid connection`

> **实现建议**：Spider fallback 机制复杂度高，MVP 阶段可先实现为直接返回错误并断开连接。

---

## 7. 关键技术难点与 Rust 实现策略

### 7.1 TLS 指纹伪装 (uTLS 等效实现)

**挑战**：Rust 生态中无直接的 `uTLS` 等效库。`rustls` 的 ClientHello 是标准库风格，易被识别。

**方案**：
1. **craftls** (`github.com/3andne/craftls`)：Rustls 的分支，支持 customizable ClientHello fingerprint，预定义了 Chrome/Safari/Firefox 指纹
2. **自定义 TLS 栈**：基于 `ring` / `aws-lc-rs` 从头构建 TLS 1.3 客户端，完全控制 ClientHello 的每个字节（工作量极大，不推荐）
3. **参考 shoes 的实现**：`cfal/shoes` 已用 Rust 实现了 REALITY 和 Vision，其 TLS 层实现可作为核心参考

**推荐**：优先调研 `craftls` 是否能满足 REALITY 对 ClientHello 的精细控制需求（特别是 SessionId 注入、特定扩展顺序、GREASE 等）。若不行，研究 `shoes` 的实现方式。

### 7.2 REALITY 的底层 TLS 控制

**挑战**：REALITY 需要在 TLS 握手前修改 ClientHello 的 SessionId，并在握手后验证证书签名，这要求对 TLS 客户端有深度控制。

**方案**：
- `craftls` 若提供足够的 ClientHello 定制能力（pre-handshake hook），可满足需求
- 否则需要使用更低层的 TLS 库，或参考 `shoes` 的实现（可能基于 `tokio-rustls` 的 fork 或自定义 TLS 状态机）

### 7.3 XTLS Vision 的 DirectCopy

**挑战**：Go 中用 `unsafe` 访问 `crypto/tls.Conn` 内部缓冲区。Rust 中无此能力。

**方案**：
1. **保守方案**：收到 DirectCopy 信号后，直接关闭 `rustls` reader，后续从底层 TCP stream 读取。由于 TLS 1.3 记录层在此阶段已进入 Application Data，且服务器端也在直接透传，只要同步正确，数据不会丢失
2. **研究 rustls 内部 API**：查看是否可通过非公开 API 获取其内部缓冲区中的未处理数据
3. **参考 shoes**：其 Vision 实现可能采用了不同的 DirectCopy 策略

### 7.4 异步运行时选择

**推荐**：Tokio
- 成熟的 async TCP/UDP 网络编程支持
- 丰富的生态（tokio-rustls, tokio-util 等）
- 与 `shoes` 等参考项目一致

---

## 8. 推荐的 Rust 依赖库

| 功能 | 推荐库 | 说明 |
|------|--------|------|
| 异步运行时 | `tokio` | TCP/UDP 异步 IO |
| TLS (标准) | `tokio-rustls` / `rustls` | 若 craftls 不适用时的备选 |
| TLS (指纹) | `craftls` | rustls 分支，支持 fingerprint |
| 加密 | `ring` | AES-GCM, HKDF, HMAC, SHA256/512 |
| X25519 | `x25519-dalek` | 密钥交换 |
| Ed25519 | `ed25519-dalek` | REALITY 证书验证 |
| ML-DSA-65 | `pqcrypto-mldsa` | 后量子签名验证（可选） |
| UUID | `uuid` | VLESS UUID 解析和处理 |
| 配置解析 | `serde` + `toml`/`json` | 用户配置 |
| 日志 | `tracing` | 结构化日志 |
| 命令行 | `clap` | CLI 参数解析 |
| Protobuf | `prost` | VLESS Addons 编解码 |
| 字节操作 | `bytes` | 零拷贝缓冲区管理 |
| HTTP 解析 | `httparse` | HTTP CONNECT 请求解析 |

---

## 9. 实现阶段规划

### Phase 1: 基础架构 (Week 1-2)
- [ ] 项目脚手架（Cargo workspace 结构）
- [ ] 配置系统（TOML/JSON 格式，支持 VLESS + REALITY 参数）
- [ ] 日志系统 (`tracing`)
- [ ] 基本 TCP 连接管理和 Tokio runtime

### Phase 2: 入站代理 (Week 2-3)
- [ ] SOCKS5 无认证模式
- [ ] SOCKS5 用户名密码认证
- [ ] HTTP CONNECT 代理
- [ ] Mixed 模式自动协议识别
- [ ] SOCKS5 UDP ASSOCIATE (MVP 可先跳过 UDP)

### Phase 3: VLESS 协议核心 (Week 3-4)
- [ ] VLESS Request Header 编码
- [ ] VLESS Response Header 解码
- [ ] Address (IPv4/IPv6/Domain) 编解码
- [ ] 基础 TCP 数据转发（无 XTLS）
- [ ] UUID 处理（ProcessUUID 清零逻辑）

### Phase 4: TLS + REALITY (Week 4-6)
- [ ] 调研并集成 TLS 指纹库 (`craftls` 或参考 `shoes`)
- [ ] REALITY ClientHello SessionId 构造
- [ ] X25519 + HKDF AuthKey 计算
- [ ] AES-GCM 加密 SessionId
- [ ] 自定义证书验证 (HMAC-SHA512)
- [ ] REALITY 握手流程完整打通

### Phase 5: XTLS Vision (Week 6-8)
- [ ] TrafficState 状态机实现
- [ ] XtlsFilterTls TLS 识别引擎
- [ ] Padding / Unpadding 编解码
- [ ] VisionWriter (上行)
- [ ] VisionReader (下行)
- [ ] DirectCopy 切换逻辑

### Phase 6: UDP 和优化 (Week 8-10)
- [ ] VLESS UDP over TCP (Length-prefixed)
- [ ] SOCKS5 UDP ASSOCIATE 完整实现
- [ ] 性能优化 (zero-copy, buffer pool)
- [ ] 错误处理和连接重试
- [ ] 跨平台测试 (Linux/macOS/Windows)

---

## 10. 配置示例

```toml
# config.toml
[log]
level = "info"

[inbound]
listen = "127.0.0.1:1080"
protocol = "mixed"  # socks / http / mixed
# username = "user"
# password = "pass"

[[outbound]]
protocol = "vless"
address = "your.server.com"
port = 443
uuid = "66ad4540-b58c-4ad2-9926-ea63445a9b57"
flow = "xtls-rprx-vision"

[outbound.transport]
type = "reality"
server_name = "www.example.com"
public_key = "YOUR_BASE64URL_ENCODED_PUBLIC_KEY"
short_id = "0123456789abcdef"
fingerprint = "chrome"
# spider_x = "/"
```

---

## 11. 参考项目与资源

### 11.1 核心参考

| 项目 | 语言 | 说明 |
|------|------|------|
| [XTLS/Xray-core](https://github.com/XTLS/Xray-core) | Go | 官方实现，本方案的直接参考源 |
| [cfal/shoes](https://github.com/cfal/shoes) | Rust | **已用 Rust 实现 SOCKS5/HTTP Mixed + VLESS + REALITY + Vision**，最重要的参考项目 |
| [3andne/craftls](https://github.com/3andne/craftls) | Rust | rustls 分支，支持 customizable ClientHello fingerprint |
| [XTLS/reality](https://github.com/XTLS/reality) | Go | REALITY 底层 Go 实现（被 Xray-core 依赖） |

### 11.2 关键文件速查

Xray-core 源码中与本方案直接相关的文件：

| 文件 | 作用 |
|------|------|
| `proxy/socks/server.go` | Mixed 入站、协议识别 |
| `proxy/socks/protocol.go` | SOCKS5 握手全流程 |
| `proxy/http/server.go` | HTTP CONNECT 入站 |
| `proxy/vless/encoding/encoding.go` | VLESS 请求/响应头编解码 |
| `proxy/vless/encoding/addons.go` | Addons (flow) 编解码 |
| `proxy/vless/outbound/outbound.go` | VLESS 出站主流程 |
| `proxy/proxy.go` | **XTLS Vision 核心**：VisionReader/VisionWriter/TrafficState/XtlsPadding/XtlsFilterTls |
| `transport/internet/reality/reality.go` | REALITY 客户端封装 |
| `transport/internet/reality/config.go` | REALITY 配置转换 |
| `transport/internet/tcp/dialer.go` | TCP 出站连接建立流程 |
| `common/protocol/address.go` | Address/Port 编解码 |
| `common/protocol/headers.go` | RequestHeader 结构定义 |

### 11.3 推荐阅读

1. [Xray 官方文档 - VLESS](https://xtls.github.io/en/config/outbounds/vless.html)
2. [Xray 官方文档 - REALITY](https://xtls.github.io/en/config/transports/reality.html)
3. [Xray 官方文档 - XTLS Vision](https://xtls.github.io/en/config/outbounds/vless.html#xtls-rprx-vision)
4. REALITY 设计白皮书（若有）

---

## 12. 风险与注意事项

1. **TLS 指纹库成熟度**：`craftls` 是 rustls 的分支，与上游同步可能滞后。需评估其维护状态和对 TLS 1.3 最新特性的支持
2. **REALITY 复杂度**：REALITY 涉及底层 TLS 握手操控，实现难度大。强烈推荐参考 `shoes` 的实现
3. **性能**：Vision 的 DirectCopy 在 Rust 中可能无法做到 Go 的 `splice` 零拷贝水平，需做好性能测试
4. **协议兼容性**：VLESS 协议和 REALITY 在持续演进，需跟踪 Xray-core 的更新
5. **安全性**：涉及敏感加密材料（X25519 私钥、UUID），需谨慎处理内存安全（避免密钥残留）

---

*本方案基于对 XTLS/Xray-core 源码的深入分析，以及 Rust 生态调研结果编写。建议在正式编码前，先对 `cfal/shoes` 的源码进行精读，以其架构为基础进行裁剪和改造。*
