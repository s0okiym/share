# Xray-core Rust 重写实现方案

## 概述

本文档规划了将 Xray-core 部分功能用 Rust 重写的实现方案。目标不是完整复刻，而是实现以下四个核心组件：

| 组件 | 描述 |
|------|------|
| HTTP Inbound | HTTP CONNECT 代理入口 |
| SOCKS5 Inbound | SOCKS5 代理入口（TCP + UDP Associate） |
| VLESS + REALITY + Vision Outbound | VLESS 协议出口，REALITY TLS 伪装 + XTLS Vision 流量整形 |
| VLESS + XHTTP + REALITY Outbound | VLESS 协议出口，XHTTP 分块传输 + REALITY TLS |

对应的 Go 源码关键文件：
- HTTP Inbound: `proxy/http/server.go`
- SOCKS5 Inbound: `proxy/socks/server.go`, `proxy/socks/protocol.go`
- VLESS Outbound: `proxy/vless/outbound/outbound.go`
- VLESS 编码: `proxy/vless/encoding/encoding.go`, `proxy/vless/encoding/addons.go`
- REALITY: `transport/internet/reality/reality.go`
- XHTTP (SplitHTTP): `transport/internet/splithttp/` 整个目录
- Vision: `proxy/proxy.go` (VisionReader/VisionWriter)

---

## 1. 项目结构

### Cargo Workspace

```
xray-rs/
├── Cargo.toml                  # workspace 根
├── xray-core/                  # 主二进制 crate
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # CLI 入口、配置加载、启动
│       └── config.rs           # 配置结构体（serde + JSON 解析）
├── xray-inbounds/              # 入口协议 crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── http.rs             # HTTP CONNECT 代理
│       └── socks5.rs           # SOCKS5 代理（TCP + UDP）
├── xray-vless/                 # VLESS 协议 crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── header.rs           # VLESS 请求/响应头编码解码
│       ├── addons.rs           # VLESS header addons（protobuf）
│       ├── encryption.rs       # ML-KEM-768 加密层（可选，Phase 3+）
│       └── outbound.rs         # VLESS 出站处理器
├── xray-reality/               # REALITY TLS crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── client.rs           # REALITY 客户端（UClient 的 Rust 实现）
│       ├── session_id.rs       # SessionId 编码/加密
│       ├── verify.rs           # VerifyPeerCertificate 逻辑
│       └── spider.rs           # MITM 兜底 Spider 行为
├── xray-vision/                # XTLS Vision crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── traffic_state.rs    # 流量状态追踪
│       ├── reader.rs           # VisionReader
│       ├── writer.rs           # VisionWriter
│       └── filter.rs           # XtlsFilterTls（TLS 1.3 检测）
├── xray-xhttp/                 # XHTTP (SplitHTTP) 传输 crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── client.rs           # HTTP 客户端（h1/h2/h3）
│       ├── session.rs          # 会话管理
│       ├── upload_queue.rs     # 乱序包重排序
│       ├── connection.rs       # splitConn 封装
│       └── xmux.rs             # 连接多路复用
└── xray-common/                # 共享工具 crate
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── buf.rs              # 8KB 缓冲池
        ├── address.rs          # 地址解析（IPv4/IPv6/Domain + Port）
        └── pipe.rs             # 异步管道（uplink/downlink）
```

### 关键依赖选择

| 用途 | Crate | 理由 |
|------|-------|------|
| 异步运行时 | `tokio` (full) | 生态最成熟 |
| HTTP 客户端 | `hyper` + `hyper-util` | HTTP/1.1 和 HTTP/2 支持 |
| HTTP/3 | `h3` + `h3-quinn` | QUIC-based HTTP/3 |
| TLS | `rustls` + `tokio-rustls` | 纯 Rust，安全审计 |
| TLS 指纹 | `rustls` custom ClientHello | 需定制 ClientHello 扩展 |
| 序列化 | `serde` + `serde_json` | 配置解析 |
| Protobuf | `prost` | VLESS addons 需要 |
| UUID | `uuid` | VLESS UUID 解析 |
| X25519 | `x25519-dalek` | ECDH 密钥交换 |
| AES-GCM | `aes-gcm` (via `aes` + `aes-gcm`) | REALITY SessionId 加密 |
| HKDF | `hkdf` + `sha2` | REALITY 密钥派生 |
| Ed25519 | `ed25519-dalek` | REALITY 证书验证 |
| ML-DSA-65 | `ml-dsa` (或 `circl` FFI) | 后量子验证（可选） |
| 零化敏感数据 | `zeroize` | 密钥安全清理 |
| 日志 | `tracing` + `tracing-subscriber` | 结构化日志 |
| 配置 | `toml` | 可选 TOML 配置格式 |

### Cargo feature flags

```toml
[features]
default = ["http-inbound", "socks5-inbound", "vless-outbound", "reality", "vision", "xhttp"]
http-inbound = []
socks5-inbound = []
vless-outbound = []
reality = ["dep:x25519-dalek", "dep:aes-gcm", "dep:hkdf", "dep:sha2"]
vision = []
xhttp = ["dep:hyper", "dep:hyper-util", "dep:h3", "dep:h3-quinn"]
```

---

## 2. 核心抽象

### 2.1 异步读写抽象

所有内部数据流使用 tokio 的 `AsyncRead` + `AsyncWrite` trait，通过 pipe 连接。

```rust
/// 内部数据流：uplink（客户端→服务器）和 downlink（服务器→客户端）
pub struct Link {
    pub uplink: (mpsc::Sender<Bytes>, mpsc::Receiver<Bytes>),    // 上行管道
    pub downlink: (mpsc::Sender<Bytes>, mpsc::Receiver<Bytes>),  // 下行管道
}

/// 或者使用自定义的 pipe（类似 Go 的 transport/pipe）
pub struct Pipe {
    reader: AsyncReader,
    writer: AsyncWriter,
}
```

### 2.2 缓冲池

8KB 缓冲池，减少分配开销：

```rust
use tokio::sync::Mutex;

pub const BUF_SIZE: usize = 8192;

pub struct BufferPool {
    pool: Mutex<Vec<Box<[u8; BUF_SIZE]>>>,
}

impl BufferPool {
    pub fn acquire(&self) -> Box<[u8; BUF_SIZE]> { ... }
    pub fn release(&self, buf: Box<[u8; BUF_SIZE]>) { ... }
}
```

### 2.3 地址表示

```rust
#[derive(Clone, Debug)]
pub enum Address {
    Ipv4(Ipv4Addr),
    Ipv6(Ipv6Addr),
    Domain(String),
}

#[derive(Clone, Debug)]
pub struct Destination {
    pub address: Address,
    pub port: u16,
}

/// VLESS 地址类型编码（用于线格式）
impl Address {
    pub fn wire_type(&self) -> u8 {
        match self {
            Address::Ipv4(_) => 0x01,
            Address::Domain(_) => 0x02,
            Address::Ipv6(_) => 0x03,
        }
    }
}
```

---

## 3. 各组件实现细节

### 3.1 HTTP Inbound (`xray-inbounds/src/http.rs`)

**Go 参考**: `proxy/http/server.go`

**流程**:
1. 监听 TCP 端口，接受连接
2. 使用 `httparse` crate 解析 HTTP 请求（或使用 `hyper` 的 server 端）
3. 解析 `CONNECT` 方法 → 回复 `HTTP/1.1 200 Connection established\r\n\r\n`
4. 将客户端连接与 outbound 链路双向转发

```rust
pub async fn handle_connection(
    mut stream: TcpStream,
    config: &HttpConfig,
    dispatcher: &Dispatcher,
) -> Result<(), Error> {
    // 1. 解析 HTTP 请求
    let request = read_http_request(&mut stream).await?;

    // 2. 认证检查（如果配置了 accounts）
    if !config.accounts.is_empty() {
        if let Some(auth) = request.headers.get("Proxy-Authorization") {
            if !config.verify_basic_auth(auth) {
                stream.write_all(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n").await?;
                return Err(Error::AuthFailed);
            }
        }
    }

    // 3. 解析目标地址
    let dest = parse_host(&request.host)?;

    // 4. 处理 CONNECT
    if request.method == "CONNECT" {
        stream.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n").await?;
        // 5. 创建 pipe 并双向转发
        let link = dispatcher.dispatch(dest).await?;
        tokio::spawn(bidirectional_copy(stream, link));
    }
}
```

**关键点**:
- 使用 `httparse` 做低层级 HTTP 解析（比 hyper server 更灵活，可以拿到原始 stream）
- CONNECT 成功后，直接做 TCP 双向转发，不再解析 HTTP 内容
- Plain HTTP 模式（透明代理）在初始版本可跳过，只实现 CONNECT

### 3.2 SOCKS5 Inbound (`xray-inbounds/src/socks5.rs`)

**Go 参考**: `proxy/socks/server.go`, `proxy/socks/protocol.go`

**TCP 握手流程**:
```
Client → Server: 0x05 NMETHODS METHODS...
Server → Client: 0x05 METHOD
[如果 METHOD == 0x02] 用户名密码认证
Client → Server: 0x05 CMD 0x00 ATYPE ADDR PORT
Server → Client: 0x05 STATUS 0x00 ATYPE ADDR PORT
```

**协议常量**:
```rust
pub const SOCKS5_VERSION: u8 = 0x05;
pub const AUTH_NONE: u8 = 0x00;
pub const AUTH_PASSWORD: u8 = 0x02;
pub const CMD_CONNECT: u8 = 0x01;
pub const CMD_UDP_ASSOCIATE: u8 = 0x03;
pub const ATYPE_IPV4: u8 = 0x01;
pub const ATYPE_DOMAIN: u8 = 0x03;
pub const ATYPE_IPV6: u8 = 0x04;
pub const STATUS_SUCCEEDED: u8 = 0x00;
```

**UDP Associate**:
- TCP 控制连接协商出 UDP 中继地址
- UDP 包格式: `[RESERVED:2][FRAG:1][ATYPE:1][ADDR:var][PORT:2][PAYLOAD]`
- 仅处理 FRAG=0 的包（丢弃分片包）
- Cone NAT 模式：第一个目标地址作为后续所有包的默认目标

**UDP 过滤**:
- 简单 IP 白名单：TCP 认证通过的 IP 才能发 UDP
- 用 `HashSet<SocketAddr>` 实现

```rust
pub struct Socks5Server {
    config: Socks5Config,
    udp_filter: Arc<Mutex<HashSet<IpAddr>>>,
}

impl Socks5Server {
    pub async fn handle_tcp(&self, mut stream: TcpStream) -> Result<(), Error> {
        // 1. 读取方法协商
        let methods = read_methods(&mut stream).await?;
        let auth_method = self.select_auth_method(&methods)?;
        stream.write_all(&[SOCKS5_VERSION, auth_method]).await?;

        // 2. 如果需要密码认证
        if auth_method == AUTH_PASSWORD {
            self.authenticate(&mut stream).await?;
        }

        // 3. 读取命令
        let (cmd, dest) = read_command(&mut stream).await?;

        match cmd {
            CMD_CONNECT => {
                // 回复成功，然后双向转发
                self.reply_success(&mut stream, &dest).await?;
                let link = self.dispatcher.dispatch(dest).await?;
                tokio::spawn(bidirectional_copy(stream, link));
            }
            CMD_UDP_ASSOCIATE => {
                // 回复 UDP 中继地址
                let udp_addr = self.udp_relay_addr(&stream);
                self.reply_udp_assoc(&mut stream, udp_addr).await?;
                // TCP 连接保持开放直到客户端断开
                stream.read_to_end(&mut vec![]).await.ok();
            }
        }
    }
}
```

### 3.3 VLESS 协议层 (`xray-vless/`)

**Go 参考**: `proxy/vless/encoding/encoding.go`, `proxy/vless/encoding/addons.go`

#### 请求头格式

```
[VERSION:1][UUID:16][ADDONS_LEN:1][ADDONS:N][CMD:1][ADDR:var]
```

| 字段 | 大小 | 说明 |
|------|------|------|
| Version | 1 byte | 始终 0x00 |
| UUID | 16 bytes | 用户 ID |
| AddonsLen | 1 byte | 0（普通）或 >0（Vision） |
| Addons | N bytes | Protobuf Addons 消息 |
| Command | 1 byte | TCP=1, UDP=2, MUX=3, RVS=4 |
| Address+Port | 可变 | Port(u16 BE) + ATYPE(1) + ADDR(var) |

#### 响应头格式

```
[VERSION:1][ADDONS_LEN:1][ADDONS:N]
```

#### 关键实现

```rust
// xray-vless/src/header.rs

pub const VLESS_VERSION: u8 = 0x00;
pub const CMD_TCP: u8 = 0x01;
pub const CMD_UDP: u8 = 0x02;
pub const CMD_MUX: u8 = 0x03;
pub const CMD_RVS: u8 = 0x04;

pub fn encode_request_header(
    buf: &mut Vec<u8>,
    uuid: &[u8; 16],
    addons: &[u8],  // 已序列化的 protobuf Addons
    cmd: u8,
    dest: &Destination,
) {
    buf.push(VLESS_VERSION);
    buf.extend_from_slice(uuid);
    buf.push(addons.len() as u8);
    buf.extend_from_slice(addons);
    buf.push(cmd);
    // Port first, then address (PortThenAddress 格式)
    buf.extend_from_slice(&dest.port.to_be_bytes());
    match &dest.address {
        Address::Ipv4(ip) => {
            buf.push(0x01);
            buf.extend_from_slice(&ip.octets());
        }
        Address::Domain(d) => {
            buf.push(0x02);
            buf.push(d.len() as u8);
            buf.extend_from_slice(d.as_bytes());
        }
        Address::Ipv6(ip) => {
            buf.push(0x03);
            buf.extend_from_slice(&ip.octets());
        }
    }
}

pub fn encode_response_header(buf: &mut Vec<u8>, addons: &[u8]) {
    buf.push(VLESS_VERSION);
    buf.push(addons.len() as u8);
    buf.extend_from_slice(addons);
}
```

#### VLESS Addons (Protobuf)

```protobuf
// xray-vless/src/addons.proto
message Addons {
  string Flow = 1;  // "xtls-rprx-vision" for Vision
  bytes Seed = 2;
}
```

使用 `prost` 生成 Rust 代码。

#### UDP 包长度前缀

非 Vision 模式下，UDP 数据流中每个包前面加 2 字节 big-endian 长度：

```rust
// xray-vless/src/header.rs

pub struct LengthPacketWriter<W> {
    inner: W,
}

impl<W: AsyncWrite + Unpin> LengthPacketWriter<W> {
    pub async fn write_packet(&mut self, data: &[u8]) -> io::Result<()> {
        let len = data.len() as u16;
        self.inner.write_all(&len.to_be_bytes()).await?;
        self.inner.write_all(data).await?;
        self.inner.flush().await
    }
}
```

### 3.4 REALITY 传输 (`xray-reality/`)

**Go 参考**: `transport/internet/reality/reality.go`

**这是最复杂的部分。** REALITY 客户端的核心逻辑：

#### 流程

1. **使用 uTLS 风格的 ClientHello 指纹**（Chrome, Firefox 等）
2. **在 TLS SessionId 中嵌入认证数据**:
   ```
   SessionId[0:3]   = Xray 版本号 (X.Y.Z)
   SessionId[3]     = 0 (reserved)
   SessionId[4:8]   = Unix 时间戳 (u32 BE)
   SessionId[8:16]  = ShortId (8 bytes)
   SessionId[0:16]  = AES-GCM 加密（用 AuthKey）
   ```
3. **ECDH 密钥交换**: 用服务器的 X25519 公钥计算共享密钥
4. **HKDF 派生 AuthKey**: `HKDF-SHA256(shared_key, salt=hello.Random[:20], info="REALITY")`
5. **AES-GCM 加密 SessionId 前 16 字节**: `AES-GCM(AuthKey, nonce=hello.Random[20:], plaintext=SessionId[0:16], aad=hello.Raw)`
6. **验证服务器证书**: 检查是否为 REALITY 服务器自签名的 Ed25519 证书
7. **如果验证失败（MITM）**: 启动 Spider 行为，模拟正常浏览器访问目标网站

#### Rust 实现策略

**方案选择**: 使用 `rustls` 但需要深度定制 ClientHello 和证书验证。

`rustls` 支持通过 `ClientConfig::dangerous()` + `CustomCertVerifier` 自定义证书验证。但 `rustls` 的 ClientHello 构建是自动的，不暴露原始的 `ClientHello` 结构体。

**推荐方案**: 使用 `rustls` + `tokio-rustls` 作为基础 TLS，但需要：

1. **定制 ClientHello**: 使用 `rustls::crypto::CryptoProvider` 和 `rustls::client::ClientConfig` 的 `enable_sni` 和 `alpn_protocols` 设置。对于 uTLS 指纹，需要定制扩展顺序和内容。

2. **替代方案 — 使用 `boring` crate**: `boring` 是 BoringSSL 的 Rust 绑定，支持完整的 ClientHello 定制（包括 uTLS 风格的指纹）。这是最接近 Go utls 的方案。

3. **最实用方案 — 混合使用**:
   - 使用 `tokio-rustls` 做基础 TLS 连接
   - 在 TLS 握手前，手动构造 ClientHello 消息并发送
   - 然后用 `rustls` 的 `ServerCertVerified` 做证书验证
   - 这需要深入到 `rustls` 的内部，比较 hacky

**对于本项目，推荐方案**: 使用 `rustls` + 手动 ClientHello 构造。原因：
- 纯 Rust，无 C 依赖
- ClientHello 本质上是一个结构化的字节流，可以手动构建
- 需要访问 `rustls::ClientConnection` 的底层 writer

```rust
// xray-reality/src/client.rs

use x25519_dalek::{EphemeralSecret, PublicKey};
use hkdf::Hkdf;
use sha2::Sha256;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

pub struct RealityClient {
    config: RealityConfig,
    server_name: String,
    auth_key: Vec<u8>,
    verified: bool,
}

impl RealityClient {
    pub fn new(config: RealityConfig) -> Self { ... }

    pub async fn connect(
        &mut self,
        stream: TcpStream,
    ) -> Result<RealityConn, Error> {
        // 1. 生成 ECDH 密钥对
        let secret = EphemeralSecret::random();
        let public = PublicKey::from(&secret);

        // 2. 用服务器公钥计算共享密钥
        let server_public = PublicKey::from(self.config.public_key);
        let shared = secret.diffie_hellman(&server_public);

        // 3. HKDF 派生 AuthKey
        let hk = Hkdf::<Sha256>::new(Some(&hello_random[..20]), shared.as_bytes());
        let mut auth_key = [0u8; 32];
        hk.expand(b"REALITY", &mut auth_key)?;
        self.auth_key = auth_key.to_vec();

        // 4. 构造 ClientHello with embedded SessionId
        let client_hello = self.build_client_hello(&auth_key)?;

        // 5. 发送 ClientHello，接收 ServerHello + Certificate + Finished
        let (conn, handshake_state) = self.do_handshake(stream, client_hello).await?;

        // 6. 验证服务器证书
        self.verify_certificate(&handshake_state)?;

        Ok(RealityConn { conn, verified: self.verified })
    }

    fn build_client_hello(&self, auth_key: &[u8; 32]) -> Result<Vec<u8>, Error> {
        // 手动构造 TLS ClientHello 消息
        // - 版本: TLS 1.2 (0x0303)
        // - Random: 32 bytes 随机
        // - Session ID: 32 bytes（嵌入认证数据）
        // - Cipher Suites: 根据指纹选择
        // - Extensions: SNI, supported_versions, key_share, etc.
        ...
    }

    fn verify_certificate(&mut self, handshake: &HandshakeState) -> Result<(), Error> {
        // 检查 Ed25519 公钥 + HMAC 签名
        // 如果通过，设置 self.verified = true
        // 如果失败，启动 Spider
        ...
    }
}
```

**关于 TLS 指纹**: 需要支持至少 `chrome` 和 `firefox` 指纹。每个指纹定义了：
- Cipher Suites 列表和顺序
- 扩展列表和顺序
- 压缩方法
- 椭圆曲线列表

这些信息可以硬编码为常量结构体。

### 3.5 XTLS Vision (`xray-vision/`)

**Go 参考**: `proxy/proxy.go` (VisionReader, VisionWriter, XtlsFilterTls)

#### 填充结构（21 字节开销）

```
[UUID:16][CMD:1][CONTENT_LEN:2][PADDING_LEN:2][CONTENT:var][PADDING:var]
```

| 字段 | 大小 | 说明 |
|------|------|------|
| UUID | 16 bytes | 用户 ID（首次发送，后续清零） |
| Command | 1 byte | 0=Continue, 1=End, 2=Direct |
| ContentLen | 2 bytes | 内容长度（big-endian u16） |
| PaddingLen | 2 bytes | 填充长度（big-endian u16） |
| Content | 变长 | 实际数据 |
| Padding | 变长 | 随机填充 |

#### 三种命令

```rust
pub const CMD_PADDING_CONTINUE: u8 = 0x00;  // 更多数据跟随
pub const CMD_PADDING_END: u8 = 0x01;       // 填充序列结束
pub const CMD_PADDING_DIRECT: u8 = 0x02;    // 切换到直接复制（splice 模式）
```

#### TrafficState

```rust
// xray-vision/src/traffic_state.rs

pub struct TrafficState {
    pub user_uuid: Vec<u8>,
    pub packets_to_filter: i32,     // 剩余需要分类的包数（初始 ~8）
    pub enable_xtls: bool,          // 是否检测到 TLS 1.3 + 非 CCM-8
    pub is_tls12_or_above: bool,
    pub is_tls: bool,
    pub cipher: u16,
    pub within_padding_buffers: bool,
    pub direct_fragmented: bool,
    pub inbound_direct_copy: bool,
    pub outbound_direct_copy: bool,
}
```

#### VisionWriter (上行填充)

```rust
// xray-vision/src/writer.rs

pub struct VisionWriter<W> {
    inner: W,
    state: Arc<Mutex<TrafficState>>,
    uuid_sent: bool,
    is_padding: bool,
    switch_to_direct: bool,
}

impl<W: AsyncWrite + Unpin> VisionWriter<W> {
    pub async fn write(&mut self, data: &[u8]) -> io::Result<()> {
        let mut state = self.state.lock().await;

        // 阶段 1: TLS 分类（前 ~8 个包）
        if state.packets_to_filter > 0 {
            self.filter_tls(&mut state, data).await?;
            state.packets_to_filter -= 1;
        }

        // 阶段 2: 填充阶段
        if state.is_tls12_or_above && !state.enable_xtls {
            // 非 XTLS 模式：发送 End 命令
            self.write_padded(&mut state, data, CMD_PADDING_END).await?;
            self.is_padding = false;
        } else if state.enable_xtls {
            // XTLS 模式：发送 Direct 命令，切换到直接复制
            self.write_padded(&mut state, data, CMD_PADDING_DIRECT).await?;
            state.outbound_direct_copy = true;
            self.switch_to_direct = true;
        }

        // 阶段 3: 直接复制（splice 模式）
        if self.switch_to_direct {
            self.inner.write_all(data).await?;
        }

        Ok(())
    }

    async fn filter_tls(&mut self, state: &mut TrafficState, data: &[u8]) -> io::Result<()> {
        // 检查是否是 TLS ClientHello 或 ServerHello
        if data.len() >= 3 && data[0] == 0x16 && data[1] == 0x03 {
            if data[2] == 0x03 {
                // ServerHello: 检查 TLS 1.3
                if data.len() >= 5 && data[3] == 0x00 && data[4] == 0x02 {
                    // 查找 supported_versions 扩展 (0x002b)
                    if self.is_tls13(data) {
                        state.enable_xtls = true;
                        state.is_tls = true;
                    }
                }
            } else if data[2] == 0x01 {
                // ClientHello
                state.is_tls = true;
            }
        }
    }

    fn is_tls13(&self, data: &[u8]) -> bool {
        // 查找 TLS 1.3 supported_versions 扩展
        // 扩展类型: 0x002b, 长度: 0x0002, 版本: 0x0304
        let pattern = [0x00, 0x2b, 0x00, 0x02, 0x03, 0x04];
        // 在 ServerHello 扩展区域搜索
        ...
    }
}
```

#### VisionReader (下行解填充)

```rust
// xray-vision/src/reader.rs

pub struct VisionReader<R> {
    inner: R,
    state: Arc<Mutex<TrafficState>>,
    buffer: Vec<u8>,  // 累积缓冲区
}

impl<R: AsyncRead + Unpin> VisionReader<R> {
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // 1. 从底层读取数据
        let n = self.inner.read(&mut self.buffer).await?;

        let mut state = self.state.lock().await;

        // 2. TLS 分类（与 VisionWriter 对称）
        if state.packets_to_filter > 0 {
            self.filter_tls(&mut state, &self.buffer[..n])?;
            state.packets_to_filter -= 1;
        }

        // 3. 解填充
        if state.within_padding_buffers {
            return self.unpad(buf, &self.buffer[..n], &mut state);
        }

        // 4. 直接复制模式
        if state.inbound_direct_copy {
            // 直接返回原始数据
            let copy_len = n.min(buf.len());
            buf[..copy_len].copy_from_slice(&self.buffer[..copy_len]);
            return Ok(copy_len);
        }

        // 默认：直接返回
        let copy_len = n.min(buf.len());
        buf[..copy_len].copy_from_slice(&self.buffer[..copy_len]);
        Ok(copy_len)
    }

    fn unpad(&mut self, out: &mut [u8], data: &[u8], state: &mut TrafficState) -> io::Result<usize> {
        // 解析 21 字节头部: [UUID:16][CMD:1][CONTENT_LEN:2][PADDING_LEN:2]
        if data.len() < 21 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "incomplete vision header"));
        }

        let cmd = data[16];
        let content_len = u16::from_be_bytes([data[17], data[18]]) as usize;
        let padding_len = u16::from_be_bytes([data[19], data[20]]) as usize;

        // 提取内容
        let content_start = 21;
        let content_end = content_start + content_len;
        let copy_len = content_len.min(out.len());
        out[..copy_len].copy_from_slice(&data[content_start..content_start + copy_len]);

        // 处理命令
        match cmd {
            CMD_PADDING_DIRECT => {
                state.inbound_direct_copy = true;
                state.within_padding_buffers = false;
            }
            CMD_PADDING_END => {
                state.within_padding_buffers = false;
            }
            _ => {} // CONTINUE: 保持填充模式
        }

        Ok(copy_len)
    }
}
```

### 3.6 XHTTP 传输 (`xray-xhttp/`)

**Go 参考**: `transport/internet/splithttp/` 整个目录

#### 三种模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `stream-one` | 单个双向 HTTP 请求 | REALITY 模式默认 |
| `stream-up` | 上行 POST + 下行 GET 分离 | REALITY + DownloadSettings |
| `packet-up` | 多个 POST 分块 + 长轮询 GET | 默认模式，CDN 友好 |

#### HTTP 版本选择

| 条件 | HTTP 版本 |
|------|-----------|
| 有 REALITY | HTTP/2 |
| 无 TLS | HTTP/1.1 |
| TLS + h3 | HTTP/3 (QUIC) |
| 其他 | HTTP/2 |

#### 核心实现

```rust
// xray-xhttp/src/lib.rs

pub enum XHttpMode {
    StreamOne,
    StreamUp,
    PacketUp,
}

pub struct XHttpClient {
    mode: XHttpMode,
    http_version: HttpVersion,
    base_url: String,
    session_id: String,
    seq: AtomicU64,
    upload_queue: UploadQueue,
}

impl XHttpClient {
    pub async fn dial(&self, dest: &Destination) -> Result<XHttpConn, Error> {
        match self.mode {
            XHttpMode::StreamOne => self.dial_stream_one().await,
            XHttpMode::StreamUp => self.dial_stream_up().await,
            XHttpMode::PacketUp => self.dial_packet_up().await,
        }
    }

    async fn dial_stream_one(&self) -> Result<XHttpConn, Error> {
        // 单个 HTTP POST 请求，body 是双向管道
        // 使用 hyper 创建流式 POST 请求
        let (tx, rx) = mpsc::channel::<Bytes>(32);
        let body = Body::wrap_stream(rx);
        let request = Request::builder()
            .method("POST")
            .uri(format!("{}/{}", self.base_url, self.session_id))
            .header("Content-Type", "application/octet-stream")
            .body(body)?;

        let response = self.client.request(request).await?;
        Ok(XHttpConn::new(tx, response.into_body()))
    }

    async fn dial_packet_up(&self) -> Result<XHttpConn, Error> {
        // 下行：长轮询 GET
        let (down_tx, down_rx) = mpsc::channel::<Bytes>(32);
        // 上行：多个 POST 请求
        let (up_tx, up_rx) = mpsc::channel::<Bytes>(32);

        // 启动 GET 长轮询
        tokio::spawn(self.poll_downlink(down_rx.clone()));
        // 启动 POST 发送器
        tokio::spawn(self.push_uplink(up_rx));

        Ok(XHttpConn::new(up_tx, Body::wrap_stream(down_rx)))
    }
}
```

#### Upload Queue (乱序包重排序)

```rust
// xray-xhttp/src/upload_queue.rs

use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub struct UploadQueue {
    heap: BinaryHeap<Reverse<u64>>,          // 按 seq 排序的堆
    data_map: HashMap<u64, Bytes>,           // seq -> data
    next_expected: u64,
    max_buffered: usize,
    ready_tx: mpsc::Sender<Bytes>,           // 排序后输出
}

impl UploadQueue {
    pub fn push(&mut self, seq: u64, data: Bytes) {
        if self.heap.len() >= self.max_buffered {
            // 丢弃最旧的或报错
            return;
        }
        self.heap.push(Reverse(seq));
        self.data_map.insert(seq, data);
        self.flush_ready();
    }

    fn flush_ready(&mut self) {
        while let Some(&Reverse(next)) = self.heap.peek() {
            if next == self.next_expected {
                self.heap.pop();
                if let Some(data) = self.data_map.remove(&next) {
                    self.ready_tx.try_send(data).ok();
                    self.next_expected += 1;
                }
            } else {
                break;
            }
        }
    }
}
```

---

## 4. 配置文件格式

使用 JSON 配置（与 Xray 原生格式兼容）：

```json
{
  "inbounds": [
    {
      "protocol": "http",
      "listen": "127.0.0.1",
      "port": 10809,
      "settings": {
        "accounts": { "user": "pass" },
        "allowTransparent": false
      }
    },
    {
      "protocol": "socks",
      "listen": "127.0.0.1",
      "port": 10808,
      "settings": {
        "auth": "password",
        "accounts": { "user": "pass" },
        "udp": true
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "vless",
      "settings": {
        "vnext": [{
          "address": "example.com",
          "port": 443,
          "users": [{
            "id": "uuid-here",
            "flow": "xtls-rprx-vision",
            "encryption": "none"
          }]
        }]
      },
      "streamSettings": {
        "network": "tcp",
        "security": "reality",
        "realitySettings": {
          "serverName": "cloudflare.com",
          "publicKey": "base64-public-key",
          "shortId": "hex-short-id",
          "fingerprint": "chrome",
          "spiderX": "/"
        }
      }
    },
    {
      "protocol": "vless",
      "settings": {
        "vnext": [{
          "address": "example.com",
          "port": 443,
          "users": [{
            "id": "uuid-here",
            "flow": "xtls-rprx-vision"
          }]
        }]
      },
      "streamSettings": {
        "network": "splithttp",
        "security": "reality",
        "splithttpSettings": {
          "mode": "auto",
          "host": "example.com",
          "path": "/path",
          "xmux": {
            "maxConcurrency": "16-32",
            "maxConnections": 0
          }
        },
        "realitySettings": {
          "serverName": "cloudflare.com",
          "publicKey": "base64-public-key",
          "shortId": "hex-short-id",
          "fingerprint": "chrome"
        }
      }
    }
  ]
}
```

---

## 5. 分阶段实施计划

### Phase 1: 基础框架 + HTTP Inbound (1-2 周)

**目标**: 项目骨架可编译运行，HTTP CONNECT 代理可用。

1. 创建 Cargo workspace，设置所有 crate
2. 实现 `xray-common`:
   - `buf.rs`: 8KB 缓冲池
   - `address.rs`: 地址解析和编码解码
   - `pipe.rs`: 基于 `tokio::sync::mpsc` 的异步管道
3. 实现 `xray-inbounds/http.rs`:
   - TCP 监听
   - HTTP 请求解析（用 `httparse`）
   - Basic 认证
   - CONNECT 方法处理
   - 双向数据转发
4. 实现 `xray-core/main.rs`:
   - JSON 配置加载
   - 启动 HTTP 入口
   - 日志系统
5. **验证**: 用 `curl -x http://127.0.0.1:10809 https://example.com` 测试

### Phase 2: SOCKS5 Inbound (1 周)

**目标**: SOCKS5 TCP 代理可用，可选 UDP Associate。

1. 实现 `xray-inbounds/socks5.rs`:
   - 方法协商 + 密码认证
   - TCP CONNECT 命令
   - 地址解析（IPv4/IPv6/Domain）
   - 双向数据转发
   - UDP Associate（简化版，仅 Cone 模式）
   - UDP 包编码解码
2. **验证**: 用 `curl --socks5 127.0.0.1:10808 https://example.com` 测试 TCP
3. **验证**: 用 `curl --socks5-hostname` 测试远程 DNS 解析

### Phase 3: VLESS Outbound (基础，无 Vision) (1-2 周)

**目标**: VLESS TCP 出站可用，无 REALITY、无 Vision。

1. 实现 `xray-vless/`:
   - `header.rs`: 请求/响应头编码解码
   - `addons.rs`: Protobuf Addons 消息（prost）
   - `outbound.rs`: VLESS 出站处理器
2. 集成基础 TLS 传输（`tokio-rustls`）
3. VLESS 握手 → TCP 数据转发
4. UDP over VLESS（长度前缀模式）
5. **验证**: 连接到一个已知的 VLESS 服务器（非 REALITY，非 Vision），测试 TCP 转发

### Phase 4: REALITY 传输 (2-3 周)

**目标**: REALITY 客户端可用，能连接到 REALITY 服务器。

1. 实现 `xray-reality/`:
   - `session_id.rs`: SessionId 编码和 AES-GCM 加密
   - `verify.rs`: Ed25519 证书验证
   - `client.rs`: REALITY 客户端
     - ECDH 密钥交换（x25519-dalek）
     - HKDF 派生（hkdf + sha2）
     - ClientHello 构造（TLS 指纹）
     - 证书验证 + Spider 兜底
2. TLS 指纹支持（至少 Chrome）
3. 集成到 VLESS outbound
4. **验证**: 连接到 REALITY 服务器，验证 TLS 指纹和目标网站证书
5. **难点**: `rustls` 的 ClientHello 不暴露原始字节，可能需要：
   - 方案 A: 直接使用 `rustls::Connection` 的 writer 手动写入
   - 方案 B: 使用 `boring` crate（BoringSSL 绑定）替代 `rustls`

### Phase 5: XTLS Vision (2-3 周)

**目标**: Vision 流量整形可用。

1. 实现 `xray-vision/`:
   - `traffic_state.rs`: 流量状态
   - `writer.rs`: VisionWriter（填充）
   - `reader.rs`: VisionReader（解填充）
   - `filter.rs`: XtlsFilterTls（TLS 1.3 检测）
2. 集成到 VLESS outbound
3. 与 REALITY 联调
4. **验证**: `VLESS + REALITY + Vision` 完整链路，测试 TCP 转发

### Phase 6: XHTTP 传输 (2-3 周)

**目标**: XHTTP 三种模式均可用。

1. 实现 `xray-xhttp/`:
   - `client.rs`: HTTP 客户端（h1/h2/h3）
   - `session.rs`: 会话管理
   - `upload_queue.rs`: 乱序包重排序
   - `connection.rs`: splitConn 封装
   - `xmux.rs`: 连接多路复用
2. 集成到 VLESS outbound
3. **验证**: `VLESS + XHTTP + REALITY` 完整链路

### Phase 7: 集成和优化 (1-2 周)

1. 配置系统完善（支持多入口、多出口）
2. 错误处理和日志完善
3. 性能优化（缓冲池、连接复用）
4. 文档和示例配置
5. **最终验证**: 完整的端到端测试

---

## 6. 技术难点与风险

### 6.1 REALITY 的 TLS 指纹

**风险**: `rustls` 不支持完全自定义 ClientHello 的字节级构造。

**缓解**:
- 先尝试用 `rustls` 的 `write_hs()` 方法手动写入 ClientHello
- 备选: 使用 `boring` crate（BoringSSL Rust 绑定），它支持完整的 ClientHello 控制
- 备选: 使用 FFI 绑定到 Go 的 utls 库（增加构建复杂度）

### 6.2 XTLS Vision 的 splice 优化

**风险**: Go 的 Vision 通过 `unsafe` 访问 TLS 内部 buffer 实现零拷贝，Rust 中 `rustls` 不提供这种访问。

**缓解**:
- Rust 版本可以不实现 splice 优化，用缓冲复制代替
- 功能上等价，只是性能略低
- 如果追求极致性能，可以后期用 `io_uring` 或 `splice` syscall 优化

### 6.3 HTTP/3 支持

**风险**: `h3` + `h3-quinn` 生态还不成熟，可能有兼容性问题。

**缓解**:
- 初始版本只支持 HTTP/1.1 和 HTTP/2
- HTTP/3 作为 Phase 6 的可选增强

### 6.4 ML-KEM-768 加密层

**风险**: VLESS 支持 ML-KEM-768 后量子加密，但 Rust 生态中 ML-KEM 实现较少。

**缓解**:
- 初始版本不支持 ML-KEM（`encryption: "none"`）
- 后期可用 `pqcrypto` crate 或 FFI 到 `liboqs`

---

## 7. 验证计划

### 7.1 单元测试

- VLESS 头编码解码：验证编码后的字节与 Go 版本一致
- SOCKS5 协议解析：验证握手流程
- REALITY SessionId 加密：验证与 Go 版本的 SessionId 一致
- Vision 填充解填充：验证 roundtrip 正确性

### 7.2 集成测试

- HTTP Inbound + 直接出站：测试 CONNECT 隧道
- SOCKS5 Inbound + 直接出站：测试 TCP 和 UDP
- VLESS Outbound + TLS（无 REALITY）：测试基础 VLESS 连接
- VLESS + REALITY + Vision：连接到 Go Xray REALITY 服务器
- VLESS + XHTTP + REALITY：测试 XHTTP 三种模式

### 7.3 互操作性测试

最重要的一点：**Rust 客户端必须能连接 Go Xray 服务器，Rust 服务器必须能被 Go Xray 客户端连接**。

- 用 Go Xray 作为服务器，Rust 作为客户端连接
- 用 Rust 作为服务器，Go Xray 作为客户端连接
- 验证数据完整性和延迟

---

## 8. 编译环境

**重要**: 在编译和运行 Rust 代码之前，必须先执行：

```bash
. "$HOME/.cargo/env"
```

Rust 版本: 1.95.0 (2026-04-14)

```bash
. "$HOME/.cargo/env" && cargo build --release
. "$HOME/.cargo/env" && cargo test
. "$HOME/.cargo/env" && cargo run -- -c config.json
```

---

## 9. 预期产出

最终产出一个名为 `xray-rs` 的二进制文件，支持：

```bash
xray-rs run -c config.json          # 启动代理
xray-rs run --dump                  # 打印解析后的配置
```

支持的最小配置示例：

```json
{
  "inbounds": [
    {
      "protocol": "socks",
      "listen": "127.0.0.1",
      "port": 10808,
      "settings": { "udp": false }
    }
  ],
  "outbounds": [
    {
      "protocol": "vless",
      "settings": {
        "vnext": [{
          "address": "example.com",
          "port": 443,
          "users": [{ "id": "uuid", "flow": "xtls-rprx-vision" }]
        }]
      },
      "streamSettings": {
        "network": "tcp",
        "security": "reality",
        "realitySettings": {
          "serverName": "www.cloudflare.com",
          "publicKey": "...",
          "shortId": "...",
          "fingerprint": "chrome"
        }
      }
    }
  ]
}
```
