# REALITY 协议 Rust 实现指南

## 目录

- "一、可行性评估"
- "二、技术选型与依赖生态"
- "三、架构映射：Go → Rust"
- "四、开发阶段划分"
- "五、开发难度评估"
- "六、测试计划"
- "七、风险与应对"
- "八、推荐实施路线图"

---

## 一、可行性评估

### 1.1 总体结论

"REALITY 协议从 Go 移植到 Rust 在技术上是完全可行的。REALITY 本质上是一个 TLS 1.3 协议的服务器端实现，叠加了一套基于 X25519 + AEAD 的隐藏认证机制。Rust 生态已有成熟的 TLS 库（rustls）、X25519 实现（x25519-dalek）、AES-GCM（aes-gcm）和后量子密码学支持（pqcrypto），这些构成了移植的基础。"

"但需要注意的是，REALITY 并非一个独立的协议，而是对 Go 标准库 crypto/tls 的深度定制。移植工作不只是翻译语法，而是需要理解 TLS 1.3 协议的每一个细节，并将其在 Rust 的类型系统和异步模型下重新实现。"

### 1.2 代码规模评估

```
Go 代码总量：约 15,741 行

按模块分解：
├── conn.go                          1,755 行  ← 记录层、连接管理（核心）
├── common.go                        1,808 行  ← 配置、常量、类型定义（核心）
├── handshake_messages.go            1,953 行  ← 握手消息序列化（核心）
├── handshake_server_tls13.go        1,258 行  ← TLS 1.3 服务端握手（REALITY 核心修改）
├── handshake_client.go              1,327 行  ← TLS 客户端握手（可裁剪）
├── handshake_client_tls13.go          910 行  ← TLS 1.3 客户端握手（可裁剪）
├── handshake_server.go              1,006 行  ← TLS 1.2 服务端握手（可裁剪）
├── tls.go                             838 行  ← 入口点、Listener、REALITY Server（核心）
├── cipher_suites.go                   704 行  ← 密码套件定义
├── ech.go                             669 行  ← 加密客户端 Hello
├── quic.go                            500 行  ← QUIC 集成（可选）
├── ticket.go                          429 行  ← 会话票据
├── hpke/hpye.go                       354 行  ← HPKE 实现
├── auth.go                            297 行  ← 签名验证
├── prf.go                             297 行  ← TLS 1.2 PRF
├── record_detect.go                   185 行  ← 记录探测（REALITY 特有）
├── generate_cert.go                   171 行  ← 证书生成工具
├── tls13/tls13.go                     179 行  ← TLS 1.3 密钥调度
├── defaults.go                        126 行  ← 默认配置
├── alert.go                           111 行  ← Alert 定义
├── key_agreement.go                   371 行  ← 密钥协商
├── common_string.go                   120 行  ← 自动生成的字符串方法
├── defaults_boring.go                  68 行  ← BoringSSL 默认值
├── defaults_fips140.go                 75 行  ← FIPS 140 默认值
├── cache.go                            43 行  ← 证书缓存
├── tls12/tls12.go                      67 行  ← TLS 1.2 PRF
├── fips140tls/fipstls.go               36 行  ← FIPS 模式
```

"其中 REALITY 特有的核心代码约 300-500 行（主要在 tls.go 的 Server 函数和 handshake_server_tls13.go 的修改部分），其余约 15,000 行是 TLS 1.2/1.3 协议的基础实现。"

### 1.3 移植策略选择

"面对约 15,000 行的 TLS 协议实现，有两条路径可选："

**路径 A：基于 rustls 构建 REALITY 层**

"rustls 是一个成熟的 Rust TLS 库，实现了 TLS 1.2 和 1.3。REALITY 的认证逻辑和握手修改可以作为 rustls 的上层封装实现。"

"优势：复用 rustls 的 TLS 1.3 协议实现，只需实现 REALITY 特有的认证、MirrorConn、记录探测、临时证书等组件。工作量约 2,000-3,000 行。"

"劣势：rustls 的 API 设计可能不完全匹配 REALITY 的需求。REALITY 需要精确控制握手消息的发送内容和时机（如使用目标网站的 ServerHello 原始字节、按目标长度发送 Post-Handshake 记录），这可能需要修改 rustls 或使用其低级 API。"

**路径 B：从零实现完整 TLS 1.3 + REALITY**

"参考 Go 源码，在 Rust 中从零实现 TLS 1.3 协议和 REALITY 层。"

"优势：完全控制协议的每一个细节，不受现有库的 API 限制。"

"劣势：工作量巨大（约 8,000-12,000 行），且 TLS 协议实现中的安全细节（如恒定时间比较、侧信道防护）需要极高的专业水平。"

"综合推荐路径 A（基于 rustls），原因如下：rustls 已经经过了严格的安全审计，其 TLS 1.3 实现是可靠的。REALITY 的核心创新在于认证机制和流量伪装，这些可以在 rustls 之上以合理的工程量实现。只有在 rustls 确实无法满足需求时，才考虑对 rustls 进行 fork 或贡献上游修改。"

---

## 二、技术选型与依赖生态

### 2.1 核心依赖选型

```toml
[dependencies]
# TLS 基础
rustls = "0.23"                        # TLS 1.2/1.3 实现
rustls-pemfile = "2"                   # PEM 文件解析

# 密码学原语
x25519-dalek = "2"                     # X25519 密钥交换
aes-gcm = "0.10"                       # AES-GCM AEAD
hkdf = "0.12"                          # HKDF 密钥派生
sha2 = "0.10"                          # SHA-256/SHA-512
hmac = "0.12"                          # HMAC

# 后量子密码学
pqcrypto-ml-dsa = "0.1"                # ML-DSA-65 签名
ml-kem = "0.2"                         # ML-KEM-768 密钥封装

# 网络和异步
tokio = { version = "1", features = ["full"] }  # 异步运行时
tokio-util = { version = "0.7", features = ["io"] }

# 工具和序列化
bytes = "1"                            # 字节缓冲区
thiserror = "2"                        # 错误类型
tracing = "0.1"                        # 日志

# 可选：uTLS 指纹模拟（记录探测用）
# Rust 生态暂无直接等价物，可能需要 C FFI 或自定义实现
```

### 2.2 依赖对照表

| Go 包 | Rust 替代 | 匹配度 | 备注 |
|-------|-----------|--------|------|
| crypto/tls | rustls | 高 | rustls 实现了 TLS 1.2/1.3，但 API 风格不同 |
| golang.org/x/crypto/curve25519 | x25519-dalek | 高 | 功能等价，API 略有差异 |
| crypto/aes + cipher.NewGCM | aes-gcm | 高 | 功能等价 |
| golang.org/x/crypto/hkdf | hkdf crate | 高 | 功能等价 |
| crypto/sha256 / crypto/sha512 | sha2 | 高 | 功能等价 |
| crypto/hmac | hmac crate | 高 | 功能等价 |
| crypto/ed25519 | ed25519-dalek | 高 | 功能等价 |
| crypto/mlkem | ml-kem crate | 高 | Go 1.24 标准库，Rust 有独立实现 |
| github.com/cloudflare/circl/mldsa | pqcrypto-ml-dsa | 中 | CIRCL 功能更全，但 pqcrypto 可用 |
| github.com/juju/ratelimit | governor / tokio-util | 中 | Rust 有多个限流库可选 |
| github.com/pires/go-proxyproto | proxy-protocol | 中 | Rust 生态有 PROXY 协议实现 |
| github.com/refraction-networking/utls | 无直接等价 | 低 | uTLS 是 Go 的 TLS 指纹模拟库，Rust 无直接替代 |
| net.Conn | tokio::io::AsyncRead + AsyncWrite | 中 | Go 的同步 I/O vs Rust 的异步 I/O 模型差异 |
| sync.Mutex / sync.WaitGroup | tokio::sync::Mutex / JoinSet | 中 | 同步 vs 异步模型差异 |
| context.Context | tokio::task / CancellationToken | 中 | 概念类似，API 不同 |

### 2.3 关键技术挑战

"uTLS 指纹模拟：REALITY 的记录探测机制使用 uTLS 库模拟 Chrome 等浏览器的 TLS 指纹去探测目标网站。Rust 生态目前没有直接等价物。解决方案：（1）使用 C FFI 调用 uTLS 的 Go 编译产物；（2）手动实现常见的 TLS 指纹 ClientHello 格式；（3）直接使用标准 TLS 握手（探测结果可能不够精确但基本可用）。"

"同步 vs 异步 I/O 模型：Go 使用同步 I/O + goroutine 调度，Rust 使用异步 I/O + Future。MirrorConn 的实现需要从根本上改变设计。Go 版本中 MirrorConn.Read() 通过 Unlock → Read → Lock → Write Target 的模式实现并发控制，这在 Rust 中需要重新设计为基于异步流（Stream）或通道（Channel）的模式。"

"内存安全和生命周期：Go 的 GC 允许自由地在结构体间共享引用，Rust 需要精确管理所有权和生命周期。Conn 结构体中的 rawInput、input、hand 等缓冲区在 Go 中可以自由借用，在 Rust 中需要明确所有权转移或使用引用计数。"

---

## 三、架构映射：Go → Rust

### 3.1 模块划分

"建议在 Rust 项目中按以下模块组织代码："

```
reality-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                          # 公共导出
│   ├── config.rs                       # Config 结构体 → common.go 映射
│   ├── conn/
│   │   ├── mod.rs                      # Conn 结构体
│   │   ├── record.rs                   # 记录层读写
│   │   ├── half_conn.rs                # halfConn 发送/接收状态
│   │   └── mirror.rs                   # MirrorConn → 异步实现
│   ├── handshake/
│   │   ├── mod.rs                      # 握手状态机
│   │   ├── server_tls13.rs             # REALITY TLS 1.3 服务端握手
│   │   └── messages.rs                 # 握手消息序列化
│   ├── crypto/
│   │   ├── mod.rs                      # 密码学工具
│   │   ├── x25519_auth.rs              # X25519 认证流程
│   │   ├── temp_cert.rs                # 临时 Ed25519 证书
│   │   └── key_schedule.rs             # 密钥调度
│   ├── detect/
│   │   ├── mod.rs                      # 记录探测
│   │   ├── post_handshake.rs           # PostHandshake 记录长度探测
│   │   └── ccs.rs                      # ChangeCipherSpec 容忍度探测
│   ├── fallback/
│   │   ├── mod.rs                      # 回退模式
│   │   ├── proxy.rs                    # 双向转发
│   │   └── ratelimit.rs                # 限速连接
│   └── error.rs                        # 错误类型
└── tests/
    ├── integration_test.rs             # 集成测试
    └── fixtures/                       # 测试工具
```

### 3.2 核心结构体映射

**Config 结构体：**

```
Go:
    type Config struct {
        DialContext func(ctx context.Context, network, address string) (net.Conn, error)
        Show bool
        Type string
        Dest string
        Xver byte
        ServerNames  map[string]bool
        PrivateKey   []byte
        ...
    }

Rust:
    pub struct Config {
        dialer: Arc<dyn Dialer + Send + Sync>,     // func → trait object
        pub show: bool,
        pub dest_type: String,                      // Type → dest_type
        pub dest: String,
        pub xver: u8,
        pub server_names: HashSet<String>,          // map[string]bool → HashSet
        pub private_key: [u8; 32],                  // []byte → [u8; 32]
        pub min_client_ver: Option<[u8; 3]>,        // []byte → Option<[u8; 3]>
        pub max_client_ver: Option<[u8; 3]>,
        pub max_time_diff: Duration,
        pub short_ids: HashSet<[u8; 8]>,            // map[[8]byte]bool → HashSet
        pub mldsa65_key: Option<Vec<u8>>,
        pub limit_fallback_upload: Option<LimitFallback>,
        pub limit_fallback_download: Option<LimitFallback>,
    }
```

**Conn 结构体：**

```
Go:
    type Conn struct {
        AuthKey           []byte
        ClientVer         [3]byte
        ClientTime        time.Time
        ClientShortId     [8]byte
        MaxUselessRecords int
        conn              net.Conn
        isClient          bool
        handshakeFn       func(context.Context) error
        isHandshakeComplete atomic.Bool
        handshakeMutex    sync.Mutex
        in, out           halfConn
        rawInput          bytes.Buffer
        input             bytes.Reader
        hand              bytes.Buffer
        ...
    }

Rust:
    pub struct Conn<S> {
        pub auth_key: Option<Vec<u8>>,
        pub client_ver: Option<[u8; 3]>,
        pub client_time: Option<SystemTime>,
        pub client_short_id: Option<[u8; 8]>,
        pub max_useless_records: usize,
        stream: S,                                        // AsyncRead + AsyncWrite
        is_client: bool,
        is_handshake_complete: AtomicBool,
        handshake_mutex: tokio::sync::Mutex<()>,
        handshake_state: Mutex<HandshakeState>,            // 替代 handshakeFn
        in_state: Mutex<HalfConn>,                         // halfConn → HalfConn
        out_state: Mutex<HalfConn>,
        raw_input: BytesMut,                               // bytes.Buffer → BytesMut
        hand_buffer: BytesMut,
        // ... 其余状态
    }
```

**MirrorConn → AsyncMirrorStream：**

"Go 的 MirrorConn 依赖 sync.Mutex + goroutine 调度来实现读取时同步写入目标。在 Rust 中，这需要重新设计："

```rust
/// 异步镜像流：在读取上游数据的同时镜像到下游
pub struct AsyncMirrorStream<S, T> {
    upstream: S,       // 客户端连接 AsyncRead + AsyncWrite
    downstream: T,     // 目标连接 AsyncWrite
    mirrored: Arc<AtomicBool>,  // 是否已开始镜像
}

impl<S, T> AsyncMirrorStream<S, T>
where
    S: AsyncRead + AsyncWrite + Unpin,
    T: AsyncWrite + Unpin,
{
    /// 读取客户端数据并镜像到目标
    pub async fn read_and_mirror(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.upstream.read(buf).await?;
        if n > 0 {
            let _ = self.downstream.write_all(&buf[..n]).await;
        }
        Ok(n)
    }
}
```

"REALITY 的 Server 函数在 Go 中启动两个 goroutine 分别处理客户端→目标和目标→客户端的转发。在 Rust 中，这对应使用 tokio::spawn 或 tokio::join! 来并发执行两个异步任务。"

### 3.3 认证流程映射

```
Go 认证流程（tls.go Server 函数）:
    1. 从 keyShare 提取 X25519 公钥
    2. curve25519.X25519(config.PrivateKey, peerPub) → AuthKey
    3. HKDF(SHA256, AuthKey, random[:20], "REALITY") → AuthKey
    4. AES-GCM Open(plainText[:0], random[20:], session_id_ciphertext, clientHello.original)
    5. 解析明文: ClientVer + ClientTime + ClientShortId
    6. 校验: 版本范围、时间差、ShortId 合法性

Rust 认证流程（crypto/x25519_auth.rs）:
    1. 从 ClientHello key_share 扩展提取 X25519 公钥
    2. x25519_dalek::x25519(private_key, peer_pub) → shared_secret
    3. Hkdf::<Sha256>::extract_and_expand(&shared_secret, b"REALITY") → auth_key
       (salt = client_hello_random[..20])
    4. Aes256Gcm::new(&auth_key.into()).open(
         nonce: &client_hello_random[20..],
         aad: &client_hello_raw,
         ciphertext: &session_id
       )
    5. 解析明文: [client_ver(3)] [client_time(4)] [short_id(8)] [padding..]
    6. 校验: config.min_client_ver <= client_ver <= config.max_client_ver
             |now - client_time| <= config.max_time_diff
             config.short_ids.contains(&short_id)
```

### 3.4 临时证书生成映射

```
Go（handshake_server_tls13.go init + handshake）:
    var ed25519Priv ed25519.PrivateKey
    var signedCert []byte
    func init() {
        certificate := x509.Certificate{SerialNumber: &big.Int{}}
        _, ed25519Priv, _ = ed25519.GenerateKey(rand.Reader)
        signedCert, _ = x509.CreateCertificate(rand.Reader, &certificate, &certificate,
            ed25519.PublicKey(ed25519Priv[32:]), ed25519Priv)
    }
    // handshake 中:
    cert := bytes.Clone(signedCert)
    h := hmac.New(sha512.New, c.AuthKey)
    h.Write(ed25519Priv[32:])
    h.Sum(cert[:len(cert)-64])  // 替换证书签名

Rust（crypto/temp_cert.rs）:
    struct TempCert {
        cert_der: Vec<u8>,             // DER 编码证书
        private_key: ed25519_dalek::SigningKey,
    }

    impl TempCert {
        fn generate() -> Self {
            let csprng = OsRng;
            let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
            let cert = create_self_signed_cert(&signing_key.verifying_key());
            TempCert { cert_der: cert, private_key: signing_key }
        }

        fn sign_with_auth_key(&self, auth_key: &[u8]) -> Vec<u8> {
            let mut cert = self.cert_der.clone();
            let mut h = Hmac::<Sha512>::new_from_slice(auth_key).unwrap();
            h.update(self.private_key.verifying_key().as_bytes());
            let sig = h.finalize().into_bytes();
            cert[len-64..].copy_from_slice(&sig[..]);  // 替换签名
            cert
        }
    }
```

---

## 四、开发阶段划分

### 4.1 阶段一：基础设施与认证模块（2-3 周）

"目标：建立项目框架，实现 REALITY 的核心认证机制。"

```
Week 1:
├── 项目初始化：Cargo.toml、模块结构、错误类型
├── Config 结构体定义与解析
├── X25519 认证模块实现
│   ├── key_share 解析
│   ├── X25519 密钥交换
│   ├── HKDF 派生
│   └── AES-GCM AEAD 解密
└── 单元测试：认证通过/失败的各种场景

Week 2:
├── 临时 Ed25519 证书生成
│   ├── 自签名证书创建
│   └── HMAC 签名替换
├── ClientHello 解析模块
│   ├── TLS 记录层读取
│   ├── ClientHello 反序列化
│   └── 扩展解析（key_share、server_name 等）
└── 集成测试：模拟客户端发送 ClientHello，验证认证流程

Week 3（缓冲）:
├── ML-DSA-65 后量子签名集成（可选）
├── ClientVer 版本校验逻辑
├── MaxTimeDiff 时间差校验
└── 完善错误类型和日志
```

### 4.2 阶段二：TLS 1.3 握手集成（3-4 周）

"目标：集成 rustls 或自定义 TLS 1.3 握手，实现 REALITY 特有的握手修改。"

```
Week 4:
├── rustls 集成评估
│   ├── 研究 rustls ServerConfig 的自定义能力
│   ├── 评估是否可以使用 CustomCertificateVerifier
│   ├── 研究 rustls 的 ServerConnection 状态机
│   └── 确定是否需要 fork rustls
├── 如果 rustls 可用：
│   ├── 实现 CustomCertifiedKey（提供临时证书）
│   ├── 拦截 ServerHello 发送（使用目标网站的参数）
│   └── 实现握手消息长度记录
└── 如果 rustls 不可用：
    ├── 设计自定义 TLS 1.3 握手状态机
    └── 实现 ServerHello 发送逻辑

Week 5-6:
├── 完整 TLS 1.3 服务端握手实现
│   ├── ServerHello 发送
│   ├── ChangeCipherSpec 发送（兼容性）
│   ├── EncryptedExtensions 发送
│   ├── Certificate 发送（临时证书）
│   ├── CertificateVerify 发送
│   ├── Server Finished 发送
│   └── 握手密钥调度
├── Post-Handshake 记录发送
│   ├── 从 GlobalPostHandshakeRecordsLens 获取长度
│   └── 按相同长度构造并发送加密记录
└── Client Finished 接收与验证

Week 7:
├── MirrorConn 的异步实现
│   ├── AsyncMirrorStream 设计
│   ├── 双向并发转发
│   └── 与 tokio 异步运行时集成
├── Server 入口函数实现
│   ├── 连接接收
│   ├── 目标连接建立
│   ├── PROXY 协议支持（可选）
│   └── 认证 + 握手 + 回退的完整流程
└── 集成测试：完整的 REALITY 连接建立流程
```

### 4.3 阶段三：回退模式与记录探测（2-3 周）

"目标：实现回退转发机制和目标服务器的记录探测。"

```
Week 8:
├── 回退模式实现
│   ├── 双向透明转发（客户端 <-> 目标）
│   ├── io_copy 等价实现
│   ├── CloseWrite 支持
│   └── 错误处理：RST 断开连接
├── 限速连接（可选）
│   ├── 令牌桶算法实现
│   ├── RatelimitedStream 封装
│   └── 上传/下载独立限速
└── 集成测试：回退模式转发验证

Week 9:
├── 记录探测模块
│   ├── PostHandshakeRecordDetectConn 实现
│   ├── CCSDetectConn 实现
│   ├── GlobalPostHandshakeRecordsLens 存储
│   └── GlobalMaxCSSMsgCount 存储
├── 探测触发机制
│   ├── Listener 启动时自动探测
│   └── 手动探测 API
└── 集成测试：探测结果的准确性验证

Week 10（缓冲）:
├── uTLS 指纹模拟（如果需要高精确度探测）
│   ├── 手动实现常见浏览器指纹
│   └── 或使用 C FFI 调用 Go uTLS
├── 探测失败的回退处理
└── 性能优化
```

### 4.4 阶段四：完善与优化（2-3 周）

"目标：完善边界情况、性能优化、QUIC 支持（可选）。"

```
Week 11:
├── ECH（加密客户端 Hello）支持（可选）
├── 会话票据/恢复支持（可选）
├── QUIC 集成（可选）
└── FIPS 140-3 模式（可选）

Week 12:
├── 性能基准测试
│   ├── 握手延迟测量
│   ├── 吞吐量测试
│   └── 与 Go 版本对比
├── 内存使用分析
├── 并发压力测试
└── 安全审计准备

Week 13:
├── 文档完善
├── API 稳定性保证
├── 发布准备
└── CI/CD 集成
```

---

## 五、开发难度评估

### 5.1 整体难度

```
难度评估矩阵（1-5，5 为最高）：

密码学实现难度：        ████░  (4/5)
网络编程难度：           ███░░  (3/5)
异步编程难度：           ████░  (4/5)
TLS 协议复杂度：         █████  (5/5)
REALITY 特有逻辑难度：   ███░░  (3/5)
测试覆盖率保证难度：     ████░  (4/5)
安全风险控制难度：       █████  (5/5)

综合难度：               ████░  (4/5)
```

### 5.2 各模块难度分析

**模块 1：X25519 认证（难度：3/5，工作量：1-2 周）**

"X25519 密钥交换、HKDF、AES-GCM 在 Rust 中都有成熟库支持。核心难点在于正确理解 Go 版本的 AEAD nonce 构造方式（使用 ClientHello Random 的后 12 字节）和 associated data（使用 ClientHello 原始字节）。如果理解有误，解密会失败。"

"风险控制：先用 Go 版本生成测试向量（输入和输出），在 Rust 中用相同的输入验证输出是否一致。"

**模块 2：临时证书生成（难度：3/5，工作量：1 周）**

"Ed25519 密钥生成和自签名证书在 Rust 中有很好的支持。难点在于精确复现 Go x509.CreateCertificate 的输出格式，以及正确地替换证书签名位置。"

"风险控制：对比 Go 版本和 Rust 版本生成的证书的 DER 编码，逐字节验证。"

**模块 3：TLS 1.3 握手集成（难度：5/5，工作量：3-4 周）**

"这是整个移植中难度最高的部分。TLS 1.3 握手涉及复杂的密钥调度、状态机管理、加密状态转换。如果使用 rustls，难度在于如何在不修改 rustls 源码的情况下注入 REALITY 特有的行为（如使用目标网站的 ServerHello、控制握手消息长度等）。如果从零实现，难度在于正确实现 RFC 8446 的所有细节。"

"关键风险点："

"rustls 的 API 可能不暴露足够的自定义点。rustls 的 ServerConnection 内部状态机是私有的，无法直接干预 ServerHello 的内容。可能需要使用 rustls::crypto::CryptoProvider 自定义加密后端，或 fork rustls。"

"握手消息长度模仿需要精确控制每个握手消息的序列化结果和加密后的记录长度。rustls 的默认行为可能与目标网站的行为不同，导致 Post-Handshake 记录长度不匹配。"

**模块 4：MirrorConn 异步化（难度：4/5，工作量：1-2 周）**

"Go 的 MirrorConn 利用 goroutine 调度和 sync.Mutex 实现读取时同步写入目标。在 Rust 中，这需要重新设计为异步流模式。难点在于："

"并发控制：Go 中 Read 操作先 Unlock 释放锁，读取数据后再 Lock 写入目标。这保证了处理 ClientHello 的 goroutine 有优先权。在 Rust 中，需要设计类似的并发控制机制，可能使用 tokio::sync::Semaphore 或自定义的流控制。"

"错误传播：MirrorConn 的 Read 返回错误时需要关闭目标连接，同时另一个方向的转发也需要正确终止。在 Rust 中，这需要 tokio::select! 和正确的取消语义。"

**模块 5：记录探测（难度：3/5，工作量：1-2 周）**

"记录探测的核心是连接到目标网站、完成 TLS 握手、分析收到的记录。在 Rust 中，TLS 握手可以使用 rustls Client，记录分析可以直接在字节流上进行。"

"主要难点在于 uTLS 指纹模拟。Rust 生态没有直接等价物，但可以手动实现常见浏览器的 ClientHello 格式，或者使用 C FFI 调用 Go 编译的 uTLS 库。"

**模块 6：回退转发（难度：2/5，工作量：1 周）**

"回退转发是最简单的部分，本质上就是两个方向的 io_copy。Rust 的 tokio::io::copy 可以直接使用。限速功能可以使用现成的令牌桶库。"

### 5.3 人员要求

```
最低要求：
├── Rust 开发经验：1 年以上
├── TLS/SSL 协议理解：深入理解 TLS 1.3 握手流程
├── 密码学基础：理解 X25519、AES-GCM、HKDF、HMAC
├── 异步编程：tokio 异步运行时使用经验
└── 网络编程：TCP 连接、I/O 模型理解

加分项：
├── rustls 使用经验
├── Go crypto/tls 源码阅读经验
├── 后量子密码学基础
└── 安全审计经验
```

### 5.4 时间估算

```
乐观估计（熟练开发者，无重大阻塞）：   8-10 周
正常估计（中等经验开发者，合理阻塞）：  12-14 周
保守估计（新手或遇到重大技术障碍）：    16-20 周

各阶段时间分布：
├── 阶段一（认证 + 基础）：  2-3 周   (20%)
├── 阶段二（TLS 1.3 握手）： 3-4 周   (30%)
├── 阶段三（回退 + 探测）：  2-3 周   (20%)
├── 阶段四（完善 + 优化）：  2-3 周   (20%)
└── 缓冲：                  1-2 周   (10%)
```

---

## 六、测试计划

### 6.1 测试策略总览

"测试分为四个层次，从底层的单元正确性到顶层的端到端功能验证："

```
测试层级金字塔：

                    ┌─────────────────┐
                    │  端到端测试 (E2E) │  ← 真实网络环境，与 Xray 客户端联调
                   ╱                   ╲
                  ╱  集成测试 (Integration)╲  ← 模块间交互，完整握手流程
                 ╱                           ╲
                ╱      属性测试 (Property)      ╲  ← 密码学正确性，随机输入验证
               ╱                                  ╲
              ╱    单元测试 (Unit)                  ╲  ← 单个函数/方法的正确性
             ╱________________________________________╲
```

### 6.2 单元测试计划

**6.2.1 密码学模块测试**

```rust
#[cfg(test)]
mod crypto_tests {
    // 测试 1: X25519 密钥交换正确性
    #[test]
    fn test_x25519_key_exchange() {
        "验证: server_private + client_public == client_private + server_public"
    }

    // 测试 2: HKDF 派生确定性
    #[test]
    fn test_hkdf_derivation() {
        "验证: 相同输入始终产生相同输出（确定性）"
        "验证: 不同输入产生不同输出"
    }

    // 测试 3: AES-GCM 加解密往返
    #[test]
    fn test_aes_gcm_roundtrip() {
        "验证: 加密后再解密得到原始明文"
        "验证: 错误的 nonce 导致解密失败"
        "验证: 错误的 AAD 导致解密失败"
        "验证: 篡改密文导致解密失败"
    }

    // 测试 4: 完整认证流程向量测试
    #[test]
    fn test_auth_vectors() {
        "使用 Go 版本生成的测试向量："
        "  输入: client_hello_raw, session_id_ciphertext, private_key"
        "  期望输出: auth_key, client_ver, client_time, short_id"
        "验证: Rust 版本输出与 Go 版本完全一致"
    }

    // 测试 5: 认证失败场景
    #[test]
    fn test_auth_failure_cases() {
        "错误的私钥 → 解密失败"
        "错误的 session_id → 解密失败"
        "篡改的 ClientHello → 解密失败"
        "过期的 ClientTime → 时间校验失败"
        "无效的 ShortId → ShortId 校验失败"
        "版本不匹配 → 版本校验失败"
    }

    // 测试 6: HMAC 签名替换
    #[test]
    fn test_temp_cert_hmac_sign() {
        "验证: 相同 AuthKey 和私钥产生相同签名"
        "验证: 不同 AuthKey 产生不同签名"
        "验证: 签名后的证书可以被正确解析"
    }
}
```

**6.2.2 消息序列化测试**

```rust
#[cfg(test)]
mod message_tests {
    // 测试 7: ClientHello 解析往返
    #[test]
    fn test_client_hello_parse() {
        "使用已知的 ClientHello 字节流（从 Wireshark 抓包）"
        "验证: 解析后的字段值正确"
        "验证: 重新序列化后的字节流与原始一致"
    }

    // 测试 8: 扩展解析
    #[test]
    fn test_extension_parsing() {
        "验证: key_share 扩展正确解析"
        "验证: server_name 扩展正确解析"
        "验证: supported_versions 扩展正确解析"
        "验证: alpn 扩展正确解析"
    }

    // 测试 9: 边界情况
    #[test]
    fn test_edge_cases() {
        "空的 ClientHello → 解析错误"
        "过长的 ClientHello → 拒绝"
        "无效的扩展长度 → 解析错误"
    }
}
```

**6.2.3 配置和校验测试**

```rust
#[cfg(test)]
mod config_tests {
    // 测试 10: 版本校验
    #[test]
    fn test_version_validation() {
        "1.0.0 >= 1.0.0 (最小值) → 通过"
        "1.0.0 >= 1.0.1 (最小值) → 失败"
        "2.0.0 <= 2.0.0 (最大值) → 通过"
        "2.0.1 <= 2.0.0 (最大值) → 失败"
        "None (无限制) → 通过"
    }

    // 测试 11: 时间差校验
    #[test]
    fn test_time_diff() {
        "|now - client_time| <= max_diff → 通过"
        "|now - client_time| > max_diff → 失败"
        "max_diff = 0 (无限制) → 通过"
    }

    // 测试 12: ShortId 校验
    #[test]
    fn test_short_id() {
        "short_id in set → 通过"
        "short_id not in set → 失败"
        "空字符串 short_id → 通过（如果配置允许）"
    }
}
```

### 6.3 集成测试计划

**6.3.1 完整握手流程测试**

```rust
#[cfg(test)]
mod integration_tests {
    // 测试 13: 成功的 REALITY 握手
    #[tokio::test]
    async fn test_successful_reality_handshake() {
        "搭建 REALITY 服务端（配置: dest, private_key, server_names, short_ids）"
        "模拟 REALITY 客户端发送正确的 ClientHello"
        "验证: 服务端认证成功"
        "验证: TLS 握手完成"
        "验证: 返回的 Conn 可以用于后续数据传输"
    }

    // 测试 14: 回退模式测试
    #[tokio::test]
    async fn test_fallback_mode() {
        "发送非 REALITY 的 TLS ClientHello（如普通浏览器）"
        "验证: 认证失败，进入回退模式"
        "验证: 客户端收到目标网站的真实证书"
        "验证: 客户端与目标网站的通信正常转发"
    }

    // 测试 15: SNI 不匹配测试
    #[tokio::test]
    async fn test_sni_mismatch() {
        "发送 ClientHello 但 SNI 不在 ServerNames 列表中"
        "验证: 直接进入回退模式（不进行认证）"
    }

    // 测试 16: TLS 版本不支持
    #[tokio::test]
    async fn test_unsupported_tls_version() {
        "发送 TLS 1.2 ClientHello"
        "验证: 服务端拒绝连接"
    }
}
```

**6.3.2 记录探测测试**

```rust
#[cfg(test)]
mod detect_tests {
    // 测试 17: Post-Handshake 记录长度探测
    #[tokio::test]
    async fn test_post_handshake_detect() {
        "配置目标网站（可使用本地 mock 服务器）"
        "启动记录探测"
        "验证: 探测到正确的 Post-Handshake 记录数量"
        "验证: 每条记录的长度与目标网站一致"
    }

    // 测试 18: CCS 容忍度探测
    #[tokio::test]
    async fn test_ccs_detect() {
        "配置目标网站"
        "启动 CCS 探测"
        "验证: 探测到目标网站的 CCS 容忍度"
    }
}
```

### 6.4 端到端测试计划

**6.4.1 与 Xray-core 联调测试**

```
测试环境搭建：
├── REALITY Rust 服务端（监听 127.0.0.1:8443）
├── Xray-core 客户端（使用 reality outbounds 配置）
├── 目标网站（使用 Nginx 监听 127.0.0.1:9443，配置自签名证书）
└── 测试客户端（curl 或自定义 HTTP 客户端）

测试用例：
├── E2E-1: REALITY 代理 HTTP 请求成功
│   "客户端通过 Xray → REALITY Rust 服务端 → Nginx 目标"
│   "验证: 收到 Nginx 的 HTTP 响应"
│   "验证: TLS 证书为临时 Ed25519 证书"
├── E2E-2: 直接浏览器访问目标网站
│   "浏览器直接访问 REALITY Rust 服务端"
│   "验证: 浏览器显示 Nginx 的真实证书"
│   "验证: 网页内容正常显示"
├── E2E-3: 错误的 REALITY 密钥
│   "使用错误 private_key 的 Xray 客户端"
│   "验证: 认证失败，进入回退模式"
│   "验证: 客户端收到目标网站真实证书"
├── E2E-4: 高并发连接
│   "同时建立 100 个 REALITY 连接"
│   "验证: 所有连接成功握手"
│   "验证: 无内存泄露"
│   "验证: CPU 使用率在合理范围"
└── E2E-5: 长时间稳定性
    "持续运行 24 小时"
    "每 10 秒发起一个 REALITY 连接"
    "验证: 无连接失败"
    "验证: 内存使用稳定"
```

### 6.5 交叉验证测试

**6.5.1 Go ↔ Rust 互操作性测试**

```
测试策略：使用相同的输入，验证 Go 和 Rust 版本的输出一致。

交叉验证表：
┌────────────────────┬──────────────┬──────────────┬──────────┐
│ 测试项              │ Go 输入       │ Rust 输入     │ 验证方式  │
├────────────────────┼──────────────┼──────────────┼──────────┤
│ X25519 共享密钥     │ 私钥+公钥    │ 相同私钥+公钥 │ 字节比较   │
│ HKDF 派生结果       │ salt+ikm+info│ 相同参数      │ 字节比较   │
│ AES-GCM 加密结果    │ key+nonce+aad│ 相同参数      │ 字节比较   │
│ AES-GCM 解密结果    │ key+nonce+aad│ 相同参数      │ 字节比较   │
│ HMAC-SHA512 结果    │ key+data     │ 相同参数      │ 字节比较   │
│ Ed25519 签名        │ key+message  │ 相同参数      │ 验证通过   │
│ 临时证书 DER        │ 公钥+序列号  │ 相同参数      │ 字节比较   │
│ ClientHello 解析    │ 字节流       │ 相同字节流    │ 字段比较   │
└────────────────────┴──────────────┴──────────────┴──────────┘
```

### 6.6 性能基准测试

```rust
#[cfg(test)]
mod bench_tests {
    // 基准测试 1: 认证延迟
    #[tokio::test]
    async fn bench_auth_latency() {
        "测量 X25519 + HKDF + AES-GCM 解密的总延迟"
        "目标: < 1ms（与 Go 版本对比）"
    }

    // 基准测试 2: 握手延迟
    #[tokio::test]
    async fn bench_handshake_latency() {
        "测量完整 TLS 1.3 REALITY 握手的延迟"
        "目标: < 5ms（与 Go 版本对比）"
    }

    // 基准测试 3: 数据吞吐量
    #[tokio::test]
    async fn bench_throughput() {
        "测量握手后的数据传输吞吐量"
        "目标: > 1Gbps（与 Go 版本对比）"
    }

    // 基准测试 4: 内存使用
    #[tokio::test]
    async fn bench_memory() {
        "测量 1000 个并发连接的内存使用"
        "目标: < 100MB 总内存"
    }
}
```

### 6.7 模糊测试（Fuzzing）

```
使用 cargo-fuzz 对关键模块进行模糊测试：

fuzz targets:
├── fuzz_client_hello_parse:
│   "随机字节输入 → ClientHello 解析"
│   "目标: 不 panic、不 crash、不内存泄露"
├── fuzz_aead_decrypt:
│   "随机密钥、nonce、密文 → AEAD 解密"
│   "目标: 不 panic、不 crash"
├── fuzz_cert_parse:
│   "随机 DER 字节 → X.509 证书解析"
│   "目标: 不 panic、不 crash"
└── fuzz_handshake_state:
    "随机握手消息序列 → 握手状态机"
    "目标: 状态转换正确、无死锁"
```

### 6.8 测试覆盖率目标

```
覆盖率目标：

模块                 │ 行覆盖率  │ 分支覆盖率  │ 函数覆盖率
─────────────────────┼──────────┼────────────┼──────────
crypto/x25519_auth   │  ≥ 95%   │   ≥ 90%    │   100%
crypto/temp_cert     │  ≥ 90%   │   ≥ 85%    │   100%
conn/mirror          │  ≥ 85%   │   ≥ 80%    │   ≥ 95%
handshake/server_tls13│ ≥ 90%   │   ≥ 85%    │   100%
detect/*             │  ≥ 85%   │   ≥ 80%    │   ≥ 95%
fallback/*           │  ≥ 85%   │   ≥ 80%    │   ≥ 95%
─────────────────────┼──────────┼────────────┼──────────
总体                 │  ≥ 90%   │   ≥ 85%    │   ≥ 95%
```

---

## 七、风险与应对

### 7.1 技术风险

```
风险矩阵：

风险 ID │ 风险描述                          │ 概率 │ 影响 │ 缓解措施
────────┼──────────────────────────────────┼──────┼──────┼─────────────────────────
R1     │ rustls API 无法支持 REALITY 握手  │ 中   │ 高   │ 早期验证（Week 1-2），准备 fork 方案
R2     │ uTLS 指纹模拟不精确               │ 高   │ 中   │ 手动实现常见指纹作为 fallback
R3     │ AES-GCM nonce 构造理解错误        │ 低   │ 高   │ 交叉验证测试（Go ↔ Rust 向量对比）
R4     │ 异步 MirrorConn 并发控制不当      │ 中   │ 中   │ 参考 Go 源码的锁模式，编写并发测试
R5     │ TLS 1.3 密钥调度实现错误          │ 中   │ 高   │ 使用 RFC 8446 测试向量逐层验证
R6     │ 后量子密码库不成熟                │ 高   │ 低   │ 作为可选功能，不影响核心流程
R7     │ 记录探测结果与 Go 版本不一致      │ 中   │ 中   │ 与 Go 版本对比探测结果
R8     │ 性能不达标                        │ 低   │ 中   │ 早期基准测试，持续监控
```

### 7.2 项目风险

```
风险 ID │ 风险描述                          │ 概率 │ 影响 │ 缓解措施
────────┼──────────────────────────────────┼──────┼──────┼─────────────────────────
P1     │ 开发周期超出预期                  │ 高   │ 中   │ 分阶段交付，优先核心功能
P2     │ 缺乏 TLS 协议专家                 │ 中   │ 高   │ 提前招募有经验的开发者
P3     │ 上游 rustls 重大变更              │ 低   │ 中   │ 锁定依赖版本，关注更新
P4     │ 安全审计发现漏洞                  │ 中   │ 高   │ 开发过程中引入安全审查
P5     │ Xray-core 协议变更                │ 低   │ 高   │ 与 Xray 团队保持沟通
```

### 7.3 关键里程碑检查点

```
检查点 1（Week 2 结束）:
├── 认证模块通过 Go ↔ Rust 交叉验证
├── 临时证书生成与 Go 版本字节级一致
└── 如果 rustls 集成不可行，决定 fork 或自研

检查点 2（Week 6 结束）:
├── 完整握手流程在本地测试通过
├── 与 Xray 客户端成功建立 REALITY 连接
└── 如果握手失败，回溯问题并调整方案

检查点 3（Week 10 结束）:
├── 回退模式和记录探测功能完整
├── 端到端测试全部通过
└── 如果仍有未通过的测试，评估是否可以发布

检查点 4（Week 13 结束）:
├── 性能基准测试达标
├── 测试覆盖率达标
├── 模糊测试通过
└── 可以发布 v0.1.0
```

---

## 八、推荐实施路线图

### 8.1 总体路线图

```
Month 1: 基础 + 认证 + rustls 集成验证
Week 1-2:  项目初始化 + X25519 认证模块 + rustls 可行性验证
Week 3-4:  临时证书 + ClientHello 解析 + 交叉验证测试

Month 2: TLS 1.3 握手 + MirrorConn
Week 5-6:  rustls 深度集成 + 完整 TLS 1.3 REALITY 握手
Week 7-8:  异步 MirrorConn + Server 入口函数 + 端到端联调

Month 3: 回退 + 探测 + 完善
Week 9-10: 回退模式 + 记录探测 + 限速
Week 11-12: 性能优化 + 模糊测试 + 文档 + 发布准备
```

### 8.2 最小可行产品（MVP）范围

"如果要快速验证可行性，MVP 应包含以下功能："

```
MVP 功能清单：
├── X25519 认证（核心：解密 ClientHello session_id）
├── 临时 Ed25519 证书生成
├── 基本 TLS 1.3 握手（使用 rustls + 自定义证书）
├── MirrorConn 基础版（双向转发，无锁优化）
├── 回退模式（直接 io_copy 转发）
└── 基本错误处理和日志

MVP 不包含：
├── ML-DSA-65 后量子签名
├── X25519MLKEM768 混合密钥交换
├── 记录探测机制
├── 回退限速
├── PROXY 协议
├── ECH 支持
├── QUIC 集成
└── FIPS 140-3 模式
```

### 8.3 关键决策树

```
开始
  │
  ▼
rustls 是否支持 REALITY 握手？（Week 1-2 验证）
  │
  ├── 是 → 基于 rustls 构建 REALITY 层（路径 A）
  │         │
  │         ▼
  │     rustls 是否支持自定义 ServerHello 内容？
  │         │
  │         ├── 是 → 直接使用 rustls API
  │         │
  │         └── 否 → Fork rustls，修改 ServerHello 发送逻辑
  │
  └── 否 → 从零实现 TLS 1.3 + REALITY（路径 B）
            │
            ▼
        评估工作量（+4-6 周）和安全性风险
            │
            ├── 接受 → 参考 Go 源码逐模块实现
            │
            └── 不接受 → 重新评估项目可行性
```

### 8.4 成功标准

"项目成功需要满足以下标准："

```
功能标准：
├── 与 Xray-core 客户端成功建立 REALITY 连接
├── 认证失败的连接正确回退到目标网站
├── 临时证书被客户端正确接受
├── 回退模式下客户端收到目标网站的真实证书
└── Post-Handshake 记录长度与目标网站一致

性能标准：
├── 握手延迟 ≤ Go 版本的 120%
├── 吞吐量 ≥ Go 版本的 80%
├── 内存使用 ≤ Go 版本的 120%
└── 100 并发连接无异常

安全标准：
├── 通过模糊测试（无 crash、无 panic）
├── 通过交叉验证测试（与 Go 版本输出一致）
├── 无已知的常量时间攻击漏洞
└── 无缓冲区溢出、空指针解引用等内存安全问题（Rust 天然保证）

质量标准：
├── 测试覆盖率 ≥ 90%
├── 所有集成测试通过
├── 端到端测试通过
└── 文档完整（API 文档 + 使用指南）
```
