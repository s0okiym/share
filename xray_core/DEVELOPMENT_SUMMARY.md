# Xray-rs 开发总结

## 项目概览

本项目是 Xray-core 的 Rust 实现，命名为 `xray-rs`。目标是在 Rust 生态中重建 Xray 的核心功能，提供等效的网络代理能力。

**实现范围**:
- HTTP CONNECT 代理入口（完整，含 Dispatcher 集成）
- SOCKS5 代理入口（TCP + UDP Associate，含 Dispatcher 集成）
- VLESS 协议出口（TCP 隧道，含 Vision 模式）
- VLESS 请求/响应头编码解码
- VLESS Addons protobuf 序列化
- XTLS Vision 流量检测 + 完整填充/解填充逻辑
- REALITY SessionId 编码加密
- XHTTP 配置结构
- 完整的 CLI 和配置加载系统
- Dispatcher trait 抽象：入站到出站的完整桥接

## 项目结构

```
impl/
├── Cargo.toml                          # Workspace 根
├── xray-common/                        # 共享工具库 (6 tests)
│   ├── src/
│   │   ├── address.rs                  # 地址编码解码 (IPv4/IPv6/Domain + Port)
│   │   ├── buf.rs                      # 8KB 缓冲池
│   │   └── pipe.rs                     # 异步管道 (2 tests)
├── xray-inbounds/                      # 入口协议 (11 tests)
│   ├── src/
│   │   ├── http_inbound.rs             # HTTP CONNECT 代理 (10 tests, 含 dispatch 测试)
│   │   └── socks5_inbound.rs           # SOCKS5 代理 (1 test)
├── xray-vless/                         # VLESS 协议库 (6 tests)
│   ├── src/
│   │   ├── header.rs                   # 请求/响应头编码解码 (2 tests)
│   │   ├── addons.rs                   # Protobuf Addons 序列化 (2 tests)
│   │   └── outbound.rs                 # VLESS 出站处理器 (2 tests)
├── xray-reality/                       # REALITY 传输 (1 test)
│   ├── src/
│   │   ├── client.rs                   # REALITY 客户端框架
│   │   └── session_id.rs               # SessionId 编码加密 (1 test)
├── xray-vision/                        # XTLS Vision (9 tests)
│   ├── src/
│   │   ├── traffic_state.rs            # 流量状态
│   │   ├── reader.rs                   # VisionReader 完整实现 (4 tests)
│   │   ├── writer.rs                   # VisionWriter 完整实现 (2 tests)
│   │   └── filter.rs                   # TLS 1.3 检测 (3 tests)
├── xray-xhttp/                         # XHTTP 传输
│   ├── src/
│   │   ├── client.rs                   # XHTTP 客户端配置
│   │   └── connection.rs               # splitConn 封装
└── xray-core/                          # 主二进制 + 集成测试 (9 tests)
    ├── src/main.rs                     # CLI、配置加载、启动逻辑
    └── tests/integration_test.rs       # 线格式兼容性测试 (9 tests)
```

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 异步运行时 | tokio | 生态最成熟，性能最优 |
| TLS | rustls | 纯 Rust，安全审计 |
| 序列化 | serde + serde_json | Rust 生态标准 |
| Protobuf | 手动编码 | VLESS addons 结构简单，避免引入 prost 依赖 |
| UUID | uuid crate | 标准 UUID 解析 |
| 加密 | x25519-dalek, aes-gcm, hkdf, sha2 | 模块化加密库 |
| 错误处理 | thiserror | 标准派生错误类型 |
| 日志 | tracing + tracing-subscriber | 结构化日志，支持 env-filter |
| CLI | clap (derive) | 类型安全的参数解析 |

## 关键设计决策

### 1. 地址编码：Port-Then-Address 格式

VLESS 协议使用 Go 版本的 `PortThenAddress()` 格式，即端口在前（2 bytes BE），地址类型和地址在后。Rust 实现完全匹配这一格式。

### 2. 缓冲池：8KB 固定大小

匹配 Go 的 `buf.Size = 8192`，使用 `Box<[u8; 8192]>` 避免栈溢出，通过 `tokio::sync::Mutex<Vec<>>` 实现池化。

### 3. Dispatcher 模式

通过 `Dispatcher` trait 抽象入站到出站的转发逻辑。入bounds 接受一个实现了 `Dispatcher` trait 的对象，实现协议无关的流量转发。HTTP CONNECT 和 SOCKS5 CONNECT 都通过 Dispatcher 将流量转发到 VLESS outbound。

### 4. XTLS Vision 流量整形

VisionWriter 在出站方向添加 21 字节头部（UUID + CMD + ContentLen + PaddingLen）和随机填充，隐藏真实流量特征。VisionReader 在入站方向解析头部、提取内容、跳过填充。支持 Continue/End/Direct 三种命令状态机。

### 4. 模块化 crate 结构

每个协议/传输层独立为 crate，通过 workspace 管理依赖。这允许：
- 独立测试每个组件
- 按需启用功能（cargo features）
- 清晰的模块边界

## 编译与运行

```bash
# 环境准备（必须）
. "$HOME/.cargo/env"

# 编译
cargo build --release

# 测试
cargo test

# 验证配置
cargo run --bin xray-rs -- run -c config.json --test

# 打印配置
cargo run --bin xray-rs -- run -c config.json --dump

# 启动服务
cargo run --bin xray-rs -- run -c config.json
```

## 测试统计

| 模块 | 单元测试 | 集成测试 | 总计 |
|------|---------|---------|------|
| xray-common | 6 | 0 | 6 |
| xray-inbounds | 11 | 0 | 11 |
| xray-reality | 1 | 0 | 1 |
| xray-vision | 9 | 0 | 9 |
| xray-vless | 6 | 0 | 6 |
| xray-core | 0 | 9 | 9 |
| **总计** | **33** | **9** | **42** |

## 与 Go 实现的等价性验证

### 已验证的等价项

1. **VLESS 请求头编码**: 字节级匹配 Go 实现
   - 版本号、UUID、Addons 长度、命令、地址、端口
   - IPv4/IPv6/Domain 三种地址类型
   - Port-Then-Address 字节序

2. **VLESS 响应头编码**: 版本 + Addons 长度 + Addons

3. **VLESS Addons protobuf**: 手动编码与 Go 的 prost 输出兼容
   - Field 1 (Flow): tag=0x0A
   - Field 2 (Seed): tag=0x12

4. **地址编码**: IPv4/IPv6/Domain + Port 的线格式

5. **REALITY SessionId**: AES-GCM 加密、HKDF 派生

6. **TLS 检测**: ClientHello/ServerHello 模式匹配

### 待实现的等价项

1. **REALITY TLS 完整握手**:
   - uTLS 指纹（Chrome/Firefox ClientHello 字节级模拟）
   - rustls 依赖已添加，TLS 握手框架已就位
   - 需要使用 rustls 的 `dangerous()` API 实现自定义证书验证

2. **XHTTP 三种传输模式**:
   - stream-one: 单双向 HTTP 请求
   - stream-up: 分离的上行/下行
   - packet-up: 分块 POST + 长轮询 GET

3. **标准 TLS 出站（非 REALITY）**:
   - 使用 rustls 建立普通 TLS 连接

4. **UDP over VLESS 隧道**:
   - 长度前缀编码已定义
   - 需要完整的 UDP 转发逻辑

## 已知限制

1. **REALITY**: 仅实现了 SessionId 加密框架，完整的 TLS 握手需要 rustls 集成（已添加依赖，待实现）
2. **Vision**: 填充/解填充逻辑已完整实现（VisionWriter/VisionReader），包含状态机、多块处理、UUID 验证
3. **XHTTP**: 配置结构已定义，HTTP 客户端未实现
4. **TLS**: 标准 TLS 出站（非 REALITY）未实现
5. **UDP over VLESS**: 长度前缀编码已定义，但完整 UDP 隧道未实现
6. **Dispatcher 集成**: HTTP 和 SOCKS5 入站已通过 Dispatcher trait 与 VLESS 出站打通

## 后续开发建议

### 已完成 (Phase 8)
1. ~~HTTP/SOCKS5 入站通过 Dispatcher 与 VLESS 出站集成~~
2. ~~XTLS Vision Writer/Reader 完整实现（21字节头部、随机填充、状态机）~~
3. ~~Vision 编码解码 roundtrip 测试~~

### 优先级 P0（核心功能）
4. 实现 REALITY TLS 完整握手（使用 rustls，已添加依赖）
5. 将 REALITY TLS 集成到 VLESS outbound.process()

### 优先级 P1（完整功能）
6. 实现 XHTTP 三种模式（stream-one/stream-up/packet-up）
7. 实现标准 TLS 出站（rustls）
8. UDP over VLESS 隧道

### 优先级 P2（优化）
9. 使用 io_uring 实现 splice 零拷贝
10. 连接池和复用
11. 多出站负载均衡
