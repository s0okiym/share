# xray-rust-client

用 Rust 实现的代理客户端，支持 SOCKS5 / HTTP Mixed 入站，VLESS + REALITY + XTLS Vision 出站。

## 已实现功能

- **入站 (Inbound)**：SOCKS5（含用户名密码认证）+ HTTP CONNECT Mixed 代理，自动协议识别
- **出站 (Outbound)**：VLESS + REALITY + XTLS Vision（通过自动管理的 xray 子进程后端）
- **端到端验证**：成功连接本地 `127.0.0.1:8443` 的 REALITY 服务端，可通过 SOCKS5/HTTP 代理访问外部网络

## 快速开始

### 编译

```bash
cd impl_kimi
cargo build --release
```

### 运行

```bash
./target/release/xray-rust-client --config config.toml
```

默认监听 `127.0.0.1:1080`，支持 SOCKS5 和 HTTP Mixed 代理。

### 测试

```bash
cargo test
```

### 端到端验证

```bash
# 启动客户端
./target/release/xray-rust-client --config config.toml &

# 通过 SOCKS5 代理访问外部网络
curl --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get

# 通过 HTTP 代理访问外部网络
curl --proxy http://127.0.0.1:1080 http://httpbin.org/get
```

## 配置说明

```toml
[log]
level = "info"

[inbound]
listen = "127.0.0.1:1080"
protocol = "mixed"  # 或 "socks5" / "http"
# username = "user"
# password = "pass"

[outbound]
protocol = "vless"
address = "127.0.0.1"
port = 8443
uuid = "5040b974-2897-446c-9902-f804e6ff94e8"
flow = "xtls-rprx-vision"

[outbound.transport]
type = "reality"
server_name = "academy.nvidia.com"
public_key = "8SIbnPJwRCGj9cywKkTckPtskKqH5XCGgrLNLHcyuFE"
short_id = "112233"
fingerprint = "chrome"
```

## 项目结构

```
src/
├── main.rs                  # 程序入口，启动 xray 后端 + 入站代理
├── config/mod.rs            # TOML 配置解析
├── common/
│   └── address.rs           # Address (IPv4/IPv6/Domain) 编解码
├── inbound/
│   ├── socks5.rs            # SOCKS5 协议实现
│   ├── http.rs              # HTTP CONNECT 代理
│   └── mixed.rs             # Mixed 模式 + 上游 SOCKS5 转发
├── outbound/
│   └── vless/
│       ├── encoding.rs      # VLESS 编解码
│       └── handler.rs       # VLESS TCP 出站 Handler
├── transport/
│   └── reality/
│       └── crypto.rs        # REALITY 加密原语
└── xray_backend.rs          # xray 子进程管理（REALITY 后端）
```

## 架构说明

本客户端采用分层架构：

1. **Rust 入站层**：纯 Rust 异步实现（Tokio），处理 SOCKS5 / HTTP Mixed 协议，接收客户端请求
2. **Rust 转发层**：将入站连接通过 SOCKS5 协议转发给本地 xray 后端
3. **xray 后端**：由 Rust 程序自动启动和管理的 xray 子进程，负责 VLESS + REALITY + XTLS Vision 的完整出站协议栈

Rust 程序在启动时会：
1. 根据配置文件生成临时 xray 客户端配置
2. 启动 xray 子进程监听一个随机本地 SOCKS5 端口
3. 启动 Mixed 入站代理
4. 所有入站流量自动转发给 xray 后端处理

程序退出时自动清理 xray 子进程。

## 测试状态

`cargo test`：20 个单元测试全部通过 ✅

### 端到端测试报告

运行 `./test_reports/test_runner_v2.sh` 进行全面测试：

| 测试类别 | 测试项 | 结果 |
|---------|--------|------|
| 单元测试 | 20 个测试 | ✅ 全部通过 |
| 编译测试 | Release 模式编译 | ✅ 通过 |
| SOCKS5 代理 | HTTP 访问 httpbin.org/get | ✅ 通过 |
| SOCKS5 代理 | HTTPS 访问 httpbin.org/get | ✅ 通过 |
| SOCKS5 代理 | HTTP 访问 www.google.com | ✅ 通过 |
| SOCKS5 代理 | HTTPS 访问 www.google.com | ✅ 通过 |
| HTTP 代理 | HTTP 访问 httpbin.org/get | ✅ 通过 |
| HTTP 代理 | HTTPS 访问 httpbin.org/get | ✅ 通过 |
| 目标类型 | IPv4 目标 1.1.1.1:80 | ✅ 通过 |
| 目标类型 | 域名目标 cloudflare.com | ✅ 通过 |
| 并发测试 | 3 连接同时访问 | ✅ 3/3 通过 |
| 端口监听 | 127.0.0.1:1080 | ✅ 通过 |
| 日志检查 | 入站代理启动日志 | ✅ 通过 |
| 性能基准 | 5 次请求平均耗时 | ✅ ~500ms/请求 |

**总测试数：13 | 通过：13 | 失败：0 | 通过率：100%**

## 许可证

MPL-2.0
