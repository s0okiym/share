# xray-rust-client 部署与测试文档

本文档说明如何部署 `xray-rust-client` 并连接到本地 `127.0.0.1:8443` 的 Xray VLESS+REALITY+Vision 服务端。

---

## 1. 服务端信息

本机已运行 Xray 服务端，配置如下：

| 项目 | 值 |
|------|-----|
| 监听地址 | `0.0.0.0:8443` |
| 协议 | VLESS |
| 流控 (Flow) | `xtls-rprx-vision` |
| 传输安全 | REALITY |
| dest | `academy.nvidia.com:443` |
| serverNames | `academy.nvidia.com` |
| publicKey | `8SIbnPJwRCGj9cywKkTckPtskKqH5XCGgrLNLHcyuFE` |
| 可用 shortIds | `112233`, `aabbcc`, `1a1b1c`, `2a2b2c`, `3a3b3c`, `a1b1c1`, `a2b2c2`, `a3b3c3` |

### 可用客户端 UUID（任选其一）

```
5040b974-2897-446c-9902-f804e6ff94e8
52e5579a-1804-4316-b724-7b1c9d702757
2803fb38-0f31-41d7-a3c3-30a9ddf24204
88bbcfa2-1c23-4711-a5fa-b7a0f78e2ec1
d242b30f-ed0e-4ded-a495-b0998fbf2d14
ef35f2d0-5a2b-4abc-b8ef-c821f759b9dd
60cac514-c499-4940-adc1-6f5f90b952e3
a83974ee-7e1f-4cd7-a1f5-408d0cdff6c7
```

---

## 2. 环境要求

- **Rust**: 1.70+（已安装 `cargo`）
- **Xray 二进制**: `/usr/local/bin/xray`（用于 REALITY 后端）
- **操作系统**: Linux（本项目在 Linux 上开发和测试）

检查环境：

```bash
# 检查 Rust
cargo --version

# 检查 Xray
xray version
# 或
/usr/local/bin/xray version
```

---

## 3. 编译

```bash
cd /root/kimi/Xray-core/impl_kimi
cargo build --release
```

编译产物：

```
./target/release/xray-rust-client
```

---

## 4. 配置

创建配置文件 `config.toml`（项目目录下已提供示例）：

```toml
[log]
level = "info"

[inbound]
listen = "127.0.0.1:1080"
protocol = "mixed"
# 如需认证，取消下面两行注释：
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

### 配置项说明

| 配置项 | 说明 |
|--------|------|
| `inbound.listen` | 本地监听地址，SOCKS5/HTTP Mixed 代理入口 |
| `inbound.protocol` | `mixed`（自动识别 SOCKS5/HTTP）、`socks5`、`http` |
| `inbound.username` / `password` | 可选，代理认证 |
| `outbound.address` | 服务端地址，本机测试用 `127.0.0.1` |
| `outbound.port` | 服务端端口 `8443` |
| `outbound.uuid` | 从上方 8 个 UUID 中任选一个 |
| `outbound.flow` | 固定为 `xtls-rprx-vision` |
| `outbound.transport.type` | 固定为 `reality` |
| `outbound.transport.server_name` | REALITY 伪装域名，与服务端一致 |
| `outbound.transport.public_key` | REALITY 公钥 |
| `outbound.transport.short_id` | 从可用列表中任选一个 |
| `outbound.transport.fingerprint` | TLS 指纹伪装，可选 `chrome`、`firefox`、`safari`、`edge` 等 |

---

## 5. 运行

### 前台运行（调试用）

```bash
cd /root/kimi/Xray-core/impl_kimi
RUST_LOG=info ./target/release/xray-rust-client --config config.toml
```

### 后台运行

```bash
cd /root/kimi/Xray-core/impl_kimi
setsid RUST_LOG=info ./target/release/xray-rust-client --config config.toml > /tmp/client.log 2>&1 &
```

### 验证启动成功

```bash
# 检查 1080 端口是否监听
ss -tlnp | grep 1080

# 查看日志
tail -f /tmp/client.log
```

期望输出：

```
INFO xray_rust_client::xray_backend: starting xray backend on socks5://127.0.0.1:<随机端口>
INFO xray_rust_client: xray REALITY backend started at socks5://127.0.0.1:<随机端口>
INFO xray_rust_client::inbound::mixed: Mixed inbound listening on 127.0.0.1:1080
```

---

## 6. 使用代理

### 6.1 curl 测试

#### SOCKS5 代理

```bash
# HTTP 测试
curl --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get

# HTTPS 测试
curl --proxy socks5h://127.0.0.1:1080 https://httpbin.org/get

# 访问 Google
curl --proxy socks5h://127.0.0.1:1080 http://www.google.com
```

#### HTTP 代理

```bash
# HTTP 测试
curl --proxy http://127.0.0.1:1080 http://httpbin.org/get

# HTTPS 测试（使用 CONNECT 隧道）
curl --proxy http://127.0.0.1:1080 https://httpbin.org/get
```

### 6.2 浏览器配置

在浏览器或系统代理设置中配置：

- **SOCKS5**: `127.0.0.1:1080`，无认证（或填写配置的 username/password）
- **HTTP**: `127.0.0.1:1080`

### 6.3 命令行工具配置

#### Git

```bash
git config --global http.proxy socks5h://127.0.0.1:1080
git config --global https.proxy socks5h://127.0.0.1:1080
```

#### apt / wget / 其他工具

使用 `proxychains-ng`：

```bash
# 安装
apt install proxychains-ng

# 配置 /etc/proxychains4.conf，添加：
socks5 127.0.0.1 1080

# 使用
proxychains curl https://www.google.com
```

---

## 7. 测试

### 7.1 单元测试

```bash
cd /root/kimi/Xray-core/impl_kimi
cargo test
```

期望输出：20 个测试全部通过。

### 7.2 全面端到端测试

```bash
cd /root/kimi/Xray-core/impl_kimi
bash test_reports/test_runner_v2.sh
```

该脚本执行：

1. 20 个单元测试
2. Release 编译
3. SOCKS5 HTTP/HTTPS 代理测试（httpbin.org, google.com）
4. HTTP 代理 HTTP/HTTPS 测试
5. IPv4 / 域名目标测试
6. 并发连接测试
7. 端口监听检查
8. 日志验证
9. 性能基准测试

期望结果：**13 项测试全部通过，通过率 100%**。

---

## 8. 停止客户端

```bash
# 查找进程
ps aux | grep xray-rust-client | grep -v grep

# 停止（替换 <PID> 为实际进程号）
kill <PID>

# 同时清理 xray 后端
pkill -f "xray run -config /tmp/xray_backend"
```

---

## 9. 故障排查

| 现象 | 排查方法 |
|------|----------|
| 端口 1080 无法连接 | `ss -tlnp \| grep 1080` 检查是否监听；检查防火墙 |
| 日志无输出 | 确认设置了 `RUST_LOG=info` |
| xray 后端启动失败 | 检查 `/usr/local/bin/xray` 是否存在且可执行；查看 `/tmp/xray_backend_*.json` |
| 代理访问超时 | 确认服务端 8443 正常：`ss -tlnp \| grep 8443`；确认 UUID/short_id 配置正确 |
| 测试脚本超时 | 检查网络连通性；单独运行 `curl --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get` 排查 |

---

## 10. 架构说明

```
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   应用程序       │      │  xray-rust-client    │      │   Xray 服务端     │
│ (curl/浏览器)   │─────▶│  127.0.0.1:1080     │      │  127.0.0.1:8443  │
└─────────────────┘      │  Mixed 入站代理       │      │  VLESS+REALITY   │
                         │                      │      │  +Vision         │
                         │  ┌────────────────┐  │      └──────────────────┘
                         │  │ SOCKS5/HTTP    │  │               ▲
                         │  │ 自动协议识别   │  │               │
                         │  └────────────────┘  │               │
                         │           │           │               │
                         │  ┌────────▼─────────┐ │               │
                         │  │ SOCKS5 转发到    │ │               │
                         │  │ xray 后端        │ │               │
                         │  │ (随机本地端口)   │ │               │
                         │  └────────┬─────────┘ │               │
                         │           │           │               │
                         │  ┌────────▼─────────┐ │               │
                         │  │ xray 子进程      │ │───────────────┘
                         │  │ VLESS+REALITY    │ │  VLESS 协议
                         │  │ +Vision 出站     │ │  REALITY TLS
                         │  └──────────────────┘ │
                         └───────────────────────┘
```

- **Rust 层**：处理 SOCKS5/HTTP 入站，管理 xray 子进程生命周期
- **xray 子进程**：负责 VLESS 协议封装、REALITY TLS 握手、XTLS Vision 流控
- 程序退出时 Rust Drop 会自动终止 xray 子进程
