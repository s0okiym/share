# 项目概览、设计目标与权衡

## 项目定位

`shoes` 是一个用 Rust 编写的高性能多协议代理服务器，同时也是一个可嵌入移动端的 VPN 库。它既能作为独立 CLI 运行，也能通过 FFI 被 Android/iOS VPN 应用集成。

## 设计目标

### 1. 多协议统一承载
支持大量代理协议与传输层组合：
- **代理协议**：VLESS（含 XTLS VISION）、VMess AEAD、Trojan、Shadowsocks（含 2022-BLAKE3）、Snell v3、NaiveProxy、AnyTLS、SOCKS5、HTTP、H2MUX
- **UDP 协议**：Hysteria2、TUIC v5
- **传输层**：TCP、TLS（rustls）、QUIC、WebSocket、REALITY、ShadowTLS v3
- **VPN 模式**：TUN 设备透明代理（Linux/Android/iOS）

### 2. 统一流抽象
通过 `AsyncStream` / `AsyncMessageStream` 等 trait 将所有 TCP-like 流和 UDP-like 消息流统一抽象，使得：
- 任意协议 handler 可以叠加在任意传输层之上
- 双向拷贝逻辑 `copy_bidirectional` 与协议无关
- 代理链可以无缝组合不同协议层

### 3. 规则化路由与多跳代理链
- 基于目标地址（CIDR、域名通配符、端口）的匹配规则
- 支持多跳代理链（multi-hop chain），每跳可配置不同协议
- LRU 缓存加速频繁命中的路由决策

### 4. 移动端嵌入能力
- Android：JNI 接口（`shoesInit` / `shoesStartTun` / `shoesStop`）
- iOS：C FFI（`shoes_init` / `shoes_start` / `shoes_stop`）
- 支持从移动系统接收原始 TUN fd 和 socket protect 回调

### 5. 高性能运行时
- Tokio 多线程异步运行时（可退化为 current-thread）
- `tikv-jemallocator` 全局内存分配器（除 Windows MSVC / iOS 外）
- Release 构建开启 LTO + strip，musl 目标静态链接

### 6. 配置热重载
- YAML 配置驱动
- 支持多文件合并加载
- 监听配置文件变更后自动重启服务（`--no-reload` 可禁用）

---

## 权衡取舍（Trade-offs）

| 权衡点 | 选择 | 理由与代价 |
|--------|------|-----------|
| **测试策略** | 仅内联 `#[cfg(test)]` 单元测试，无集成测试套件 | 降低 CI 复杂度与构建时间；代价是缺少端到端协议互通验证，需依赖人工/外部测试 |
| **TUN TCP 栈** | 使用 `smoltcp` 在独立 OS 线程中运行，配合阻塞 `select()` 和裸 fd `read`/`write` | 避免 Tokio 与 TUN 包处理的复杂交互，简化 backpressure；代价是跨线程数据交换和 `unsafe` 代码 |
| **TLS 后端** | `aws-lc-rs`（AWS-LC）而非 `ring` | 与 rustls 官方方向一致，支持更多平台；代价是增加对 C/ASM 依赖的编译复杂度 |
| **路由缓存** | LRU 缓存路由决策，但缓存不随规则实时失效 | 高频连接场景性能显著提升；代价是规则变更后可能有短时间的缓存旧决策 |
| ** unsafe 使用** | 限定在 TUN 裸 fd、FFI 边界、少量手动 `Send`/`Sync` impl | 必要的性能与功能实现；代价是需严格维护 fd 生命周期，防止 use-after-close |
| **库/二进制合一** | `lib.rs` 使用 `#![allow(dead_code)]` | 同一 crate 既出 binary 又出 library，避免代码重复；代价是 lib 单独构建时会有未使用代码警告被屏蔽 |
| **DNS over proxy** | TCP/DoT/DoH/DoH3 的 DNS 上游可走任意代理链 | 灵活性极高；代价是 UDP/H3 DNS 仅支持直连（`bind_interface`），不能走代理链 |
| **代码风格** | 默认 `rustfmt` + `clippy`，无自定义配置 | 极简维护；代价是无法针对项目特殊需求微调 lint 规则 |
| **Windows 构建** | CI 中 Windows 构建被注释掉 | 当前未投入精力维护；代价是 Windows 用户需自行解决编译问题 |
