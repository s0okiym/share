# impl_kimi

本目录包含用 Rust 实现 VLESS + REALITY + XTLS Vision 代理客户端的方案设计文档。

## 文件说明

- `DESIGN.md` — 完整的实现方案设计文档，涵盖：
  - 整体架构设计
  - SOCKS5 / HTTP Mixed 入站代理设计
  - VLESS 协议出站实现细节
  - XTLS Vision Flow 工作原理
  - REALITY 传输层客户端握手流程
  - Rust 技术选型与依赖推荐
  - 分阶段实现规划
  - 参考项目与资源

## 快速开始

阅读 `DESIGN.md` 获取完整的实现方案。建议优先参考 [cfal/shoes](https://github.com/cfal/shoes) 项目，这是一个已经用 Rust 实现了完整协议栈（SOCKS5/HTTP Mixed + VLESS + REALITY + Vision）的开源项目。
