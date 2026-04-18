# shoes 代码文档

本文档从源码层面梳理 `shoes` 的功能模块、设计目标、权衡取舍、核心机制与关键流程。

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_overview.md](./01_overview.md) | 项目定位、设计目标、权衡取舍 |
| [02_modules.md](./02_modules.md) | 功能模块全景与职责划分 |
| [03_mechanisms.md](./03_mechanisms.md) | 核心机制：流抽象、路由、代理链、DNS、TUN、QUIC |
| [04_flows.md](./04_flows.md) | 关键流程与 Mermaid 流程图 |
| [05_vless_vision_reality_client.md](./05_vless_vision_reality_client.md) | VLESS + VISION + REALITY 客户端支持说明与配置参考 |

## 快速定位

- **想理解一条 TCP 连接怎么从入站到出站**：见 [04_flows.md -> TCP 代理链完整流程](./04_flows.md#tcp-代理链完整流程)
- **想理解 TUN/VPN 模式怎么工作**：见 [04_flows.md -> TUN TCP 流程](./04_flows.md#tun-tcp-流程) 与 [TUN UDP 流程](./04_flows.md#tun-udp-流程)
- **想理解配置加载与校验**：见 [04_flows.md -> 配置加载与校验流程](./04_flows.md#配置加载与校验流程)
- **想理解 DNS 查询如何走代理链**：见 [04_flows.md -> DNS 解析与代理链路由流程](./04_flows.md#dns-解析与代理链路由流程)
- **想理解 QUIC 服务器如何分发连接**：见 [04_flows.md -> QUIC 服务器连接处理流程](./04_flows.md#quic-服务器连接处理流程)
