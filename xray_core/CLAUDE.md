# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

Xray-core is a high-performance network proxy platform written in Go 1.26, forked from v2fly/v2ray-core. It supports multiple proxy protocols (VLESS, VMess, Trojan, Shadowsocks, WireGuard, Hysteria, etc.), advanced transports (TCP, mKCP, WebSocket, gRPC, SplitHTTP/XHTTP, HTTPUpgrade), and security layers (TLS, uTLS, REALITY, XTLS Vision).

For a comprehensive guide, see `AGENTS.md`.

## Commands

### Build

```bash
CGO_ENABLED=0 go build -o xray -trimpath -buildvcs=false -ldflags="-s -w -buildid=" -v ./main
```

### Test

```bash
# All tests (some integration tests take ~1 hour)
go test -timeout 1h -v ./...

# Unit tests only (faster)
go test -v ./common/... ./core/... ./features/... ./app/...

# Integration/scenario tests
go test -timeout 1h -v ./testing/scenarios/...
```

Some tests require `geoip.dat` and `geosite.dat` in a `resources/` directory.

### Code Generation

```bash
# Regenerate protobuf files (requires protoc toolchain)
go generate ./core/proto.go

# Regenerate mocks
go generate ./core/mocks.go

# Format code (gofumpt + gci)
go run ./infra/vformat/main.go -pwd ./..
```

### Run

```bash
./xray run -c config.json          # Start with config file
./xray run -confdir /etc/xray/     # Load all configs from directory
./xray run -test                   # Validate config without starting
./xray run -dump                   # Print merged config only
```

## Architecture

### Layered Structure

| Directory | Purpose |
|-----------|---------|
| `main/` | CLI entry point, command framework, config loaders (json/toml/yaml), distro registration |
| `core/` | `Instance` lifecycle, feature dependency resolution, protobuf config definitions |
| `features/` | Feature interface contracts (DNS, inbound, outbound, policy, routing, stats) |
| `app/` | Feature implementations (dispatcher, router, DNS, policy, stats, observatory, commander) |
| `proxy/` | Proxy protocol handlers (VLESS, VMess, Trojan, Shadowsocks, WireGuard, Hysteria, etc.) |
| `transport/` | Network transport layer (TCP/UDP, WebSocket, gRPC, KCP, TLS, REALITY) |
| `common/` | Shared utilities (buffer pool, networking, crypto, session, mux, xudp) |
| `infra/` | Config parsing helpers (`infra/conf/`) and dev tools (vformat, vprotogen) |
| `testing/` | Test infrastructure, mock servers, integration scenario tests |

### Key Concepts

- **Instance** (`core/xray.go`): Central container managing all Features. Features are registered via `AddFeature()` and resolved via `RequireFeatures()` using reflection-based callbacks.
- **Feature** (`features/feature.go`): Components implementing `HasType` + `Runnable` (Start/Close). Examples: DNS client, Router, Dispatcher, Inbound/Outbound managers.
- **Dispatcher** (`app/dispatcher/`): Receives traffic from inbounds, performs protocol sniffing, consults the router, and dispatches to outbounds.
- **Registration**: All features, proxies, and transports self-register in `init()` functions. `main/distro/all/all.go` blank-imports everything for the full build. To add a new protocol, create the package and add a blank import there.
- **Data flow**: Internal data flows through `transport/pipe/` (buffered in-memory streams), not direct `net.Conn` passing. A `transport.Link` pairs uplink + downlink pipes.

### Important Conventions

- **Buffers**: Use `common/buf/` with 8KB pooled buffers (`buf.Buffer` / `buf.MultiBuffer`), never raw `[]byte` for I/O.
- **Errors**: Use `common/errors/` for chaining: `errors.New("msg").Base(err)`. Use `common.Must(err)` for fatal initialization errors.
- **Config**: Protobuf is the internal representation. User-facing configs (JSON/TOML/YAML) are parsed by `infra/conf/` and converted to protobuf. Master config schema: `core/config.proto`.
- **Formatting**: `gofumpt` + `gci` are required. Run `go run ./infra/vformat/main.go -pwd ./..` before committing.

### Versioning

Version is hardcoded in `core/core.go` as `Version_x`, `Version_y`, `Version_z`. The build stamp is injected at link time via `-ldflags`.

## Security

- Use GitHub Security Advisories for vulnerability reports (see `SECURITY.md`).
- `AllowInsecure` TLS settings are for testing only.
- REALITY and XTLS Vision modules require extra review — they're designed to resist traffic analysis.
