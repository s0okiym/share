# Xray-core Agent Guide

> This file is intended for AI coding agents. It assumes no prior knowledge of the project.

## Project Overview

**Xray-core** is a high-performance network proxy platform written in Go, originating from the XTLS protocol. It was forked from v2fly/v2ray-core and has since accumulated extensive enhancements in performance, security, and functionality.

The project provides a unified platform for building proxies, supporting multiple inbound/outbound protocols, intelligent routing, traffic statistics, and various transport-layer security mechanisms.

**Repository**: `github.com/xtls/xray-core`  
**Go Version**: 1.26 (see `go.mod`)  
**License**: Mozilla Public License Version 2.0

### Core Capabilities

- **Proxy Protocols**: VLESS, VMess, Trojan, Shadowsocks (including Shadowsocks 2022), Socks, HTTP, WireGuard, Hysteria, Dokodemo-door, Freedom, Blackhole, Loopback, DNS
- **Transports**: TCP, mKCP, WebSocket, gRPC, SplitHTTP (XHTTP), HTTPUpgrade
- **Security Layers**: TLS, uTLS (browser fingerprinting), REALITY, XTLS Vision
- **Infrastructure**: DNS (with FakeDNS), Router, Dispatcher, Policy Manager, Stats Manager, Observatory, Mux, XUDP

## Technology Stack

- **Language**: Go 1.26+
- **Configuration**: Protocol Buffers (protobuf) for internal representation; JSON, TOML, YAML for user-facing config files
- **RPC**: gRPC for API/command interfaces
- **Key Dependencies** (see `go.mod`):
  - `github.com/apernet/quic-go` — QUIC transport
  - `github.com/refraction-networking/utls` — TLS fingerprint spoofing
  - `github.com/xtls/reality` — REALITY transport layer
  - `github.com/gorilla/websocket` — WebSocket transport
  - `github.com/miekg/dns` — DNS resolution
  - `github.com/vishvananda/netlink` — Linux netlink sockets
  - `gvisor.dev/gvisor` — Userspace TCP/IP (for TUN/WireGuard)
  - `google.golang.org/protobuf` — Protobuf code generation
  - `github.com/golang/mock` — Mock generation for tests

## Build and Test Commands

### Build

```bash
# Standard build (Linux/macOS)
CGO_ENABLED=0 go build -o xray -trimpath -buildvcs=false -ldflags="-s -w -buildid=" -v ./main

# Windows
$env:CGO_ENABLED=0
go build -o xray.exe -trimpath -buildvcs=false -ldflags="-s -w -buildid=" -v ./main

# Reproducible release build (set build stamp)
CGO_ENABLED=0 go build -o xray -trimpath -buildvcs=false -gcflags="all=-l=4" -ldflags="-X github.com/xtls/xray-core/core.build=REPLACE -s -w -buildid=" -v ./main

# For 32-bit MIPS/MIPSLE, use -gcflags="-l=4" instead of "all=-l=4"
```

### Test

```bash
# Run all tests (requires ~1 hour timeout for scenario tests)
go test -timeout 1h -v ./...

# Run unit tests only (faster)
go test -v ./common/... ./core/... ./features/... ./app/...

# Run scenario/integration tests
go test -timeout 1h -v ./testing/scenarios/...
```

> **Note**: Some tests require `geoip.dat` and `geosite.dat` data files in a `resources/` directory. The CI workflow downloads/caches these. If tests fail due to missing geodata, create `resources/` and place the `.dat` files there.

### Code Generation

```bash
# Regenerate protobuf Go files (requires protoc, protoc-gen-go, protoc-gen-go-grpc)
go generate ./core/proto.go
# Equivalent to:
go install -v google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install -v google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
go run ./infra/vprotogen/main.go -pwd ./..

# Regenerate mocks
go generate ./core/mocks.go

# Format code (requires gofumpt and gci)
go generate ./core/format.go
# Equivalent to:
go install -v github.com/daixiang0/gci@latest
go install -v mvdan.cc/gofumpt@latest
go run ./infra/vformat/main.go -pwd ./..
```

## Code Organization

The codebase follows a layered architecture:

```
├── core/               # Core Instance, config loading, dependency resolution, protobuf definitions
├── features/           # Feature interfaces (DNS, inbound, outbound, policy, routing, stats)
├── app/                # Feature implementations (dispatcher, router, DNS, policy, stats, etc.)
├── common/             # Shared utilities (buf, net, crypto, protocol sniffers, mux, xudp, etc.)
├── proxy/              # Proxy protocol implementations (inbound + outbound handlers)
├── transport/          # Transport layer (internet dialers, TCP/UDP/WS/gRPC/KCP/REALITY/TLS)
├── infra/              # Config parsing helpers and development tools (vformat, vprotogen)
├── main/               # CLI entry point, command framework, config loaders (json/toml/yaml), distro registration
└── testing/            # Test utilities, mock servers, integration scenarios
```

### Key Architectural Concepts

- **Instance** (`core/xray.go`): The central container that holds all `Feature`s and resolves inter-feature dependencies via reflection-based callbacks.
- **Feature** (`features/feature.go`): Any component implementing `common.HasType` + `common.Runnable` (Start/Close). Examples: DNS client, Router, Dispatcher, Inbound/Outbound Manager.
- **Dispatcher** (`app/dispatcher/`): Receives traffic from inbound handlers, optionally sniffs protocols, consults the router, and dispatches to outbound handlers.
- **Pipe** (`transport/pipe/`): In-memory buffered stream abstraction used internally instead of direct `net.Conn` passing.
- **Buffer Pool** (`common/buf/`): 8KB pooled buffers (`Size = 8192`) to reduce GC pressure.

## Module Divisions

### `proxy/*` — Protocol Handlers
Each protocol has its own package, typically with `inbound/` and `outbound/` subpackages:
- `proxy/vless/` — VLESS protocol (stateless, UUID-based, supports XTLS Vision)
- `proxy/vmess/` — VMess protocol (AEAD encryption, time-based auth)
- `proxy/trojan/` — Trojan protocol
- `proxy/shadowsocks/` / `proxy/shadowsocks_2022/` — Shadowsocks variants
- `proxy/wireguard/` — WireGuard inbound/outbound
- `proxy/hysteria/` — Hysteria protocol
- `proxy/freedom/` — Direct outbound (no proxy)
- `proxy/dokodemo/` — Transparent proxy inbound
- `proxy/socks/` / `proxy/http/` — Standard proxy protocols

### `transport/internet/*` — Network Transports
- `tcp/`, `udp/` — Base dialers/listeners
- `websocket/`, `grpc/`, `httpupgrade/`, `splithttp/` — HTTP-based transports
- `kcp/` — mKCP (reliable UDP)
- `tls/`, `reality/` — Security wrappers
- `headers/` — Transport header modifiers

### `app/*` — Internal Services
- `dispatcher/` — Traffic dispatching
- `router/` — Rule-based routing (domain, IP, protocol, etc.)
- `dns/` — DNS client with caching and FakeDNS support
- `proxyman/inbound/` & `proxyman/outbound/` — Handler lifecycle managers
- `policy/` — Connection timeout/buffer limits
- `stats/` — Traffic and connection counters
- `observatory/` — Outbound health monitoring
- `commander/` — gRPC API server
- `log/` — Structured logging

## Code Style Guidelines

The project enforces the following formatting rules:

- **Formatter**: `gofumpt` (stricter than `gofmt`)
- **Import Organizer**: `gci`
- **Generated files excluded**: `*.pb.go`, `testing/mocks/*`, `main/distro/all/all.go`

Run the formatter before committing:

```bash
go run ./infra/vformat/main.go -pwd ./..
```

### General Conventions

- Use `common.Must(err)` and `common.Must2(n, err)` for fatal initialization errors.
- Use `errors.New("...").Base(err)` for error chaining (custom error package in `common/errors/`).
- Prefer `buf.Buffer` / `buf.MultiBuffer` over raw `[]byte` for I/O.
- Register handlers/transports via blank imports in `main/distro/all/all.go`.
- Protocol config structs use protobuf definitions (`.proto` files) with generated `.pb.go`.

## Testing Instructions

### Test Structure

- **Unit tests**: Co-located with source files (`*_test.go` in the same package).
- **Integration/scenario tests**: `testing/scenarios/` — These spin up real Xray processes, configure them via protobuf configs, and test end-to-end proxy behavior.
- **Mocks**: Generated in `testing/mocks/` via `golang/mock`.

### Running Scenario Tests

Scenario tests (`testing/scenarios/*_test.go`) build a temporary Xray binary and launch it as a subprocess. They require:

1. A working Go toolchain.
2. Sufficient timeout (`-timeout 1h` recommended).
3. Patience — they test real TCP/UDP/WS/gRPC/TLS connections.

Example helpers in `testing/scenarios/common.go`:
- `InitializeServerConfig(config *core.Config)` — builds and starts a test Xray instance.
- `CloseAllServers(servers)` — cleanly shuts down test instances.
- `testTCPConn2(conn, payloadSize, timeout)` — verifies a connection with XOR echo.

### Writing New Tests

For protocol tests, the typical pattern is:
1. Start a raw TCP/UDP test server (`testing/servers/tcp`, `testing/servers/udp`).
2. Construct `core.Config` with inbound + outbound + routing rules using protobuf structs.
3. Call `InitializeServerConfig(config)`.
4. Dial through the proxy and verify data integrity.
5. Call `CloseServer` in `defer`.

## Configuration System

Xray accepts multiple config formats:

- **JSON** (default): `config.json` or `config.jsonc`
- **TOML**: `config.toml`
- **YAML**: `config.yaml`, `config.yml`
- **Protobuf**: `.pb` (binary protobuf, mostly internal)

Config loading order:
1. `-c` / `-config` flags (multiple allowed)
2. `-confdir` flag (loads all matching files in directory)
3. `config.{json,jsonc,toml,yaml,yml}` in working directory
4. Platform-specific config path (environment variable)
5. `stdin:` (reads from standard input)

### Important Protobuf Conventions

- All `.proto` files live alongside their Go packages.
- Generated `.pb.go` files must share the same version header. The CI checks this with `check-proto`.
- After modifying `.proto` files, run `go generate ./core/proto.go`.

## Security Considerations

- **Do not** publish security vulnerabilities or protocol-identification issues publicly before a fix is released. Use GitHub's "[Report a vulnerability](https://github.com/XTLS/Xray-core/security/advisories/new)" feature.
- `AllowInsecure` TLS settings exist for testing but should never be enabled in production.
- REALITY and XTLS Vision are designed to resist traffic analysis and fingerprinting; changes to these modules should be reviewed with extra care.
- The project handles raw network traffic and cryptographic material; memory safety and constant-time operations matter.

## Deployment Processes

### GitHub Actions Workflows

- **`.github/workflows/test.yml`** — Runs on every push/PR:
  - Checks geodata assets (`geoip.dat`, `geosite.dat`)
  - Verifies protobuf version header consistency
  - Runs `go test -timeout 1h -v ./...` on Windows, Ubuntu, and macOS

- **`.github/workflows/release.yml`** — Builds cross-platform binaries:
  - Targets: Windows, Linux, macOS, FreeBSD, OpenBSD, Android
  - Architectures: amd64, 386, arm64, armv5/6/7, mips/mipsle/mips64/mips64le, ppc64/ppc64le, riscv64, loong64, s390x
  - Packages binaries with geodata and wintun DLL (Windows) into ZIP archives
  - Attaches checksums (MD5, SHA1, SHA256, SHA512)

- **`.github/workflows/docker.yml`** — Builds and pushes multi-arch Docker images to `ghcr.io/xtls/xray-core` on releases.

- **`.github/workflows/scheduled-assets-update.yml`** — Updates `geoip.dat`, `geosite.dat`, and wintun DLLs via scheduled runs.

### Docker

Official images use distroless base (`gcr.io/distroless/static:nonroot`):

```dockerfile
# Multi-stage build; see .github/docker/Dockerfile
FROM --platform=$BUILDPLATFORM golang:latest AS build
# ...
FROM gcr.io/distroless/static:nonroot
ENTRYPOINT ["/usr/local/bin/xray"]
CMD ["-confdir", "/usr/local/etc/xray/"]
```

Volumes:
- `/usr/local/etc/xray` — configuration files
- `/var/log/xray` — log output
- `/usr/local/share/xray` — geodata files

## Common Development Tasks

### Adding a New Proxy Protocol

1. Create a package under `proxy/<protocol>/`.
2. Implement `proxy.Inbound` and/or `proxy.Outbound` interfaces.
3. Add config protobuf definitions (`config.proto`) and generate `.pb.go`.
4. Register handlers in `main/distro/all/all.go` with blank imports.
5. Add scenario tests in `testing/scenarios/`.
6. Run `go run ./infra/vformat/main.go -pwd ./..` before committing.

### Adding a New Transport

1. Create a package under `transport/internet/<transport>/`.
2. Implement dialer/listener conforming to `internet.Dialer` / `internet.Listener` patterns.
3. Register in `main/distro/all/all.go`.
4. Add tests and update config schema in `infra/conf/` if needed.

### Modifying Protobuf Definitions

1. Edit the `.proto` file.
2. Run `go generate ./core/proto.go`.
3. Ensure all `.pb.go` files have consistent headers (CI `check-proto` will verify).
4. Commit both `.proto` and `.pb.go` changes.

## Versioning

Version is hardcoded in `core/core.go`:

```go
var (
    Version_x byte = 26
    Version_y byte = 3
    Version_z byte = 27
)
```

The build stamp (`core.build`) is injected at link time via `-ldflags` for official releases, or inferred from VCS info for local builds.

## Useful References

- Project docs and examples: [XTLS.github.io](https://xtls.github.io)
- Configuration examples: [XTLS/Xray-examples](https://github.com/XTLS/Xray-examples)
- Community Telegram: [Project X](https://t.me/projectXray)
- Security reports: Use GitHub Security Advisories (see `SECURITY.md`)
