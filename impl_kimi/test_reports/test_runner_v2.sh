#!/bin/bash
set -eu

echo "========================================"
echo "  xray-rust-client 全面测试报告"
echo "========================================"
echo ""

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] $name ... "
    if eval "$cmd" > /tmp/test_${TOTAL}.log 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS + 1))
        return 0
    else
        echo "❌ FAIL"
        FAIL=$((FAIL + 1))
        cat /tmp/test_${TOTAL}.log
        return 1
    fi
}

# 1. 单元测试
echo "### 1. 单元测试"
echo ""
cargo test --quiet 2>&1 | tee /tmp/unit_test.log
if grep -q "test result: ok" /tmp/unit_test.log; then
    UNIT_PASS=$(grep "test result:" /tmp/unit_test.log | sed 's/.*ok\. \([0-9]*\) passed.*/\1/')
    echo "✅ 单元测试全部通过: $UNIT_PASS 个测试"
else
    echo "❌ 单元测试失败"
    FAIL=$((FAIL + 1))
fi
echo ""

# 2. 编译测试
echo "### 2. 编译测试"
echo ""
run_test "Release 模式编译" "cargo build --release --quiet"
echo ""

# 3. 启动客户端进行端到端测试
echo "### 3. 端到端集成测试"
echo ""
pkill -f "xray run -config /tmp/xray_backend" 2>/dev/null || true
sleep 1

# 启动客户端（使用 setsid 脱离终端）
cd /root/kimi/Xray-core/impl_kimi
export RUST_LOG=info
setsid ./target/release/xray-rust-client --config config.toml > /tmp/client.log 2>&1 &
CLIENT_PGID=$!
sleep 4

echo "客户端启动 PGID: $CLIENT_PGID"
echo "客户端日志:"
head -10 /tmp/client.log
ss -tlnp | grep 1080 || true
echo ""

# 3.1 SOCKS5 代理测试
echo "--- SOCKS5 代理测试 ---"
run_test "SOCKS5 HTTP 访问 httpbin.org/get" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get | grep -q 'origin'"

run_test "SOCKS5 HTTPS 访问 httpbin.org/get" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 https://httpbin.org/get | grep -q 'origin'"

run_test "SOCKS5 HTTP 访问 www.google.com" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://www.google.com | grep -i 'html' > /dev/null"

run_test "SOCKS5 HTTPS 访问 www.google.com" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 https://www.google.com | grep -i 'html' > /dev/null"

echo ""

# 3.2 HTTP 代理测试
echo "--- HTTP 代理测试 ---"
run_test "HTTP 代理 HTTP 访问 httpbin.org/get" \
    "curl -s --max-time 20 --proxy http://127.0.0.1:1080 http://httpbin.org/get | grep -q 'origin'"

run_test "HTTP 代理 HTTPS 访问 httpbin.org/get" \
    "curl -s --max-time 20 --proxy http://127.0.0.1:1080 https://httpbin.org/get | grep -q 'origin'"

echo ""

# 3.3 不同目标类型测试
echo "--- 不同目标类型测试 ---"
run_test "SOCKS5 IPv4 目标 1.1.1.1:80" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://1.1.1.1 | grep -qi 'cloudflare'"

run_test "SOCKS5 域名目标 cloudflare.com" \
    "curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://cloudflare.com | grep -qi 'cloudflare'"

echo ""

# 3.4 并发测试
echo "--- 并发连接测试 ---"
for i in 1 2 3; do
    (curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get > /tmp/concurrent_$i.log 2>&1) &
done

# 最多等待25秒
for i in $(seq 1 25); do
    sleep 1
    DONE=0
    for j in 1 2 3; do
        if [ -f /tmp/concurrent_$j.log ] && grep -q 'origin' /tmp/concurrent_$j.log 2>/dev/null; then
            DONE=$((DONE + 1))
        fi
    done
    if [ "$DONE" -eq 3 ]; then
        break
    fi
done

TOTAL=$((TOTAL + 1))
CONCURRENT_PASS=0
for i in 1 2 3; do
    if [ -f /tmp/concurrent_$i.log ] && grep -q 'origin' /tmp/concurrent_$i.log 2>/dev/null; then
        CONCURRENT_PASS=$((CONCURRENT_PASS + 1))
    fi
done

if [ "$CONCURRENT_PASS" -eq 3 ]; then
    echo "[$TOTAL] 并发 3 连接同时访问 ... ✅ PASS (3/3)"
    PASS=$((PASS + 1))
else
    echo "[$TOTAL] 并发 3 连接同时访问 ... ❌ FAIL ($CONCURRENT_PASS/3)"
    FAIL=$((FAIL + 1))
fi
echo ""

# 4. 端口监听检查
echo "### 4. 端口监听检查"
echo ""
run_test "Rust 入站监听 127.0.0.1:1080" \
    "ss -tlnp | grep '127.0.0.1:1080' | grep -q 'xray-rust'"

echo ""

# 5. 客户端日志检查
echo "### 5. 客户端日志检查"
echo ""
if grep -q "Mixed inbound listening on" /tmp/client.log; then
    echo "✅ 入站代理启动日志正确"
    PASS=$((PASS + 1))
else
    echo "❌ 入站代理启动日志缺失"
    FAIL=$((FAIL + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""

# 6. 性能基准测试
echo "### 6. 性能基准测试"
echo ""
START_TIME=$(date +%s%N)
for i in $(seq 1 5); do
    curl -s --max-time 20 --proxy socks5h://127.0.0.1:1080 http://httpbin.org/get > /dev/null 2>&1
done
END_TIME=$(date +%s%N)
DURATION_MS=$(((END_TIME - START_TIME) / 1000000))
AVG_MS=$((DURATION_MS / 5))
echo "5 次 HTTP 请求耗时: ${DURATION_MS}ms (平均 ${AVG_MS}ms/请求)"
if [ "$AVG_MS" -lt 5000 ]; then
    echo "✅ 性能可接受 (< 5s/请求)"
    PASS=$((PASS + 1))
else
    echo "⚠️ 性能较慢"
fi
TOTAL=$((TOTAL + 1))
echo ""

# 清理
kill -- -$CLIENT_PGID 2>/dev/null || true
pkill -f "xray run -config /tmp/xray_backend" 2>/dev/null || true
wait 2>/dev/null || true

echo "========================================"
echo "  测试总结"
echo "========================================"
echo "总测试数: $TOTAL"
echo "通过:     $PASS ✅"
echo "失败:     $FAIL ❌"
echo "通过率:   $(( PASS * 100 / TOTAL ))%"
echo "========================================"

if [ "$FAIL" -eq 0 ]; then
    echo "所有测试全部通过！"
    exit 0
else
    echo "存在失败的测试，请检查日志。"
    exit 1
fi
