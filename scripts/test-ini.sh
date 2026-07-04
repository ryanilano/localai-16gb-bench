#!/usr/bin/env bash
# test-ini.sh — standalone assertions for ini.sh against a fixture INI.
# Run: bash test-ini.sh   (exit 0 + "ALL PASS" or nonzero + first failure)
set -uo pipefail
cd "$(dirname "$0")"

FIX="$(mktemp)"
trap 'rm -f "$FIX"' EXIT
cat > "$FIX" <<'EOF'
; fixture
[*]
type = dense
t = 8
ngl = 99
fa = on
n-cpu-moe = 99

[dense-model]
hf = org/dense-GGUF:Q4
sys = You are a test.
; override a default
t = 12
; zero must NOT be skipped
ngl = 0
bench.p = 512
quality.b = 2048

[moe-model]
hf = org/moe-GGUF:UD-IQ4
type = moe
template = templates/fix.jinja
# empty value skips the key
fa =
EOF

INI_FILE="$FIX"
source ./ini.sh

fail=0
check() {  # $1=name $2=expected $3=actual
  if [ "$2" = "$3" ]; then echo "PASS  $1"
  else echo "FAIL  $1"; echo "  expected: [$2]"; echo "  actual:   [$3]"; fail=1; fi
}

check "sections lists labels, not [*]" \
  "dense-model
moe-model" "$(ini_sections)"

check "get: section overrides default"      "12"    "$(ini_get dense-model t)"
check "get: falls back to [*]"              "8"     "$(ini_get moe-model t)"
check "get: reserved key hf"                "org/dense-GGUF:Q4" "$(ini_get dense-model hf)"
check "get: type default is dense"          "dense" "$(ini_get dense-model type)"
check "get: type override is moe"           "moe"   "$(ini_get moe-model type)"
check "get: sys only where set"             ""      "$(ini_get moe-model sys)"
check "get: template only where set"        ""      "$(ini_get dense-model template)"
check "get: unset key is empty"             ""      "$(ini_get dense-model nope)"

# flags: dense/bench — reserved keys absent, override applied, 0 kept,
# n-cpu-moe gated out, quality.b excluded, bench.p included
mapfile -t F < <(ini_flags dense-model bench)
check "flags dense/bench" \
  "-t 12 -ngl 0 -fa on -p 512" "${F[*]}"

# flags: dense/quality — bench.p gone, quality.b present
mapfile -t F < <(ini_flags dense-model quality)
check "flags dense/quality" \
  "-t 12 -ngl 0 -fa on -b 2048" "${F[*]}"

# flags: moe/bench — n-cpu-moe passes the gate; empty fa skipped
mapfile -t F < <(ini_flags moe-model bench)
check "flags moe/bench" \
  "-t 8 -ngl 99 --n-cpu-moe 99" "${F[*]}"

[ "$fail" -eq 0 ] && echo "ALL PASS" || { echo "FAILURES"; exit 1; }
