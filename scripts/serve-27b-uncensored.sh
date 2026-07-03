#!/usr/bin/env bash
# ===========================================================================
# serve-neo-code.sh — locked llama-server launch for the benchmark winner.
#
# Model : DavidAU/Qwen3.6-27B-NEO-CODE-Di-IMatrix-MAX-GGUF : IQ3_M  (dense 27B)
# Why   : fastest dense config in the sweep (~40 tg tok/s) AND the roomiest —
#         stayed OK out to depth 81920 (80k) at ~15.4 GB VRAM on a 16 GB card.
#         Config note pins its safe ceiling at ~85-88k, so 80k leaves margin.
#
# Flags mirror the benchmark harness exactly (run-quality.sh / configs.sh) so
# the running server behaves like the thing that was measured:
#   -ngl 99   full offload (dense — everything on GPU, no --n-cpu-moe)
#   -fa on    flash attention
#   -ctk/-ctv q8_0   quantized KV cache (halves KV vs f16 — how it was benched)
#   --jinja + froggeric template   clean tool-call XML
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Tunables (override via env, e.g. CTX=65536 ./serve-neo-code.sh) --------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
REPO="DavidAU/Qwen3.6-27B-NEO-CODE-Di-IMatrix-MAX-GGUF:IQ3_M"
CTX="${CTX:-81920}"                 # 80k — the benchmarked fit ceiling for this quant
KV_QUANT="${KV_QUANT:-q8_0}"
THREADS="${THREADS:-8}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
SYS_PROMPT="${SYS_PROMPT:-You are Qwen, created by Alibaba Cloud. You are a helpful assistant.}"
TEMPLATE="${TEMPLATE:-$(cd .. && pwd)/templates/qwen36-froggeric-v20.jinja}"

# --- Preflight --------------------------------------------------------------
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR"; exit 1; }
[ -f "$TEMPLATE" ]     || { echo "chat template not found at $TEMPLATE"; exit 1; }

echo ">>> serving $REPO"
echo "    ctx=$CTX  kv=$KV_QUANT  threads=$THREADS  http://$HOST:$PORT"

exec "$LLAMA_SERVER" \
  -hf "$REPO" \
  -ngl 99 -fa on \
  -c "$CTX" \
  -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
  -t "$THREADS" \
  --jinja --chat-template-file "$TEMPLATE" \
  --system-prompt "$SYS_PROMPT" \
  --host "$HOST" --port "$PORT"
