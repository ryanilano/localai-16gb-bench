#!/usr/bin/env bash
# ===========================================================================
# serve-27b-uncensored.sh — locked llama-server launch for the dense winner.
#
# Model : DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF : IQ3_M
#         https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF
#         (dense 27B — Heretic-uncensored finetune of the NEO-CODE line)
# Why   : fastest dense config benched (~40 tg tok/s at IQ3_M) and roomy on 16 GB
#         (~12.7 GB VRAM at low ctx). Longctx sweep (run 2026-07-03_005125)
#         confirmed OK all the way to 80k at ~15.4 GB VRAM (~0.9 GB under the
#         16 GB wall); performance-identical to base NEO-CODE IQ3_M (±0.1 tg).
#
# Flags mirror the benchmark harness exactly (run-quality.sh / models.ini) so
# the running server behaves like the thing that was measured:
#   -ngl 99   full offload (dense — everything on GPU, no --n-cpu-moe)
#   -fa on    flash attention
#   -ctk/-ctv q8_0   quantized KV cache (halves KV vs f16 — how it was benched)
#   --jinja + froggeric template   clean tool-call XML
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Tunables (override via env, e.g. CTX=65536 ./serve-27b-uncensored.sh) --
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
REPO="DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF:IQ3_M"
CTX="${CTX:-81920}"                 # 80k — confirmed OK, ~0.9 GB VRAM headroom (run 2026-07-03_005125)
KV_QUANT="${KV_QUANT:-q8_0}"
THREADS="${THREADS:-8}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
# System prompt is set per-request by the API client (a {role:"system"} message),
# exactly as the benchmark harness does — current llama-server has no server-side
# system-prompt flag, so there is nothing to bake in at launch.
TEMPLATE="${TEMPLATE:-$(cd .. && pwd)/templates/qwen36-froggeric-v20.jinja}"

# --- Preflight --------------------------------------------------------------
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR"; exit 1; }
[ -f "$TEMPLATE" ]     || { echo "chat template not found at $TEMPLATE"; exit 1; }

echo ">>> serving $REPO"
echo "    ctx=$CTX  kv=$KV_QUANT  threads=$THREADS  http://$HOST:$PORT"

exec "$LLAMA_SERVER" \
  -hf "$REPO" --no-mmproj \
  -ngl 99 -fa on \
  -c "$CTX" \
  -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
  -t "$THREADS" \
  --jinja --chat-template-file "$TEMPLATE" \
  --host "$HOST" --port "$PORT"
