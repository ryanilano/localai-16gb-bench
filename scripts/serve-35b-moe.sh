#!/usr/bin/env bash
# ===========================================================================
# serve-moe.sh — locked llama-server launch for the MoE benchmark winner.
#
# Model : unsloth/Qwen3.6-35B-A3B-GGUF : UD-Q3_K_M  (35B MoE, A3B active)
# Why   : fastest config in the whole sweep (~58 tg tok/s at low depth) and the
#         roomiest — stayed OK out to depth 261632 (~255k) at only ~6 GB VRAM.
#         Idle experts live in system RAM (--n-cpu-moe), so the real ceiling is
#         RAM, not VRAM; peaked at ~3.8 GB RAM at 255k on the reference box.
#
# Flags mirror the benchmark harness exactly (run-quality.sh / configs.sh):
#   -ngl 99          attention + active tensors on GPU
#   --n-cpu-moe 99   offload idle expert FFN tensors to system RAM (the MoE win)
#   -fa on           flash attention
#   -ctk/-ctv q8_0   quantized KV cache (how it was benched)
#   --jinja + froggeric template   clean tool-call XML
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Tunables (override via env, e.g. CTX=131072 ./serve-moe.sh) -----------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
REPO="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M"
CTX="${CTX:-262144}"                # 256k — cleared 255k in the deep sweep, no FAIL
NCMOE="${NCMOE:-99}"                # idle experts -> RAM; lower it to pull experts back
                                    # onto the GPU (more speed) until VRAM ~fills
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
echo "    ctx=$CTX  n-cpu-moe=$NCMOE  kv=$KV_QUANT  threads=$THREADS  http://$HOST:$PORT"
echo "    note: this is RAM-bound, not VRAM-bound — watch system RAM at deep context."

exec "$LLAMA_SERVER" \
  -hf "$REPO" --no-mmproj \
  -ngl 99 --n-cpu-moe "$NCMOE" -fa on \
  -c "$CTX" \
  -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
  -t "$THREADS" \
  --jinja --chat-template-file "$TEMPLATE" \
  --system-prompt "$SYS_PROMPT" \
  --host "$HOST" --port "$PORT"
