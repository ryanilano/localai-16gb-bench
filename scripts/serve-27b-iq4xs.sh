#!/usr/bin/env bash
# ===========================================================================
# serve-27b-iq4xs.sh — locked llama-server launch for the max-dense-quality pick.
#
# Model : DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF : IQ4_XS
#         https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF
#         (dense 27B — Heretic-uncensored NEO-CODE, the higher-fidelity IQ4_XS quant)
# Why   : best dense answer quality on the coding/reasoning pass. IQ4_XS used to
#         be capped near ~16k on 16 GB, but the KV-quant probe (2026-07-03_151915)
#         showed the wall is KV-bound: at q4_0 KV this config reaches depth 49152
#         (~49k) at ~15.8 GB VRAM, and the q4_0 coherence A/B (2026-07-03_170204)
#         found no answer-quality loss vs q8_0. So this is the config to run when
#         you want the best dense quality AND real context (up to ~49k).
#
#         Tradeoff vs the IQ3_M winner (serve-27b-uncensored.sh): higher quality
#         per token, but ~49k max context instead of 80k, and it leans on q4_0 KV.
#
# Flags mirror the benchmark harness exactly (run-quality.sh / configs.sh) so the
# running server behaves like the thing that was measured:
#   -ngl 99          full offload (dense — everything on GPU, no --n-cpu-moe)
#   -fa on           flash attention
#   -ctk/-ctv q4_0   aggressive KV cache — the lever that unlocks ~49k (see above).
#                    q8_0 here would cap you near ~16k; f16 far sooner.
#   --jinja + froggeric template   clean tool-call XML
#
# CAVEAT: the q4_0 coherence pass used short (<=16k) prompts, so coherence is
# proven for the KV quant itself, not yet for answers written with the cache
# actually filled to ~49k. If deep-context answers ever look degraded, drop
# KV_QUANT=q8_0 (and CTX to ~16384) or switch to the IQ3_M script.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Tunables (override via env, e.g. CTX=32768 ./serve-27b-iq4xs.sh) -------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
REPO="DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF:IQ4_XS"
CTX="${CTX:-49152}"                 # 49k — validated OK at q4_0 (~15.8 GB, near the wall);
                                    # drop to 32768 for comfortable headroom.
KV_QUANT="${KV_QUANT:-q4_0}"        # q4_0 is what unlocks 49k; q8_0 caps ~16k.
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
echo "    note: KV=q4_0 is what makes ~49k fit on 16 GB — VRAM sits near the wall at max ctx."

exec "$LLAMA_SERVER" \
  -hf "$REPO" --no-mmproj \
  -ngl 99 -fa on \
  -c "$CTX" \
  -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
  -t "$THREADS" \
  --jinja --chat-template-file "$TEMPLATE" \
  --host "$HOST" --port "$PORT"
