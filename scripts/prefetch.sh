#!/usr/bin/env bash
# ===========================================================================
# prefetch.sh — download every model in configs.sh BEFORE the sweep, so the
# (model-neutral: works for any GGUF repo in the CONFIGS matrix, not just Qwen.)
# bench/quality runs never stall on a download.
#
# Uses llama.cpp's own -hf resolver (not huggingface-cli) so the cache layout
# matches exactly what run-bench.sh / run-quality.sh expect — whatever you set
# for LLAMA_CACHE / HF_HOME is honored, with zero chance of a mismatch.
# Each model is loaded CPU-only (mmap, -ngl 0) for a 1-token pass: this pulls
# the full file to disk and integrity-checks it, without needing the GPU and
# without loading all weights into RAM (mmap only faults in what it touches).
# Already-cached models are skipped instantly.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh

[ -x "$LLAMA_CLI" ] || { echo "llama-cli not found at $LLAMA_CLI — set LLAMA_DIR in configs.sh"; exit 1; }
mkdir -p "$OUTDIR"

echo "Cache target: ${LLAMA_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
echo "Prefetching ${#CONFIGS[@]} models (this is the slow part — run it once)."
echo "Tip: if you hit HF rate limits, 'export HF_TOKEN=hf_xxx' first."
echo

fail=0; n=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl <<< "$entry"   # sys/tmpl unused here; read them so they don't fold into $type
  n=$((n+1)); echo "[$n/${#CONFIGS[@]}] $label  ($repo)"
  if "$LLAMA_CLI" -hf "$repo" -ngl 0 -n 1 -p "ok" -no-cnv --no-warmup \
       > "$OUTDIR/prefetch_${label}.log" 2>&1; then
    echo "    OK (cached + verified)"
  else
    echo "    FAILED — see $OUTDIR/prefetch_${label}.log (bad quant tag? network? disk full?)"
    fail=$((fail+1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "All ${#CONFIGS[@]} models cached. You can now run ./run-bench.sh and ./run-quality.sh"
  echo "offline-fast — they'll load straight from cache."
else
  echo "$fail model(s) failed. Fix those lines in configs.sh and re-run; cached ones are skipped."
  exit 1
fi
