#!/usr/bin/env bash
# ===========================================================================
# prefetch.sh — download every model in configs.sh BEFORE the sweep, so the
# (model-neutral: works for any GGUF repo in the CONFIGS matrix, not just Qwen.)
# bench/quality runs never stall on a download.
#
# Uses llama.cpp's own -hf resolver (not huggingface-cli) so the cache layout
# matches exactly what run-bench.sh / run-quality.sh expect — whatever you set
# for LLAMA_CACHE / HF_HOME is honored, with zero chance of a mismatch.
# Each model gets a minimal CPU-only llama-bench load (-ngl 0, tiny -p/-n): the
# -hf download pulls the full file to disk and the load confirms it parses,
# without needing the GPU. We use llama-bench, NOT llama-cli, because newer
# llama.cpp builds dropped headless completion from llama-cli (it now defaults to
# interactive conversation mode and rejects -no-cnv); llama-bench is always
# non-interactive. Already-cached models are skipped instantly.
#
# Downloads run in PARALLEL, up to PREFETCH_JOBS at a time (default 3), so a big
# matrix isn't gated on one file at a time. Set PREFETCH_JOBS in configs.sh (or
# `PREFETCH_JOBS=4 ./prefetch.sh`). 2-4 is usually fastest; more just thrashes
# the disk and invites HF rate limits.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
PREFETCH_JOBS="${PREFETCH_JOBS:-3}"   # fallback if configs.sh predates this knob

[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not found at $LLAMA_BENCH — set LLAMA_DIR in configs.sh"; exit 1; }
mkdir -p "$OUTDIR"

echo "Cache target: ${LLAMA_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
echo "Prefetching ${#CONFIGS[@]} models, up to $PREFETCH_JOBS at a time (the slow part — run it once)."
echo "Tip: if you hit HF rate limits, 'export HF_TOKEN=hf_xxx' first (or lower PREFETCH_JOBS)."
echo

# Download + verify one model: a minimal CPU-only llama-bench run triggers the
# -hf download and confirms the file loads. The pass/fail verdict is recorded in
# a per-label .status file so the parallel workers can be tallied after they all
# finish (a subshell can't increment a counter in the parent).
fetch_one() {
  local label=$1 repo=$2
  if "$LLAMA_BENCH" -hf "$repo" -ngl 0 -p 1 -n 1 -r 1 \
       > "$OUTDIR/prefetch_${label}.log" 2>&1; then
    echo OK   > "$OUTDIR/prefetch_${label}.status"
    echo "    OK      $label  ($repo)"
  else
    echo FAIL > "$OUTDIR/prefetch_${label}.status"
    echo "    FAILED  $label  ($repo) — see $OUTDIR/prefetch_${label}.log (bad quant tag? network? disk full?)"
  fi
}

rm -f "$OUTDIR"/prefetch_*.status        # clear tallies from a previous run

# Launch downloads, keeping at most PREFETCH_JOBS running at once. The throttle
# polls the running background-job count once a second before starting the next.
n=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl <<< "$entry"   # sys/tmpl unused here; read them so they don't fold into $type
  n=$((n+1))
  while [ "$(jobs -rp | wc -l)" -ge "$PREFETCH_JOBS" ]; do sleep 1; done
  echo "[$n/${#CONFIGS[@]}] start: $label  ($repo)"
  fetch_one "$label" "$repo" &
done
wait                                     # let the final batch finish

fail=$(grep -lx FAIL "$OUTDIR"/prefetch_*.status 2>/dev/null | wc -l | tr -d ' ')
rm -f "$OUTDIR"/prefetch_*.status        # keep the .log files, drop the tally files

echo
if [ "$fail" -eq 0 ]; then
  echo "All ${#CONFIGS[@]} models cached. You can now run ./run-bench.sh and ./run-quality.sh"
  echo "offline-fast — they'll load straight from cache."
else
  echo "$fail model(s) failed. Fix those lines in configs.sh and re-run; cached ones are skipped."
  exit 1
fi
