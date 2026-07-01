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

# Show an aggregate download-progress line every PROGRESS_EVERY seconds while
# models are pulling, so a long download looks alive instead of hung. It's
# derived from the growing on-disk cache, NOT llama.cpp's native progress bar:
# that bar is suppressed when output isn't a TTY, and we redirect each parallel
# worker to a log. For the real per-file bar, pull one at a time on your terminal
# (PREFETCH_JOBS=1). Silence this line with PREFETCH_PROGRESS=0.
PREFETCH_PROGRESS="${PREFETCH_PROGRESS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"   # seconds between progress ticks

# Opt-in safety net: llama.cpp's -hf downloader has no timeout of its own, so a
# dead socket can hang a fetch forever (we saw a 76-min stall). Set PREFETCH_TIMEOUT
# to cap each model's fetch via `timeout`; empty = off (default). Use a duration
# LONGER than your slowest real download so only a truly stalled transfer is killed,
# e.g. PREFETCH_TIMEOUT=45m. A timed-out model is marked FAIL — re-run to resume.
PREFETCH_TIMEOUT="${PREFETCH_TIMEOUT:-}"

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
  local label=$1 repo=$2 rc
  if [ -n "$PREFETCH_TIMEOUT" ]; then
    timeout "$PREFETCH_TIMEOUT" "$LLAMA_BENCH" -hf "$repo" -ngl 0 -p 1 -n 1 -r 1 \
      > "$OUTDIR/prefetch_${label}.log" 2>&1
  else
    "$LLAMA_BENCH" -hf "$repo" -ngl 0 -p 1 -n 1 -r 1 \
      > "$OUTDIR/prefetch_${label}.log" 2>&1
  fi
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo OK   > "$OUTDIR/prefetch_${label}.status"
    echo "    OK      $label  ($repo)"
  elif [ -n "$PREFETCH_TIMEOUT" ] && [ "$rc" -eq 124 ]; then
    echo FAIL > "$OUTDIR/prefetch_${label}.status"
    echo "    TIMEOUT $label  ($repo) — no finish within $PREFETCH_TIMEOUT (stalled download?); re-run to resume from cache"
  else
    echo FAIL > "$OUTDIR/prefetch_${label}.status"
    echo "    FAILED  $label  ($repo) — see $OUTDIR/prefetch_${label}.log (bad quant tag? network? disk full?)"
  fi
}

rm -f "$OUTDIR"/prefetch_*.status        # clear tallies from a previous run

# Aggregate download progress: sum the on-disk model cache and report the delta
# since the last tick (≈ download speed). The cache dir varies by llama.cpp
# version (~/.cache/llama.cpp on newer builds, ~/.cache/huggingface on older),
# so watch every plausible one that exists and let the growing one show through.
# A run of "+0 MiB" ticks means a stalled download — kill it and re-run.
_cache_dirs=()
for _d in "${LLAMA_CACHE:-}" "${HF_HOME:-}" "$HOME/.cache/llama.cpp" "$HOME/.cache/huggingface"; do
  [ -n "$_d" ] && [ -d "$_d" ] && _cache_dirs+=("$_d")
done
_last_kib=0
show_progress() {
  [ "$PREFETCH_PROGRESS" = 1 ] && [ "${#_cache_dirs[@]}" -gt 0 ] || return 0
  local now_kib
  now_kib=$(du -sck "${_cache_dirs[@]}" 2>/dev/null | tail -1 | cut -f1)
  [ -n "$now_kib" ] || return 0
  if [ "$_last_kib" -gt 0 ] && [ "$now_kib" -ge "$_last_kib" ]; then
    printf '      .. downloading: cache %d GiB, +%d MiB in ~%ds (%s active)\n' \
      "$(( now_kib / 1048576 ))" "$(( (now_kib - _last_kib) / 1024 ))" \
      "$PROGRESS_EVERY" "$(jobs -rp | wc -l | tr -d ' ')"
  fi
  _last_kib=$now_kib
}

# Launch downloads, keeping at most PREFETCH_JOBS running at once. The throttle
# polls the running-job count every PROGRESS_EVERY seconds and prints progress.
n=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl <<< "$entry"   # sys/tmpl unused here; read them so they don't fold into $type
  n=$((n+1))
  while [ "$(jobs -rp | wc -l)" -ge "$PREFETCH_JOBS" ]; do show_progress; sleep "$PROGRESS_EVERY"; done
  echo "[$n/${#CONFIGS[@]}] start: $label  ($repo)"
  fetch_one "$label" "$repo" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do show_progress; sleep "$PROGRESS_EVERY"; done
wait                                     # reap the final batch

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
