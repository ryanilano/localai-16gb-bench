#!/usr/bin/env bash
# ===========================================================================
# bench-27b-heretic-neocode.sh — single-model throughput + fit sweep.
#
# Model : DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF
#         https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF
#         (dense 27B — Heretic-uncensored finetune of the NEO-CODE line)
#
# A one-off version of run-bench.sh for a single quant: same llama-bench
# invocation, same depth grid, same peak VRAM/RAM sampling, same self-contained
# run folder + provenance + RUN.md. Runs the whole config matrix's engine on
# exactly one model, so the CSV drops straight into results/ next to the sweep.
#
#   ./bench-27b-heretic-neocode.sh prefetch        # download+verify the quants first (run once)
#   ./bench-27b-heretic-neocode.sh                 # both quants, default depths (0..32k)
#   QUANTS="IQ3_M" ./bench-27b-heretic-neocode.sh  # restrict to one quant
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh          # LLAMA_BENCH, REPS, PROMPT_LEN, GEN_LEN, DEPTHS, OUTDIR
                             # (+ ini.sh API; shared [*] flag defaults pulled below)
source ./versions.sh         # run_slug, capture_versions, csv_to_md

# --- Model under test (quants + type overridable) --------------------------
REPO_BASE="DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF"
# Space-separated list; both are the sweet spots from the NEO-CODE line — IQ3_M
# (fastest + roomiest to 80k) and IQ4_XS (higher fidelity, ~15 GB, fits to ~32k).
QUANTS="${QUANTS:-IQ3_M IQ4_XS}"
TYPE="${TYPE:-dense}"        # dense finetune — no expert offload

# Shared llama.cpp flag defaults. These used to be configs.sh vars (KV_QUANT /
# THREADS / NCMOE_ALL); the models.ini migration moved them into the [*] section,
# so pull them from there to stay in sync with the registry's shared defaults.
CTK="$(ini_get '*' ctk)"; CTV="$(ini_get '*' ctv)"
THREADS="$(ini_get '*' t)"
NCMOE_ALL="$(ini_get '*' n-cpu-moe)"

command -v jq >/dev/null || { echo "Install jq:  sudo apt install jq"; exit 1; }
[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not found at $LLAMA_BENCH — set LLAMA_DIR in configs.sh"; exit 1; }

# --- Prefetch mode: `./bench-...sh prefetch` (or PREFETCH=1) ----------------
# Pull + verify every quant to cache first, so the bench never stalls mid-run.
# CPU-only minimal load (-ngl 0, tiny -p/-n) — same trick as prefetch.sh; the
# -hf download grabs the full GGUF and the load confirms it parses. No GPU used.
# Downloads run CONCURRENTLY, up to PREFETCH_JOBS at a time (default 3), so both
# quants pull in parallel instead of one-then-the-other.
if [ "${1:-}" = "prefetch" ] || [ "${PREFETCH:-0}" = 1 ]; then
  read -ra quants <<< "$QUANTS"
  PREFETCH_JOBS="${PREFETCH_JOBS:-3}"
  echo ">>> prefetch: $REPO_BASE  quants: ${quants[*]}  (up to $PREFETCH_JOBS concurrent)"
  mkdir -p "$OUTDIR"
  rm -f "$OUTDIR"/prefetch_HereticNEO_*.status

  fetch_one() {   # $1 = quant
    local q="$1"
    if "$LLAMA_BENCH" -hf "$REPO_BASE:$q" -ngl 0 -p 1 -n 1 -r 1 \
         > "$OUTDIR/prefetch_HereticNEO_${q}.log" 2>&1; then
      echo OK   > "$OUTDIR/prefetch_HereticNEO_${q}.status"
      echo "    OK      $REPO_BASE:$q"
    else
      echo FAIL > "$OUTDIR/prefetch_HereticNEO_${q}.status"
      echo "    FAILED  $REPO_BASE:$q  — see $OUTDIR/prefetch_HereticNEO_${q}.log (bad quant tag? network? disk full?)"
    fi
  }

  # Live progress: report per-quant done/total plus how fast the on-disk cache is
  # growing (≈ download speed). A run of "+0 MiB" ticks means a stalled download.
  # Silence with PREFETCH_PROGRESS=0; tick interval via PROGRESS_EVERY (default 5s).
  PREFETCH_PROGRESS="${PREFETCH_PROGRESS:-1}"; PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
  _cache_dirs=(); for _d in "${LLAMA_CACHE:-}" "${HF_HOME:-}" "$HOME/.cache/llama.cpp" "$HOME/.cache/huggingface"; do
    [ -n "$_d" ] && [ -d "$_d" ] && _cache_dirs+=("$_d"); done
  _last_kib=0
  show_progress() {
    local done total="${#quants[@]}" now_kib
    done=$(ls "$OUTDIR"/prefetch_HereticNEO_*.status 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PREFETCH_PROGRESS" = 1 ] && [ "${#_cache_dirs[@]}" -gt 0 ]; then
      now_kib=$(du -sck "${_cache_dirs[@]}" 2>/dev/null | tail -1 | cut -f1)
      if [ -n "$now_kib" ] && [ "$_last_kib" -gt 0 ] && [ "$now_kib" -ge "$_last_kib" ]; then
        printf '      .. %d/%d done | cache %d GiB, +%d MiB in ~%ds (%s active)\n' \
          "$done" "$total" "$(( now_kib / 1048576 ))" "$(( (now_kib - _last_kib) / 1024 ))" \
          "$PROGRESS_EVERY" "$(jobs -rp | wc -l | tr -d ' ')"
      else
        printf '      .. %d/%d done (%s active)\n' "$done" "$total" "$(jobs -rp | wc -l | tr -d ' ')"
      fi
      [ -n "$now_kib" ] && _last_kib=$now_kib
    fi
  }

  for q in "${quants[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PREFETCH_JOBS" ]; do show_progress; sleep "$PROGRESS_EVERY"; done
    echo "    start: $q ..."
    fetch_one "$q" &
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do show_progress; sleep "$PROGRESS_EVERY"; done
  wait

  fail=$(grep -lx FAIL "$OUTDIR"/prefetch_HereticNEO_*.status 2>/dev/null | wc -l | tr -d ' ')
  rm -f "$OUTDIR"/prefetch_HereticNEO_*.status
  [ "$fail" -eq 0 ] && echo "All quants cached." || { echo "$fail quant(s) failed."; exit 1; }
  exit 0
fi

command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; VRAM column will be blank"

SLUG=$(run_slug)
RUNDIR="$OUTDIR/$SLUG"
mkdir -p "$RUNDIR/json"
CSV="$RUNDIR/throughput.csv"
VERSIONS="$RUNDIR/versions.txt"
REPORT="$RUNDIR/RUN.md"
echo "label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status" > "$CSV"
capture_versions "$VERSIONS" "$LLAMA_BENCH"

# Background peak-memory sampler (identical to run-bench.sh).
sample_mem() {
  local flag="$1" out="$2" maxv=0 maxr=0 v r
  while [ -f "$flag" ]; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    r=$(free -m | awk '/^Mem:/{print $3}')
    [ -n "${v:-}" ] && [ "$v" -gt "$maxv" ] && maxv=$v
    [ -n "${r:-}" ] && [ "$r" -gt "$maxr" ] && maxr=$r
    sleep 1
  done
  echo "$maxv $maxr" > "$out"
}

moe_flags=(); [ "$TYPE" = "moe" ] && moe_flags=(--n-cpu-moe "$NCMOE_ALL")
# Single depth grid: the old longctx / DEPTHS_MOE split was dropped in the models.ini migration.
depths=("${DEPTHS[@]}")

read -ra quants <<< "$QUANTS"
total=$(( ${#quants[@]} * ${#depths[@]} )); i=0
echo ">>> bench: $REPO_BASE  ($TYPE)  quants: ${quants[*]}  depths: ${depths[*]}"

for QUANT in "${quants[@]}"; do
  LABEL="27B_Heretic_NEO_CODE_${QUANT}"
  REPO="$REPO_BASE:$QUANT"
  for depth in "${depths[@]}"; do
    i=$((i+1)); echo "[$i/$total] $LABEL  depth=$depth ..."
    flag=$(mktemp); memout=$(mktemp); : > "$flag"
    sample_mem "$flag" "$memout" & sampler=$!

    json="$RUNDIR/json/${LABEL}_d${depth}.json"
    if "$LLAMA_BENCH" -hf "$REPO" \
         -ngl 99 -fa on "${moe_flags[@]}" \
         -ctk "$CTK" -ctv "$CTV" \
         -t "$THREADS" -p "$PROMPT_LEN" -n "$GEN_LEN" -d "$depth" \
         -r "$REPS" -o json > "$json" 2> "$RUNDIR/json/${LABEL}_d${depth}.log"; then
      status=OK
      pp=$(jq -r '[.[]|select(.n_gen==0)][0].avg_ts // empty' "$json")
      tg=$(jq -r '[.[]|select(.n_prompt==0)][0].avg_ts // empty' "$json")
    else
      status=FAIL; pp=""; tg=""
      echo "      (FAIL — see $RUNDIR/json/${LABEL}_d${depth}.log; likely OOM at this depth)"
    fi

    rm -f "$flag"; wait "$sampler" 2>/dev/null; read -r vram ram < "$memout"; rm -f "$memout"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$LABEL" "$QUANT" "$TYPE" "$depth" "$pp" "$tg" "$vram" "$ram" "$status" >> "$CSV"
    echo "      pp=${pp:-NA} tg=${tg:-NA} tok/s | VRAM=${vram}MiB RAM=${ram}MiB [$status]"
  done
done

{
  echo "# Benchmark run — $SLUG"
  echo
  echo "## Results"
  echo
  csv_to_md "$CSV"
  echo
  echo "## Provenance"
  echo
  echo '```'
  cat "$VERSIONS"
  echo '```'
} > "$REPORT"

echo; echo "=== Done -> $RUNDIR/ ==="
echo "    csv:        $CSV"
echo "    provenance: $VERSIONS"
echo "    report:     $REPORT"
command -v column >/dev/null && column -s, -t "$CSV" || cat "$CSV"
