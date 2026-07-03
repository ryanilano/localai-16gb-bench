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
#   BENCH_PROFILE=longctx ./bench-27b-heretic-neocode.sh   # deep sweep (80k dense cap)
#   QUANTS="IQ3_M" ./bench-27b-heretic-neocode.sh  # restrict to one quant
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh          # LLAMA_BENCH, KV_QUANT, THREADS, REPS, PROMPT_LEN, GEN_LEN,
                             # DEPTHS[/_MOE], NCMOE_ALL, OUTDIR, HF_TOKEN handling
source ./versions.sh         # run_slug, capture_versions, csv_to_md

# --- Model under test (quants + type overridable) --------------------------
REPO_BASE="DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF"
# Space-separated list; both are the sweet spots from the NEO-CODE line — IQ3_M
# (fastest + roomiest to 80k) and IQ4_XS (higher fidelity, ~15 GB, fits to ~32k).
QUANTS="${QUANTS:-IQ3_M IQ4_XS}"
TYPE="${TYPE:-dense}"        # dense finetune — no expert offload

command -v jq >/dev/null || { echo "Install jq:  sudo apt install jq"; exit 1; }
[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not found at $LLAMA_BENCH — set LLAMA_DIR in configs.sh"; exit 1; }

# --- Prefetch mode: `./bench-...sh prefetch` (or PREFETCH=1) ----------------
# Pull + verify every quant to cache first, so the bench never stalls mid-run.
# CPU-only minimal load (-ngl 0, tiny -p/-n) — same trick as prefetch.sh; the
# -hf download grabs the full GGUF and the load confirms it parses. No GPU used.
if [ "${1:-}" = "prefetch" ] || [ "${PREFETCH:-0}" = 1 ]; then
  read -ra quants <<< "$QUANTS"
  echo ">>> prefetch: $REPO_BASE  quants: ${quants[*]}"
  fail=0
  for q in "${quants[@]}"; do
    echo "    fetching $q ..."
    if "$LLAMA_BENCH" -hf "$REPO_BASE:$q" -ngl 0 -p 1 -n 1 -r 1 >/dev/null 2>&1; then
      echo "    OK      $REPO_BASE:$q"
    else
      echo "    FAILED  $REPO_BASE:$q  (bad quant tag? network? disk full?)"; fail=$((fail+1))
    fi
  done
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
if [ "$TYPE" = "moe" ]; then depths=("${DEPTHS_MOE[@]}"); else depths=("${DEPTHS[@]}"); fi

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
         -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
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
