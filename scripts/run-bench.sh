#!/usr/bin/env bash
# ===========================================================================
# run-bench.sh — unattended throughput + fit sweep for any GGUF quants.
# Loops every config in configs.sh across every context depth, runs llama-bench
# (auto-downloads via -hf), samples peak VRAM/RAM, and writes one tidy CSV.
# No MTP, text-only.  OOM just marks that row FAIL and keeps going.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh

command -v jq >/dev/null || { echo "Install jq:  sudo apt install jq"; exit 1; }
[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not found at $LLAMA_BENCH — set LLAMA_DIR in configs.sh"; exit 1; }
command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; VRAM column will be blank"

mkdir -p "$OUTDIR/json"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUTDIR/throughput_$STAMP.csv"
echo "label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status" > "$CSV"

# Background peak-memory sampler: runs while flag file $1 exists; writes "maxv maxr" to $2.
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

total=$(( ${#CONFIGS[@]} * ${#DEPTHS[@]} )); i=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl <<< "$entry"   # sys/tmpl unused here (bench doesn't template); read them so they don't fold into $type
  quant="${repo##*:}"
  moe_flags=(); [ "$type" = "moe" ] && moe_flags=(--n-cpu-moe "$NCMOE_ALL")

  for depth in "${DEPTHS[@]}"; do
    i=$((i+1)); echo "[$i/$total] $label  depth=$depth ..."
    flag=$(mktemp); memout=$(mktemp); : > "$flag"
    sample_mem "$flag" "$memout" & sampler=$!

    json="$OUTDIR/json/${label}_d${depth}.json"
    if "$LLAMA_BENCH" -hf "$repo" \
         -ngl 99 -fa on "${moe_flags[@]}" \
         -ctk "$KV_QUANT" -ctv "$KV_QUANT" \
         -t "$THREADS" -p "$PROMPT_LEN" -n "$GEN_LEN" -d "$depth" \
         -r "$REPS" -o json > "$json" 2> "$OUTDIR/json/${label}_d${depth}.log"; then
      status=OK
      pp=$(jq -r '[.[]|select(.n_gen==0)][0].avg_ts // empty' "$json")
      tg=$(jq -r '[.[]|select(.n_prompt==0)][0].avg_ts // empty' "$json")
    else
      status=FAIL; pp=""; tg=""
      echo "      (FAIL — see $OUTDIR/json/${label}_d${depth}.log; likely OOM at this depth)"
    fi

    rm -f "$flag"; wait "$sampler" 2>/dev/null; read -r vram ram < "$memout"; rm -f "$memout"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$label" "$quant" "$type" "$depth" "$pp" "$tg" "$vram" "$ram" "$status" >> "$CSV"
    echo "      pp=${pp:-NA} tg=${tg:-NA} tok/s | VRAM=${vram}MiB RAM=${ram}MiB [$status]"
  done
done

echo; echo "=== Done -> $CSV ==="
command -v column >/dev/null && column -s, -t "$CSV" || cat "$CSV"
