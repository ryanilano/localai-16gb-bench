#!/usr/bin/env bash
# ===========================================================================
# run-bench.sh — unattended throughput + fit sweep for any GGUF quants.
# Loops every model in models.ini across every context depth, runs llama-bench
# (auto-downloads via -hf), samples peak VRAM/RAM, and writes one tidy CSV.
# No MTP, text-only.  OOM just marks that row FAIL and keeps going.
#
# Per-model flags come from models.ini via ini_flags (bench scope): [*] defaults
# plus section overrides. n-cpu-moe is applied only when type = moe.
# --dry-run: print each composed llama-bench command; run nothing.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh
DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1

command -v jq >/dev/null || { echo "Install jq:  sudo apt install jq"; exit 1; }
[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not found at $LLAMA_BENCH — set LLAMA_DIR in configs.sh"; exit 1; }
command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; VRAM column will be blank"

mapfile -t LABELS < <(ini_sections)
[ "${#LABELS[@]}" -gt 0 ] || { echo "No models registered — uncomment or add sections in models.ini"; exit 1; }

if [ "$DRY_RUN" -eq 1 ]; then
  for label in "${LABELS[@]}"; do
    repo="$(ini_get "$label" hf)"
    mapfile -t FLAGS < <(ini_flags "$label" bench)
    echo "[dry-run] $label:"
    echo "  $LLAMA_BENCH -hf $repo ${FLAGS[*]} -p $PROMPT_LEN -n $GEN_LEN -d <depth: ${DEPTHS[*]}> -r $REPS -o json"
  done
  exit 0
fi

mkdir -p "$OUTDIR/json"
SLUG=$(run_slug)                             # human-readable, per-run, collision-resistant
CSV="$OUTDIR/throughput_$SLUG.csv"
VERSIONS="$OUTDIR/versions_$SLUG.txt"
REPORT="$OUTDIR/RUN_$SLUG.md"
echo "label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status" > "$CSV"

# Version-stamp this run: co-located, per-run file so results are self-documenting
# and a later run never clobbers this one's provenance.
capture_versions "$VERSIONS" "$LLAMA_BENCH"

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

total=$(( ${#LABELS[@]} * ${#DEPTHS[@]} )); i=0
for label in "${LABELS[@]}"; do
  repo="$(ini_get "$label" hf)"
  type="$(ini_get "$label" type)"
  [ -n "$repo" ] || { echo "SKIP $label — no 'hf =' key in models.ini"; continue; }
  quant="${repo##*:}"
  # All per-model flags (defaults + overrides, MoE gate included) in one array.
  mapfile -t FLAGS < <(ini_flags "$label" bench)

  for depth in "${DEPTHS[@]}"; do
    i=$((i+1)); echo "[$i/$total] $label  depth=$depth ..."
    flag=$(mktemp); memout=$(mktemp); : > "$flag"
    sample_mem "$flag" "$memout" & sampler=$!

    json="$OUTDIR/json/${label}_d${depth}.json"
    if "$LLAMA_BENCH" -hf "$repo" \
         "${FLAGS[@]}" \
         -p "$PROMPT_LEN" -n "$GEN_LEN" -d "$depth" \
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

# Human-readable report: provenance stamp + the throughput CSV as a markdown table.
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

echo; echo "=== Done -> $CSV ==="
echo "    provenance: $VERSIONS"
echo "    report:     $REPORT"
command -v column >/dev/null && column -s, -t "$CSV" || cat "$CSV"
