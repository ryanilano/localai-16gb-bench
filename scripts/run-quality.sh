#!/usr/bin/env bash
# ===========================================================================
# run-quality.sh — unattended quality pass (model-neutral).
# For each config: start llama-server (with that model's chat template + system
# prompt, if any), run every prompt in prompts/*.txt through /v1/chat/completions
# at coding sampling settings, save each answer to quality/<label>/<prompt>.md,
# then stop the server and move on.  Read the outputs whenever you like.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh

command -v jq >/dev/null   || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null || { echo "Install curl"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }
# A global CHAT_TEMPLATE is optional now (per-model field 5 overrides it; both may be empty).
[ -z "$CHAT_TEMPLATE" ] || [ -f "$CHAT_TEMPLATE" ] || { echo "CHAT_TEMPLATE set but not found at $CHAT_TEMPLATE"; exit 1; }

GEN=768                 # max tokens per answer
# Server context window for the quality pass. Default 8192 is ample for the short
# prompts in prompts/ (each answers well inside GEN tokens) and keeps every model's
# KV small so all fit. Override for a one-off long-context quality probe, e.g.
# QCTX=81920 ./run-quality.sh — but the model must be validated to that depth (see
# the fit map / throughput.csv) or the server will OOM on boot. Env-overridable.
QCTX="${QCTX:-8192}"

# GPU-layer offload for the quality server. 99 = full offload (matches the throughput
# bench regime). Per-model override via CONFIGS field 6; env-override via NGL=. A config
# that fails to boot at its ngl is retried once with -ngl -1 (llama.cpp auto-fit) below,
# so a tight quant still yields answers instead of a blank dir.
NGL_DEFAULT="${NGL:-99}"

# This quality pass gets its own self-contained run folder ($OUTDIR/<slug>/),
# so re-runs and different model sets never overwrite each other's answers.
SLUG=$(run_slug)
RUNDIR="$OUTDIR/$SLUG"
mkdir -p prompts "$RUNDIR/quality"
VERSIONS="$RUNDIR/versions.txt"
REPORT="$RUNDIR/RUN.md"
# Version-stamp this pass so the answers record the exact toolchain that produced them.
capture_versions "$VERSIONS" "$LLAMA_SERVER"

# Seed a few example prompts on first run (add your own .txt files anytime).
if ! ls prompts/*.txt >/dev/null 2>&1; then
  printf 'Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Include 3 unit tests.\n' > prompts/01_palindrome.txt
  printf 'Refactor this for clarity and proper error handling:\ndef d(u):\n import requests;return requests.get(u).json()["data"]\n' > prompts/02_refactor.txt
  printf 'You are an agent in a sandbox. Give a step-by-step plan to find every TODO comment in a Git repo, then a single bash command that does it.\n' > prompts/03_agent_plan.txt
  echo "Seeded example prompts in ./prompts/ — replace them with your own real tasks."
fi

start_server() {   # $1 = repo:quant ; remaining args = extra flags (e.g. --n-cpu-moe, --jinja ...)
  local repo="$1"; shift
  # --no-mmproj: these Qwen3.6 GGUF repos ship a vision projector that the -hf
  # resolver auto-loads. On a 16 GB card the CLIP buffer (~888 MiB) OOMs *after*
  # the model is fully offloaded, crashing the server on boot (every answer then
  # logs "no response"). We only run text prompts, so drop it entirely.
  # -ngl is passed by the caller (per-config field 6 / auto-fit fallback), not fixed here.
  "$LLAMA_SERVER" -hf "$repo" --no-mmproj \
    -fa on "$@" \
    -ctk "$KV_QUANT" -ctv "$KV_QUANT" -t "$THREADS" -c "$QCTX" \
    --host 127.0.0.1 --port "$PORT" > "$SRV_LOG" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 600); do                       # wait up to ~20 min for first download
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV_PID" 2>/dev/null || return 1     # server died
    sleep 2
  done
  return 1
}

stop_server() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; sleep 2; }

ran_labels=()   # models actually exercised this pass, for the run report
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl ngl <<< "$entry"
  moe_flags=(); [ "$type" = "moe" ] && moe_flags=(--n-cpu-moe "$NCMOE_ALL")

  # Resolve per-model system prompt, chat template, and ngl, with global fallbacks.
  [ -n "$sys" ]  || sys="$SYS_DEFAULT"
  [ -n "$tmpl" ] || tmpl="$CHAT_TEMPLATE"
  [ -n "$ngl" ]  || ngl="$NGL_DEFAULT"
  tmpl_flags=()
  if [ -n "$tmpl" ]; then
    [ -f "$tmpl" ] || { echo "    template not found for $label: $tmpl — skipping"; continue; }
    tmpl_flags=(--jinja --chat-template-file "$tmpl")
  fi

  outdir="$RUNDIR/quality/$label"; mkdir -p "$outdir"
  # Per-model server log next to this model's answers, so a crash for one config
  # isn't clobbered by the next (and the "(no response — see _server.log)" note in
  # each answer points at the log right beside it).
  SRV_LOG="$outdir/_server.log"

  eff_ngl="$ngl"   # actual offload used, for the run report (may change on fallback)
  echo ">>> starting $label (ngl=$ngl) ..."
  if ! start_server "$repo" -ngl "$ngl" "${moe_flags[@]}" "${tmpl_flags[@]}"; then
    stop_server
    if [ "$ngl" = "-1" ]; then
      echo "    server failed to start even at auto-fit (see $SRV_LOG) — skipping $label"
      continue
    fi
    # Fallback: retry once letting llama.cpp auto-fit layers to free VRAM, so a config
    # that can't hold full offload still yields answers. Keep the first attempt's log;
    # write the retry to its own file so neither clobbers the other.
    echo "    ngl=$ngl failed to boot — retrying with auto-fit (-ngl -1); first log: $SRV_LOG"
    SRV_LOG="$outdir/_server.autofit.log"
    if ! start_server "$repo" -ngl -1 "${moe_flags[@]}" "${tmpl_flags[@]}"; then
      echo "    auto-fit also failed (see $SRV_LOG) — skipping $label"
      stop_server; continue
    fi
    eff_ngl="-1 (auto-fit fallback)"
  fi

  for pf in prompts/*.txt; do
    name=$(basename "$pf" .txt)
    echo "    $label / $name"
    # Include a system message only when a system prompt is set for this model.
    body=$(jq -n --arg sys "$sys" --rawfile u "$pf" --argjson n "$GEN" \
      '{messages: (if ($sys|length) > 0 then [{role:"system",content:$sys}] else [] end)
                  + [{role:"user",content:$u}],
        temperature:0.6, top_p:0.95, top_k:20, max_tokens:$n}')
    resp=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
             -H 'Content-Type: application/json' -d "$body" || echo '{}')
    content=$(echo "$resp" | jq -r '.choices[0].message.content // .error.message // "(no response — see _server.log)"')
    {
      echo "# $label — $name"; echo
      echo "## Prompt"; echo; cat "$pf"; echo
      echo "## Response"; echo; echo "$content"
    } > "$outdir/$name.md"
  done

  stop_server
  ran_labels+=("$label -> $repo  [ngl=$eff_ngl]")
done

# Human-readable report: provenance stamp + which models/prompts this pass ran.
{
  echo "# Quality run — $SLUG"
  echo
  echo "## Models"
  echo
  if [ "${#ran_labels[@]}" -gt 0 ]; then
    for l in "${ran_labels[@]}"; do echo "- $l"; done
  else
    echo "- (none — no active CONFIGS)"
  fi
  echo
  echo "## Prompts"
  echo
  for pf in prompts/*.txt; do echo "- $(basename "$pf")"; done
  echo
  echo "## Provenance"
  echo
  echo '```'
  cat "$VERSIONS"
  echo '```'
} > "$REPORT"

echo; echo "Done -> $RUNDIR/"
echo "    answers:    $RUNDIR/quality/<label>/*.md"
echo "    provenance: $VERSIONS"
echo "    report:     $REPORT"
