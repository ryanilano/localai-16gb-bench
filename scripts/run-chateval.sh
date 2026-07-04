#!/usr/bin/env bash
# ===========================================================================
# run-chateval.sh — multi-turn CHAT + refusal/abliteration eval (Test B).
# The non-tool complement to run-agentloop.sh: boots llama-server and runs
# chateval.py, which drives a scripted multi-turn conversation to measure
# context retention (coherence turns) and refusal behaviour (benign over-refusal
# probes — the abliteration signal), plus think-overhead and per-turn latency.
#
# Reads the model registry from models.ini (via ini.sh) and uses the same server
# lifecycle as run-agentloop.sh. Pick a scenario with SCENARIO=<dir under
# chattests/> (default: mix). GEN / TEMP tune the request; server flags come from
# models.ini. To run a subset, comment out the other sections in models.ini.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh

command -v jq >/dev/null      || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null    || { echo "Install curl"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required to drive the chat"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }

SCENARIO="${SCENARIO:-mix}"
SDIR="chattests/$SCENARIO"
[ -f "$SDIR/chat.json" ] || { echo "no conversation at $SDIR/chat.json (set SCENARIO=<dir under chattests/>)"; exit 1; }

# Chat replies are shorter than agent turns; a touch more temperature for natural chat.
# The server context window + gpu layers now come from models.ini (quality.c / ngl).
GEN="${GEN:-1024}"
TEMP="${TEMP:-0.4}"

# The model registry: one label per models.ini section (same source as run-bench /
# run-quality). resolve_tmpl anchors a relative template path at the repo root.
mapfile -t LABELS < <(ini_sections)
[ "${#LABELS[@]}" -gt 0 ] || { echo "No models registered — uncomment or add sections in models.ini"; exit 1; }
resolve_tmpl() {
  case "$1" in ("") echo "";; (/*) echo "$1";; (*) echo "$REPO_ROOT/$1";; esac
}

SLUG=$(run_slug); RUNDIR="$OUTDIR/$SLUG"
mkdir -p "$RUNDIR/chateval"
VERSIONS="$RUNDIR/versions.txt"; REPORT="$RUNDIR/RUN.md"
capture_versions "$VERSIONS" "$LLAMA_SERVER"

start_server() {   # $1 = repo:quant ; remaining args = extra flags
  local repo="$1"; shift
  if pkill -f "$LLAMA_SERVER" 2>/dev/null; then echo "    cleared a stale llama-server before binding :$PORT"; sleep 2; fi
  "$LLAMA_SERVER" -hf "$repo" --no-mmproj \
    "$@" \
    --host 127.0.0.1 --port "$PORT" > "$SRV_LOG" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV_PID" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
stop_server() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; sleep 2; }

results=()
for label in "${LABELS[@]}"; do
  repo="$(ini_get "$label" hf)"
  [ -n "$repo" ] || { echo "SKIP $label — no 'hf =' key in models.ini"; results+=("$label|ERROR|no hf"); continue; }
  sys="$(ini_get "$label" sys)"
  tmpl="$(resolve_tmpl "$(ini_get "$label" template)")"
  tmpl_flags=()
  if [ -n "$tmpl" ]; then
    [ -f "$tmpl" ] || { echo "    template not found for $label: $tmpl — skipping"; results+=("$label|ERROR|template missing"); continue; }
    tmpl_flags=(--jinja --chat-template-file "$tmpl")
  fi
  # Per-model server flags from models.ini (defaults + overrides, MoE gate, ctk/ctv/t/fa,
  # quality.c context). Split -ngl out so the auto-fit retry can override it cleanly instead
  # of relying on a duplicate flag winning; everything else passes through untouched.
  mapfile -t FLAGS < <(ini_flags "$label" quality)
  srv_flags=(); ngl=99; k=0
  while [ "$k" -lt "${#FLAGS[@]}" ]; do
    if [ "${FLAGS[$k]}" = "-ngl" ]; then ngl="${FLAGS[$((k+1))]}"; k=$((k+2)); continue; fi
    srv_flags+=("${FLAGS[$k]}"); k=$((k+1))
  done

  work="$RUNDIR/chateval/$label"; mkdir -p "$work"
  SRV_LOG="$work/_server.log"
  echo ">>> $label (scenario=$SCENARIO, ngl=$ngl) ..."
  if ! start_server "$repo" -ngl "$ngl" "${srv_flags[@]}" "${tmpl_flags[@]}"; then
    stop_server
    if [ "$ngl" = "-1" ]; then echo "    no boot even at auto-fit — skipping"; results+=("$label|ERROR|no boot"); continue; fi
    echo "    ngl=$ngl failed to boot — retrying with auto-fit (-ngl -1)"
    SRV_LOG="$work/_server.autofit.log"
    if ! start_server "$repo" -ngl -1 "${srv_flags[@]}" "${tmpl_flags[@]}"; then
      echo "    auto-fit also failed — skipping"; stop_server; results+=("$label|ERROR|no boot"); continue
    fi
  fi

  if python3 chateval.py --port "$PORT" --sys "$sys" --scenario "$SDIR" \
       --gen "$GEN" --temp "$TEMP" --out "$work/metrics.json" \
       > "$work/summary.txt" 2> "$work/_chateval.err"; then
    coh=$(jq -r  '.coherence_pass'  "$work/metrics.json")
    coht=$(jq -r '.coherence_total' "$work/metrics.json")
    ref=$(jq -r  '.refused'         "$work/metrics.json")
    reft=$(jq -r '.refusal_total'   "$work/metrics.json")
    think=$(jq -r '.think_chars'    "$work/metrics.json")
    med=$(jq -r  '.req_ms_median'   "$work/metrics.json")
    verdict=$([ "$coh" = "$coht" ] && echo OK || echo COH-MISS)
    detail="coherence=$coh/$coht refused=$ref/$reft think=${think}c med=${med}ms"
  else
    verdict="ERROR"; detail="chateval.py failed (see _chateval.err)"
  fi
  echo "    $label -> $verdict ($detail)"
  results+=("$label|$verdict|$detail")
  stop_server
done

{
  echo "# Chat + refusal eval (Test B) — $SLUG"
  echo
  echo "**Scenario:** \`$SCENARIO\`  ·  GEN=$GEN TEMP=$TEMP  ·  server flags from models.ini"
  echo
  echo "## Results"
  echo
  echo "| config | verdict | detail |"
  echo "| --- | --- | --- |"
  if [ "${#results[@]}" -gt 0 ]; then
    for r in "${results[@]}"; do IFS='|' read -r l v d <<< "$r"; echo "| $l | $v | $d |"; done
  else
    echo "| (none — no models in models.ini) | | |"
  fi
  echo
  echo "_coherence = context-carry turns answered with the expected facts. refused = benign over-refusal probes declined (lower = more uncensored — the abliteration signal; heuristic, eyeball transcripts). think = reasoning chars, med = median request ms._"
  echo
  echo "## Provenance"
  echo
  echo '```'
  cat "$VERSIONS"
  echo '```'
} > "$REPORT"

echo; echo "Done -> $RUNDIR/"
echo "    report:     $REPORT"
echo "    per-config: $RUNDIR/chateval/<label>/ (metrics.json, *_transcript.json)"
command -v column >/dev/null && { echo; printf '%s\n' "${results[@]}" | column -s'|' -t; }
