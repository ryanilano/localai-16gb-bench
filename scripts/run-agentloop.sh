#!/usr/bin/env bash
# ===========================================================================
# run-agentloop.sh — multi-turn tool-calling AGENT-LOOP eval per active config.
# The chat+agent (OpenClaude / Hermes) analogue of run-codetest.sh: instead of a
# single-shot prompt, it boots llama-server with the tool-call template (--jinja)
# and runs agentloop.py, which drives a scripted task that NEEDS several tool
# calls (read -> edit -> run -> observe) to finish. Objective completion is the
# scenario's own check command; tool-call reliability / latency / think-overhead
# are recorded alongside.
#
# Reuses configs.sh (ONLY / PORT / QCTX / NGL / GEN / TEMP) and the same server
# lifecycle as run-codetest.sh. Pick a scenario with SCENARIO=<dir under
# agenttests/> (default: fixbug). Example — just the lead dense pick:
#   ONLY=Heretic_NEO_CODE_IQ4_XS ./run-agentloop.sh
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh

command -v jq >/dev/null      || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null    || { echo "Install curl"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required to drive the agent loop"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }

SCENARIO="${SCENARIO:-fixbug}"
SDIR="agenttests/$SCENARIO"
[ -f "$SDIR/task.json" ] || { echo "no task at $SDIR/task.json (set SCENARIO=<dir under agenttests/>)"; exit 1; }
[ -d "$SDIR/seed" ]      || { echo "scenario $SCENARIO has no seed/ dir"; exit 1; }

# Agent loops re-send a growing scratchpad each turn, so give the answer room but
# keep it bounded (over-thinking eats this budget — that's part of what we measure).
GEN="${GEN:-2048}"
QCTX="${QCTX:-8192}"
NGL_DEFAULT="${NGL:-99}"
TEMP="${TEMP:-0.2}"

SLUG=$(run_slug); RUNDIR="$OUTDIR/$SLUG"
mkdir -p "$RUNDIR/agentloop"
VERSIONS="$RUNDIR/versions.txt"; REPORT="$RUNDIR/RUN.md"
capture_versions "$VERSIONS" "$LLAMA_SERVER"

# --- server lifecycle (identical contract to run-codetest.sh) ---------------
start_server() {   # $1 = repo:quant ; remaining args = extra flags (--n-cpu-moe, --jinja ...)
  local repo="$1"; shift
  if pkill -f "$LLAMA_SERVER" 2>/dev/null; then echo "    cleared a stale llama-server before binding :$PORT"; sleep 2; fi
  "$LLAMA_SERVER" -hf "$repo" --no-mmproj \
    -fa on "$@" \
    -ctk "$KV_QUANT" -ctv "$KV_QUANT" -t "$THREADS" -c "$QCTX" \
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

results=()   # "label|verdict|detail"
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl ngl <<< "$entry"
  moe_flags=(); [ "$type" = "moe" ] && moe_flags=(--n-cpu-moe "$NCMOE_ALL")
  [ -n "$sys" ]  || sys="$SYS_DEFAULT"
  [ -n "$tmpl" ] || tmpl="$CHAT_TEMPLATE"
  [ -n "$ngl" ]  || ngl="$NGL_DEFAULT"
  tmpl_flags=()
  if [ -n "$tmpl" ]; then
    [ -f "$tmpl" ] || { echo "    template not found for $label: $tmpl — skipping"; results+=("$label|ERROR|template missing"); continue; }
    tmpl_flags=(--jinja --chat-template-file "$tmpl")
  fi

  work="$RUNDIR/agentloop/$label"; mkdir -p "$work"
  SRV_LOG="$work/_server.log"
  echo ">>> $label (scenario=$SCENARIO, ngl=$ngl) ..."
  if ! start_server "$repo" -ngl "$ngl" "${moe_flags[@]}" "${tmpl_flags[@]}"; then
    stop_server
    if [ "$ngl" = "-1" ]; then echo "    no boot even at auto-fit — skipping"; results+=("$label|ERROR|no boot"); continue; fi
    echo "    ngl=$ngl failed to boot — retrying with auto-fit (-ngl -1)"
    SRV_LOG="$work/_server.autofit.log"
    if ! start_server "$repo" -ngl -1 "${moe_flags[@]}" "${tmpl_flags[@]}"; then
      echo "    auto-fit also failed — skipping"; stop_server; results+=("$label|ERROR|no boot"); continue
    fi
  fi

  if python3 agentloop.py --port "$PORT" --sys "$sys" --scenario "$SDIR" \
       --gen "$GEN" --temp "$TEMP" --out "$work/metrics.json" \
       --workdir "$work/sandbox" > "$work/summary.txt" 2> "$work/_agentloop.err"; then
    done_=$(jq -r '.task_completed'   "$work/metrics.json")
    valid=$(jq -r '.tool_calls_valid' "$work/metrics.json")
    tot=$(jq -r   '.tool_calls_total' "$work/metrics.json")
    drift=$(jq -r '.format_drift'     "$work/metrics.json")
    turns=$(jq -r '.turns'            "$work/metrics.json")
    trunc=$(jq -r '.truncated_loop'   "$work/metrics.json")
    think=$(jq -r '.think_chars'      "$work/metrics.json")
    med=$(jq -r   '.req_ms_median'    "$work/metrics.json")
    verdict=$([ "$done_" = "true" ] && echo PASS || echo FAIL)
    detail="done=$done_ tools=$valid/$tot drift=$drift turns=$turns trunc=$trunc think=${think}c med=${med}ms"
  else
    verdict="ERROR"; detail="agentloop.py failed (see _agentloop.err)"
  fi
  echo "    $label -> $verdict ($detail)"
  results+=("$label|$verdict|$detail")
  stop_server
done

{
  echo "# Agent-loop run — $SLUG"
  echo
  echo "**Scenario:** \`$SCENARIO\`  ·  GEN=$GEN QCTX=$QCTX TEMP=$TEMP"
  echo
  echo "## Results"
  echo
  echo "| config | verdict | detail |"
  echo "| --- | --- | --- |"
  if [ "${#results[@]}" -gt 0 ]; then
    for r in "${results[@]}"; do IFS='|' read -r l v d <<< "$r"; echo "| $l | $v | $d |"; done
  else
    echo "| (none — no active CONFIGS) | | |"
  fi
  echo
  echo "_verdict PASS = scenario check passed. detail: tools=valid/total tool calls, drift=raw <tool_call> not structured, trunc=hit max_turns, think=reasoning chars burned, med=median request ms._"
  echo
  echo "## Provenance"
  echo
  echo '```'
  cat "$VERSIONS"
  echo '```'
} > "$REPORT"

echo; echo "Done -> $RUNDIR/"
echo "    report:     $REPORT"
echo "    per-config: $RUNDIR/agentloop/<label>/ (metrics.json, *_transcript.json, sandbox/)"
command -v column >/dev/null && { echo; printf '%s\n' "${results[@]}" | column -s'|' -t; }
