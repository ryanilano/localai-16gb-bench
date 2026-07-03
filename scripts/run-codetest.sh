#!/usr/bin/env bash
# ===========================================================================
# run-codetest.sh — OBJECTIVE pass/fail coding test for each active config.
# For the chosen challenge in codetests/<name>/ (prompt.txt + a test_*.py suite):
#   start llama-server, send the prompt, extract the Python code block from the
#   answer, run the reference unittest suite against it, and record PASS/FAIL
#   (+ how many assertions held). No human judgement — the reference tests judge.
#
# Reuses configs.sh, so ONLY= / PORT / QCTX / NGL / GEN all apply. To test just
# the DavidAU NEO-CODE quant, for example:
#   ONLY=NEO_CODE ./run-codetest.sh
# Pick a different challenge with CHALLENGE=<dir under codetests/>.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh

command -v jq >/dev/null      || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null    || { echo "Install curl"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required to run the reference tests"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }

CHALLENGE="${CHALLENGE:-lru}"            # which codetests/<name>/ to run
CDIR="codetests/$CHALLENGE"
[ -f "$CDIR/prompt.txt" ] || { echo "no prompt at $CDIR/prompt.txt (set CHALLENGE=<dir under codetests/>)"; exit 1; }
TESTFILE=$(ls "$CDIR"/test_*.py 2>/dev/null | head -1)
[ -n "$TESTFILE" ] || { echo "no reference test (test_*.py) in $CDIR"; exit 1; }
TESTMOD=$(basename "$TESTFILE" .py)

# Reasoning + code needs headroom (the model thinks, then writes the class). Low
# temperature keeps a correctness test near-deterministic. All env-overridable.
GEN="${GEN:-4096}"
QCTX="${QCTX:-8192}"
NGL_DEFAULT="${NGL:-99}"
TEMP="${TEMP:-0.2}"

SLUG=$(run_slug); RUNDIR="$OUTDIR/$SLUG"
mkdir -p "$RUNDIR/codetest"
VERSIONS="$RUNDIR/versions.txt"; REPORT="$RUNDIR/RUN.md"
capture_versions "$VERSIONS" "$LLAMA_SERVER"

start_server() {   # $1 = repo:quant ; remaining args = extra flags (e.g. --n-cpu-moe, --jinja ...)
  local repo="$1"; shift
  # A stale llama-server from a crashed prior run can hold $PORT and make bind fail.
  # We run exactly one server at a time, so clear any lingering one before binding.
  if pkill -f "$LLAMA_SERVER" 2>/dev/null; then echo "    cleared a stale llama-server before binding :$PORT"; sleep 2; fi
  # --no-mmproj: drop the vision projector these repos ship (the CLIP buffer OOMs a
  # 16 GB card on boot after full offload). We only send text.
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

# Pull Python out of the answer: concatenate every ``` fenced block. The model is
# asked for a single implementation block; concatenating is robust if it splits.
extract_code() { awk 'BEGIN{inb=0} /^[[:space:]]*```/{inb=!inb; next} inb{print}'; }

results=()   # "label|verdict|detail" rows for the report
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

  work="$RUNDIR/codetest/$label"; mkdir -p "$work"
  SRV_LOG="$work/_server.log"
  echo ">>> $label (challenge=$CHALLENGE, ngl=$ngl) ..."
  if ! start_server "$repo" -ngl "$ngl" "${moe_flags[@]}" "${tmpl_flags[@]}"; then
    stop_server
    if [ "$ngl" = "-1" ]; then
      echo "    server failed to start even at auto-fit — skipping $label"; results+=("$label|ERROR|no boot"); continue
    fi
    echo "    ngl=$ngl failed to boot — retrying with auto-fit (-ngl -1)"
    SRV_LOG="$work/_server.autofit.log"
    if ! start_server "$repo" -ngl -1 "${moe_flags[@]}" "${tmpl_flags[@]}"; then
      echo "    auto-fit also failed — skipping $label"; stop_server; results+=("$label|ERROR|no boot"); continue
    fi
  fi

  # Ask the model, capture the answer, extract code.
  prompt=$(cat "$CDIR/prompt.txt")
  body=$(jq -n --arg sys "$sys" --arg u "$prompt" --argjson n "$GEN" --argjson t "$TEMP" \
    '{messages: ((if ($sys|length) > 0 then [{role:"system",content:$sys}] else [] end)
                 + [{role:"user",content:$u}]),
      temperature:$t, top_p:0.95, top_k:20, max_tokens:$n}')
  resp=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
           -H 'Content-Type: application/json' -d "$body" || echo '{}')
  content=$(echo "$resp" | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // ""')
  printf '%s\n' "$content" > "$work/response.md"
  printf '%s\n' "$content" | extract_code > "$work/solution.py"

  # Run the reference suite against the model's solution.py (both live in $work).
  cp "$TESTFILE" "$work/$TESTMOD.py"
  testout="$work/test_output.txt"
  if [ ! -s "$work/solution.py" ]; then
    echo "(no python code block found in model response)" > "$testout"
    verdict="FAIL"; detail="no code"
  else
    ( cd "$work" && python3 -m unittest -v "$TESTMOD" ) > "$testout" 2>&1
    rc=$?
    ran=$(grep -oE "Ran [0-9]+ test" "$testout" | grep -oE "[0-9]+" | head -1); ran=${ran:-0}
    if [ "$rc" -eq 0 ] && [ "$ran" -gt 0 ]; then
      verdict="PASS"; detail="$ran/$ran"
    else
      fails=$(grep -oE "failures=[0-9]+" "$testout" | grep -oE "[0-9]+" | head -1); fails=${fails:-0}
      errs=$(grep -oE "errors=[0-9]+"   "$testout" | grep -oE "[0-9]+" | head -1); errs=${errs:-0}
      passed=$(( ran - fails - errs )); [ "$passed" -lt 0 ] && passed=0
      verdict="FAIL"; detail="$passed/$ran"
    fi
  fi
  echo "    $label -> $verdict ($detail)"
  results+=("$label|$verdict|$detail")
  stop_server
done

# Human-readable report: verdict table + provenance.
{
  echo "# Code test run — $SLUG"
  echo
  echo "**Challenge:** \`$CHALLENGE\` (reference suite: \`$CDIR/$TESTMOD.py\`)"
  echo
  echo "## Results"
  echo
  echo "| config | verdict | tests passed |"
  echo "| --- | --- | --- |"
  if [ "${#results[@]}" -gt 0 ]; then
    for r in "${results[@]}"; do IFS='|' read -r l v d <<< "$r"; echo "| $l | $v | $d |"; done
  else
    echo "| (none — no active CONFIGS) | | |"
  fi
  echo
  echo "## Provenance"
  echo
  echo '```'
  cat "$VERSIONS"
  echo '```'
} > "$REPORT"

echo; echo "Done -> $RUNDIR/"
echo "    report:     $REPORT"
echo "    per-config: $RUNDIR/codetest/<label>/ (solution.py, test_output.txt, response.md)"
command -v column >/dev/null && { echo; printf '%s\n' "${results[@]}" | column -s'|' -t; }
