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

command -v jq >/dev/null   || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null || { echo "Install curl"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }
# A global CHAT_TEMPLATE is optional now (per-model field 5 overrides it; both may be empty).
[ -z "$CHAT_TEMPLATE" ] || [ -f "$CHAT_TEMPLATE" ] || { echo "CHAT_TEMPLATE set but not found at $CHAT_TEMPLATE"; exit 1; }

mkdir -p prompts "$OUTDIR/quality"
GEN=768                 # max tokens per answer

# Seed a few example prompts on first run (add your own .txt files anytime).
if ! ls prompts/*.txt >/dev/null 2>&1; then
  printf 'Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Include 3 unit tests.\n' > prompts/01_palindrome.txt
  printf 'Refactor this for clarity and proper error handling:\ndef d(u):\n import requests;return requests.get(u).json()["data"]\n' > prompts/02_refactor.txt
  printf 'You are an agent in a sandbox. Give a step-by-step plan to find every TODO comment in a Git repo, then a single bash command that does it.\n' > prompts/03_agent_plan.txt
  echo "Seeded example prompts in ./prompts/ — replace them with your own real tasks."
fi

start_server() {   # $1 = repo:quant ; remaining args = extra flags (e.g. --n-cpu-moe, --jinja ...)
  local repo="$1"; shift
  "$LLAMA_SERVER" -hf "$repo" \
    -ngl 99 -fa on "$@" \
    -ctk "$KV_QUANT" -ctv "$KV_QUANT" -t "$THREADS" -c 8192 \
    --host 127.0.0.1 --port "$PORT" > "$OUTDIR/quality/_server.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 600); do                       # wait up to ~20 min for first download
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV_PID" 2>/dev/null || return 1     # server died
    sleep 2
  done
  return 1
}

stop_server() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; sleep 2; }

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label repo type sys tmpl <<< "$entry"
  moe_flags=(); [ "$type" = "moe" ] && moe_flags=(--n-cpu-moe "$NCMOE_ALL")

  # Resolve per-model system prompt and chat template, with global fallbacks.
  [ -n "$sys" ]  || sys="$SYS_DEFAULT"
  [ -n "$tmpl" ] || tmpl="$CHAT_TEMPLATE"
  tmpl_flags=()
  if [ -n "$tmpl" ]; then
    [ -f "$tmpl" ] || { echo "    template not found for $label: $tmpl — skipping"; continue; }
    tmpl_flags=(--jinja --chat-template-file "$tmpl")
  fi

  outdir="$OUTDIR/quality/$label"; mkdir -p "$outdir"

  echo ">>> starting $label ..."
  if ! start_server "$repo" "${moe_flags[@]}" "${tmpl_flags[@]}"; then
    echo "    server failed to start (see $OUTDIR/quality/_server.log) — skipping $label"
    stop_server; continue
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
done

echo; echo "Done. Review answers in $OUTDIR/quality/<label>/*.md"
