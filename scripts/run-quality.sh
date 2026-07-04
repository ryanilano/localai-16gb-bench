#!/usr/bin/env bash
# ===========================================================================
# run-quality.sh — unattended quality pass (model-neutral).
# For each model in models.ini: start llama-server (with that model's chat
# template + system prompt, if any), run every prompt in prompts/*.txt through
# /v1/chat/completions at coding sampling settings, save each answer to
# quality/<label>/<prompt>.md, then stop the server and move on.
#
# Per-model server flags come from models.ini via ini_flags (quality scope):
# [*] defaults plus section overrides, n-cpu-moe applied only when type = moe,
# context size via quality.c. System prompt = `sys` key ([*] as fallback);
# chat template = `template` key ([*] as fallback; path relative to repo root).
# --dry-run: print each composed llama-server command; run nothing.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh
source ./versions.sh
DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1

command -v jq >/dev/null   || { echo "Install jq:  sudo apt install jq"; exit 1; }
command -v curl >/dev/null || { echo "Install curl"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER — set LLAMA_DIR in configs.sh"; exit 1; }

mapfile -t LABELS < <(ini_sections)
[ "${#LABELS[@]}" -gt 0 ] || { echo "No models registered — uncomment or add sections in models.ini"; exit 1; }

mkdir -p prompts "$OUTDIR/quality"
GEN=768                 # max tokens per answer

# Resolve a template path from models.ini: absolute stays as-is, relative is
# anchored at the repo root (so `template = templates/foo.jinja` just works).
resolve_tmpl() {
  case "$1" in ("") echo "";; (/*) echo "$1";; (*) echo "$REPO_ROOT/$1";; esac
}

if [ "$DRY_RUN" -eq 1 ]; then
  for label in "${LABELS[@]}"; do
    repo="$(ini_get "$label" hf)"
    sys="$(ini_get "$label" sys)"
    tmpl="$(resolve_tmpl "$(ini_get "$label" template)")"
    mapfile -t FLAGS < <(ini_flags "$label" quality)
    tmpl_str=""; [ -n "$tmpl" ] && tmpl_str=" --jinja --chat-template-file $tmpl"
    echo "[dry-run] $label:"
    echo "  $LLAMA_SERVER -hf $repo ${FLAGS[*]}$tmpl_str --host 127.0.0.1 --port $PORT"
    echo "  system prompt: ${sys:-(none)}"
  done
  exit 0
fi

# Version-stamp this quality pass alongside the answers, so a set of quality
# outputs records the exact toolchain that produced it.
SLUG=$(run_slug)
VERSIONS="$OUTDIR/quality/versions_$SLUG.txt"
REPORT="$OUTDIR/quality/RUN_$SLUG.md"
capture_versions "$VERSIONS" "$LLAMA_SERVER"

# Seed a few example prompts on first run (add your own .txt files anytime).
if ! ls prompts/*.txt >/dev/null 2>&1; then
  printf 'Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Include 3 unit tests.\n' > prompts/01_palindrome.txt
  printf 'Refactor this for clarity and proper error handling:\ndef d(u):\n import requests;return requests.get(u).json()["data"]\n' > prompts/02_refactor.txt
  printf 'You are an agent in a sandbox. Give a step-by-step plan to find every TODO comment in a Git repo, then a single bash command that does it.\n' > prompts/03_agent_plan.txt
  echo "Seeded example prompts in ./prompts/ — replace them with your own real tasks."
fi

start_server() {   # $1 = repo:quant ; remaining args = per-model flags from ini_flags + template flags
  local repo="$1"; shift
  "$LLAMA_SERVER" -hf "$repo" \
    "$@" \
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

ran_labels=()   # models actually exercised this pass, for the run report
for label in "${LABELS[@]}"; do
  repo="$(ini_get "$label" hf)"
  [ -n "$repo" ] || { echo "SKIP $label — no 'hf =' key in models.ini"; continue; }

  # Per-model system prompt and chat template ([*] section is the global fallback).
  sys="$(ini_get "$label" sys)"
  tmpl="$(resolve_tmpl "$(ini_get "$label" template)")"
  tmpl_flags=()
  if [ -n "$tmpl" ]; then
    [ -f "$tmpl" ] || { echo "    template not found for $label: $tmpl — skipping"; continue; }
    tmpl_flags=(--jinja --chat-template-file "$tmpl")
  fi

  # All per-model server flags (defaults + overrides, MoE gate, quality.c).
  mapfile -t FLAGS < <(ini_flags "$label" quality)

  outdir="$OUTDIR/quality/$label"; mkdir -p "$outdir"

  echo ">>> starting $label ..."
  if ! start_server "$repo" "${FLAGS[@]}" "${tmpl_flags[@]}"; then
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
  ran_labels+=("$label -> $repo")
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
    echo "- (none — no active models in models.ini)"
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

echo; echo "Done. Review answers in $OUTDIR/quality/<label>/*.md"
echo "    provenance: $VERSIONS"
echo "    report:     $REPORT"
