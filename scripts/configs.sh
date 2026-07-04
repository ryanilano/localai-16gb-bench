# ===========================================================================
# Shared config for the local-LLM benchmark scripts.  Edit freely.
# Sourced by prefetch.sh, run-bench.sh and run-quality.sh.
#
# This file holds INFRASTRUCTURE and RUN-PROTOCOL settings only. Everything
# per-model (models, flags, system prompts, chat templates, MoE offload)
# lives in models.ini — that is now the ONE file you edit to register models.
# ===========================================================================

# --- Paths -----------------------------------------------------------------
# Directory that contains build/bin/ (your compiled llama.cpp).
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
LLAMA_CLI="$LLAMA_DIR/build/bin/llama-cli"

# Where results go.
OUTDIR="${OUTDIR:-./bench_results}"

# Where model GGUFs are stored. Defaults to the repo's ./models so the whole
# bundle is self-contained; override for a big disk: MODELS_DIR=/mnt/models.
# All three scripts use llama.cpp's own -hf resolver, and LLAMA_CACHE below
# points that resolver here — so the cache is shared with zero mismatch.
# NOTE: repo-local models means `git clean -xdf` (or deleting the repo) nukes
# your downloads. models/ must stay in .gitignore — these files are 14-22 GB.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
export LLAMA_CACHE="$MODELS_DIR"

# --- Model registry ----------------------------------------------------------
# models.ini is parsed by ini.sh (sourced here so all scripts get the API).
INI_FILE="${INI_FILE:-./models.ini}"
source ./ini.sh

# --- Benchmark parameters (the measurement protocol) -------------------------
REPS=3                              # llama-bench repetitions per test
PROMPT_LEN=512                      # prompt-processing (prefill) test size
GEN_LEN=128                         # generation test size
DEPTHS=(0 4096 8192 16384 32768)   # context-fill depths to sweep
PORT=8080                          # llama-server port for the quality pass
PREFETCH_JOBS="${PREFETCH_JOBS:-3}" # parallel downloads in prefetch.sh; 2-4 is usually best
                                   # (more just thrashes the disk / trips HF rate limits).
                                   # Env-overridable: PREFETCH_JOBS=4 ./prefetch.sh

# Moved to models.ini [*] (per-model-overridable there):
#   KV_QUANT -> ctk / ctv          THREADS -> t
#   NCMOE_ALL -> n-cpu-moe         SYS_DEFAULT -> sys
#   CHAT_TEMPLATE -> template      (quality-pass -c 8192 -> quality.c)
