# ===========================================================================
# Shared config for the local-LLM benchmark scripts.  Edit freely.
# Sourced by prefetch.sh, run-bench.sh and run-quality.sh.
# ===========================================================================

# --- Paths -----------------------------------------------------------------
# Directory that contains build/bin/ (your compiled llama.cpp).
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
LLAMA_CLI="$LLAMA_DIR/build/bin/llama-cli"

# Where results go. Repo root (not scripts/), resolved absolutely so it's the
# same dir no matter where a script is invoked from. Each run gets its own
# self-contained subfolder: $OUTDIR/<run-slug>/ (throughput.csv, versions.txt,
# RUN.md, json/, quality/). Override by exporting OUTDIR.
OUTDIR="${OUTDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bench_results}"

# OPTIONAL global chat template, applied to every model that doesn't set its own
# in CONFIGS field 5 (below). Leave empty to use each model's built-in template.
# Most modern models work fine with their built-in template; only set this (or a
# per-model field-5 template) when the stock template mangles tool-call XML.
# Example (Qwen3.6 tool-call fix): "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/templates/qwen36-froggeric-v20.jinja"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

# OPTIONAL global system prompt, applied to every model that doesn't set its own
# in CONFIGS field 4. Empty ⇒ no system message is sent at all.
# Example (Qwen needs this to perform): "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SYS_DEFAULT="${SYS_DEFAULT:-}"

# OPTIONAL: keep downloaded GGUFs on a specific disk. prefetch.sh, run-bench.sh and
# run-quality.sh all use llama.cpp's own -hf resolver, so whatever you set here is
# shared by all of them (no cache mismatch). Leave unset to use ~/.cache/huggingface.
# export LLAMA_CACHE="/mnt/models/llama-cache"

# --- Secrets: Hugging Face token for gated/private repos -------------------
# llama.cpp's -hf resolver authenticates with $HF_TOKEN when it is set. Gated or
# private GGUF repos (e.g. fredrezones55/…HauhauCS-Aggressive) return HTTP 401 and
# never download without it. Keep the real token OUT of git: put it in
# .local/secrets.env (the whole .local/ dir is gitignored), e.g.
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# The block below sources that file (only if HF_TOKEN isn't already in the env) and
# exports it so prefetch/bench/quality all see the same credential. An HF_TOKEN
# already exported in your shell always wins.
_secrets_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.local/secrets.env"
if [[ -z "${HF_TOKEN:-}" && -f "$_secrets_file" ]]; then
  set -a; source "$_secrets_file"; set +a
fi
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

# --- Benchmark parameters --------------------------------------------------
REPS=3                              # llama-bench repetitions per test
PROMPT_LEN=512                      # prompt-processing (prefill) test size
GEN_LEN=128                         # generation test size
DEPTHS=(0 4096 8192 16384 32768)   # context-fill depths to sweep (default profile)
# Long-context profile: `BENCH_PROFILE=longctx ./run-bench.sh` sweeps deeper, capped
# at 80k. Both Qwen3.6 models are 256K-native (model-benches/qwen36.md §"Date built"),
# so 80k needs no RoPE/YaRN scaling — quality there is real, not throughput-only. 80k is
# chosen to stay inside the SAFE VRAM margin of the lightest dense 27B configs (NEO_CODE
# IQ3_M safe ~85-88k, Heretic Q3_K_M safe ~77k) rather than probe their OOM edge. On the
# 16 GB card the MoE 35B configs clear 80k trivially (~10.6 KiB/tok KV, ~3 GB at 32k, VRAM
# never binds); the IQ4_XS-class dense configs OOM past 32k and log FAIL — that's the fit
# map, not a bug.
# MoE gets its own, much deeper sweep: experts live in RAM, only attention KV is on
# the GPU (~10.6 KiB/tok), so VRAM never binds — the ceiling is the 256K native window,
# not the card. Dense configs stay capped at 80k (see above). Default: MoE == dense.
DEPTHS_MOE=("${DEPTHS[@]}")
if [[ "${BENCH_PROFILE:-}" == "longctx" ]]; then
  DEPTHS=(0 16384 32768 49152 65536 81920)              # dense: 80k cap (81920 = 80 * 1024)
  DEPTHS_MOE=(0 32768 65536 131072 196608 261632)       # MoE: 256K native window (Qwen3.6 is 262144-native,
                                                        # per HF card). Top rung = 262144 - PROMPT_LEN(512) so the
                                                        # bench's own prefill stays inside the trained window (no YaRN).
fi
KV_QUANT="q8_0"                    # KV cache type (q8_0 halves KV vs f16)
THREADS=8                          # physical cores to use (tune to your CPU)
PORT=8080                          # llama-server port for the quality pass
PREFETCH_JOBS="${PREFETCH_JOBS:-3}" # parallel downloads in prefetch.sh; 2-4 is usually best
                                   # (more just thrashes the disk / trips HF rate limits).
                                   # Env-overridable: PREFETCH_JOBS=4 ./prefetch.sh

# MoE expert offload: 99 = ALL expert layers in DDR5 RAM, attention on GPU.
# Lower it to pull experts back onto the GPU until VRAM ~fills (= faster).
# Dense models ignore this.
NCMOE_ALL=99

# --- Test matrix -----------------------------------------------------------
# Format (5 pipe-delimited fields; fields 4 and 5 are OPTIONAL):
#
#   label | hf_repo:quant | type(dense|moe) | system_prompt | chat_template_path
#
#   label              short name used for output dirs/CSV rows.
#   hf_repo:quant      a real HF GGUF repo + a quant tag that matches a file in it.
#   type               dense | moe.  Only "moe" triggers --n-cpu-moe expert offload.
#   system_prompt      (optional) per-model system prompt. Empty ⇒ use SYS_DEFAULT.
#   chat_template_path (optional) per-model .jinja template (used by run-quality.sh's
#                      llama-server only; bench doesn't template). Empty ⇒ use
#                      CHAT_TEMPLATE, else the model's built-in template.
#
# Adding a model = adding one line here; it flows through prefetch → bench → quality
# unchanged. Drop any per-model template in ../templates/. Comment a line to skip it.
#
# ACTIVE SET: the Qwen3.6 16 GB sweep (9 configs) is uncommented and runs off `main`,
# so the bench box needs NO local edits — a plain `git pull` stays conflict-free. To go
# back to model-neutral, re-comment the lines below. The gated 35B_Heretic_HauhauCS line
# stays commented (needs an HF_TOKEN — see the Secrets block above and ../model-benches/
# qwen36.md §3). The 27B_UD-Q4_K_XL example is left commented (not part of the sweep).
CONFIGS=(
  # Qwen needs the system prompt (field 4) and the froggeric tool-call template (field 5)
  # for clean agentic/tool-call output.
  # --- stock 27B dense ---
  "27B_IQ4_XS|unsloth/Qwen3.6-27B-GGUF:IQ4_XS|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # "27B_UD-Q4_K_XL|unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # --- stock 35B-A3B MoE ---
  "35B_UD-IQ4_NL_XL|unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL_XL|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  "35B_UD-Q4_K_XL|unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  "35B_UD-Q3_K_M|unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"

  # --- uncensored / Heretic (optional — isolated-sandbox track only; see model-benches/qwen36.md §3) ---
  # "27B_Heretic_Youssofal|Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF:Q4_K_M|dense||../templates/qwen36-froggeric-v20.jinja"
  # 3-bit 27B Heretic: smaller, aims to fit the dense model fully on-GPU with KV room (Q4_K_M offloads).
  #   Q3_K_M is the smallest 3-bit tag the uploader ships (qwen36.md §3). Q3_K_L was tried but the
  #   repo does NOT publish it — the -hf resolver returns "no GGUF files found" and the model never
  #   loads (confirmed in the 2026-07-02 runs). Removed; use Q3_K_M for on-GPU 3-bit.
  "27B_Heretic_Youssofal_Q3_K_M|Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF:Q3_K_M|dense||../templates/qwen36-froggeric-v20.jinja"
  # HauhauCS-Aggressive 35B is a GATED/private repo → needs a valid HF_TOKEN (see Secrets block above),
  #   else the resolver returns HTTP 401 and it never downloads (confirmed in the 2026-07-02 runs).
  # "35B_Heretic_HauhauCS|fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_P|moe||../templates/qwen36-froggeric-v20.jinja"
  # HauhauCS "Balanced" 27B dense: abliteration + K_P imatrix quants that KEEP the reasoning
  #   trace (reasons out loud, adds a short disclaimer, then answers in full). Ships an mmproj
  #   (vision). IQ4_XS (~15 GB) is the apples-to-apples quant vs stock 27B; Q3_K_P (~14 GB) /
  #   IQ3_M (~13 GB) leave more KV room on-GPU. Q4_K_P (~18 GB) exceeds 16 GB VRAM — skip.
  "27B_HauhauCS_Balanced|HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced:IQ4_XS|dense||../templates/qwen36-froggeric-v20.jinja"
  "27B_HauhauCS_Balanced_Q3_K_P|HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced:Q3_K_P|dense||../templates/qwen36-froggeric-v20.jinja"

  # --- code-specialized finetune (experimental — NOT abliteration; judge on Phase-C coding pass, not card claims) ---
  # DavidAU NEO-CODE 27B dense: a code/creative FINETUNE with "IMatrix-MAX" quants (embeddings/
  #   output kept high-precision -> files run larger than a stock tag of the same name). Unlike the
  #   Heretic track there is no KLD drift proxy here, so its only justification is your own coding
  #   quality pass. IQ4_XS (~15.4 GB) matches the stock-27B offload regime; IQ3_M (~12.9 GB) fits
  #   with KV room. Q4_K_M (~16.9 GB) exceeds 16 GB VRAM — heavy offload, skip.
  "27B_NEO_CODE_IQ4_XS|DavidAU/Qwen3.6-27B-NEO-CODE-Di-IMatrix-MAX-GGUF:IQ4_XS|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  "27B_NEO_CODE_IQ3_M|DavidAU/Qwen3.6-27B-NEO-CODE-Di-IMatrix-MAX-GGUF:IQ3_M|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
)
