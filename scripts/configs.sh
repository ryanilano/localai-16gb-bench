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

# Where results go.
OUTDIR="${OUTDIR:-./bench_results}"

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

# --- Benchmark parameters --------------------------------------------------
REPS=3                              # llama-bench repetitions per test
PROMPT_LEN=512                      # prompt-processing (prefill) test size
GEN_LEN=128                         # generation test size
DEPTHS=(0 4096 8192 16384 32768)   # context-fill depths to sweep
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
# This repo ships model-neutral: there are NO active models below. Uncomment a line
# (or add your own) to run. The Qwen3.6 block is a worked example that also shows how
# fields 4 and 5 are used — see ../model-benches/qwen36.md for the full analysis.
CONFIGS=(
  # ------------------------------------------------------------------------
  # EXAMPLE — Qwen3.6 4-bit sweep (commented out). Uncomment to run.
  # Qwen needs the system prompt (field 4) and the froggeric tool-call template
  # (field 5) for clean agentic/tool-call output.
  # ------------------------------------------------------------------------
  # --- stock 27B dense ---
  # "27B_IQ4_XS|unsloth/Qwen3.6-27B-GGUF:IQ4_XS|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # "27B_UD-Q4_K_XL|unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL|dense|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # --- stock 35B-A3B MoE ---
  # "35B_UD-IQ4_NL_XL|unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL_XL|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # "35B_UD-Q4_K_XL|unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"
  # "35B_UD-Q3_K_M|unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M|moe|You are Qwen, created by Alibaba Cloud. You are a helpful assistant.|../templates/qwen36-froggeric-v20.jinja"

  # --- uncensored / Heretic (optional — isolated-sandbox track only; see model-benches/qwen36.md §3) ---
  # "27B_Heretic_Youssofal|Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF:Q4_K_M|dense||../templates/qwen36-froggeric-v20.jinja"
  # 3-bit 27B Heretic: smaller, aims to fit the dense model fully on-GPU with KV room (Q4_K_M offloads).
  #   Q3_K_M is listed by the uploader (qwen36.md §3). Q3_K_L is NOT documented for this repo —
  #   confirm the tag exists on the HF page before uncommenting, or swap in an uploader that ships it.
  # "27B_Heretic_Youssofal_Q3_K_M|Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF:Q3_K_M|dense||../templates/qwen36-froggeric-v20.jinja"
  # "27B_Heretic_Youssofal_Q3_K_L|Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF:Q3_K_L|dense||../templates/qwen36-froggeric-v20.jinja"  # [NEEDS SOURCE: confirm Q3_K_L on HF]
  # "35B_Heretic_HauhauCS|fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_P|moe||../templates/qwen36-froggeric-v20.jinja"
)
