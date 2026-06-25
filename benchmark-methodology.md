# Benchmark Methodology: local LLMs on a 16 GB NVIDIA GPU

The "why" behind the scripts: how to empirically pick a daily-driver local model for a 16 GB
NVIDIA card under `llama.cpp`, for **any** GGUF model family, dense or MoE. The model-specific
numbers live in `model-benches/` (see [`model-benches/qwen36.md`](model-benches/qwen36.md) for a
full worked example). This doc is model-neutral.

## The loop

1. **Register** the models you want to compare, one line each in `scripts/configs.sh` `CONFIGS`.
2. **Prefetch** (`prefetch.sh`): download every GGUF up front so the sweep never stalls mid-run.
3. **Speed + fit sweep** (`run-bench.sh`): `llama-bench` over every `config × depth`; one CSV row
   per cell, with peak VRAM/RAM sampled alongside.
4. **Quality pass** (`run-quality.sh`): boot `llama-server` per model, answer your real prompts,
   save one markdown file per answer.
5. **Decide**: combine the CSV (speed/fit) with the quality outputs and pick a winner.

## How to add a model

The scripts are a model-neutral engine driven entirely by the `CONFIGS` matrix. You never edit a
script to add a model; you add **data**:

1. Add one `CONFIGS` line: `label|hf_repo:quant|type[|system_prompt][|template_path]`
   (fields 4-5 optional; see `scripts/configs.sh` for the full format docs).
2. If the model needs a fixed chat template (e.g. its stock template mangles tool-call XML), drop
   the `.jinja` in `templates/` and point field 5 at it (`../templates/<file>.jinja`).
3. Run `./prefetch.sh && ./run-bench.sh && ./run-quality.sh`; the new model flows through unchanged.
4. (Optional) Write up your findings in `model-benches/<model>.md`, using `qwen36.md` as the template.

Quality prompts in `scripts/prompts/*.txt` are **shared** across all models on purpose: running
the same tasks against every model is what makes the comparison fair.

## Reading the throughput CSV

Columns: `label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status`.

- **Sort by `tg_tok_s` (desc)** → your token-generation speed ranking (the number you feel daily).
- **Filter `status=FAIL`** → the **fit ceiling**: which quant dies at which context depth. The CSV
  doubles as a fit map because an OOM at a given depth just writes `FAIL` and the sweep continues.
- **Watch `ram_used_peak_mib` on MoE rows** → if it's near your system-RAM cap, you're at the RAM
  wall, not the VRAM wall. A MoE row that `FAIL`s while `vram_peak_mib` looks fine is almost always
  the system OOM-killer: raise the RAM limit or drop to a smaller MoE quant.

## Dense vs MoE on a 16 GB card (the load-bearing idea)

A MoE model must hold **all** expert weights resident: the router can pick any expert for any
token. "A3B"-style names describe *compute per token* (few active params → fast), **not** memory
footprint. So a MoE GGUF is often larger than its "active params" suggest and won't fit 16 GB VRAM.

`llama.cpp`'s `--n-cpu-moe` offloads idle expert FFN tensors to system RAM while attention stays on
the GPU. This is **tolerant for MoE** (only a few experts fire per token, so RAM bandwidth rarely
bottlenecks) but **costly for dense** (every parameter fires every token). Practical consequence on
16 GB: a big MoE offloaded to RAM can feel *faster* than a dense model that barely spills, and the
real ceiling for MoE is your **system RAM**, not VRAM. Tune `NCMOE_ALL` down to pull experts back
onto the GPU until VRAM ~fills, for more speed.

## General caveats (verify per model)

- **Low-bit quants can be runtime-sensitive.** A specific CUDA / `llama.cpp` build can make a given
  low-bit quant emit gibberish. Pin a known-good toolchain and **sanity-check the actual output**,
  not just that it loads. (Example: Qwen3.6 4-bit breaks on CUDA 13.2, see the case study.)
- **Some models need a specific system prompt** to perform well. Set it per-model in `CONFIGS`
  field 4. (Example: Qwen needs `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`)
- **Stock chat templates sometimes mangle tool-call XML.** If you need clean agentic/tool-call
  output, supply a fixed template per-model in field 5. (Example: Qwen3.6 + the froggeric template.)
- **Judge uncensored/abliterated variants on your own quality outputs**, not the publisher's
  refusal/KLD numbers, and keep them on an isolated network.

## Choosing the winner

Combine the speed/fit CSV with the quality markdown. If two models are close on quality, prefer the
one with more context headroom (lower VRAM at your max depth). Lock the winning `llama-server`
command (tuned `--n-cpu-moe`, `-c`, KV flags, and any `--chat-template-file`) into a systemd unit or
launch script, and archive the `bench_results/` CSV + quality outputs as your baseline for the next
model generation.
