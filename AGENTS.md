# AGENTS.md

This file provides guidance to coding agents (Claude Code, and any other agent that reads
`AGENTS.md`) when working in this repository.

## What this is

A self-contained bundle of **documentation + bash automation** for benchmarking GGUF quants (dense
or MoE) of any local LLM under `llama.cpp` on a 16 GB NVIDIA GPU. The **engine is model-neutral**
(driven entirely by the `models.ini` registry); the repo ships model-neutral, with the Qwen3.6
example as a commented-out block in `models.ini` you can uncomment or replace with your own. The numbers in
the worked example (`model-benches/qwen36.md`) are the reference figures for one configuration: a
Proxmox LXC container with a passed-through RTX 4070 Ti Super 16 GB on a 7800X3D host, with 24 GB
RAM / 8 cores allocated to the container (the 16 GB VRAM and ~22 GB usable RAM are the binding
limits). Adapt those limits to whatever hardware runs it.

There is **no application to build and no test suite**; the deliverable is the benchmark run. The
"source" is four bash scripts in `scripts/` plus optional per-model chat templates in `templates/`.
The markdown files are the spec: `INSTALL.md` is the ordered runbook (the operator follows it top to
bottom); `benchmark-methodology.md` is the model-neutral rationale behind every choice the scripts
encode; `model-benches/<model>.md` are per-model worked writeups. When changing a script, keep it
consistent with all of these. They are the contract.

## Commands

All scripts live in and are run from `scripts/`. They `source ./configs.sh`, so run them from
that directory.

```bash
chmod +x scripts/*.sh
bash -n scripts/run-bench.sh && bash -n scripts/run-quality.sh   # lint (syntax-check)
bash scripts/test-ini.sh                                         # parser test suite (the only test runner)
./run-bench.sh --dry-run                                         # print composed commands, execute nothing

cd scripts
./prefetch.sh        # one-time: download every model in models.ini and integrity-check
./run-bench.sh       # speed + fit sweep over models.ini × DEPTHS -> ../bench_results/throughput_<stamp>.csv
./run-quality.sh     # start llama-server per model, answer every prompts/*.txt -> ../bench_results/quality/<label>/<prompt>.md
```

Runtime dependencies on the target box: a compiled `llama.cpp` (with `-DGGML_CUDA=ON -DLLAMA_CURL=ON`),
`jq`, `curl`, and `nvidia-smi`. The scripts assume Linux (`free -m`, `nvidia-smi`); they are authored
on macOS but **not meant to run here**. They run on the GPU box (the reference is a Proxmox container).

## How the pieces fit together

```
models.ini   ← the ONE file an operator edits: the model registry. One [section] per model
               ([*] = shared defaults). Reserved keys hf/type/sys/template; every other
               key = value passes through to the llama command line.
   ▲
   │ parsed by ini.sh (ini_sections / ini_get / ini_flags)
   │
configs.sh   ← infra + measurement protocol: LLAMA_DIR, OUTDIR, MODELS_DIR (exported as
               LLAMA_CACHE — the shared model store, default <repo>/models), DEPTHS, REPS,
               PROMPT_LEN, GEN_LEN, PORT, PREFETCH_JOBS. Sources ini.sh; sourced by all
               three scripts.
   ▲
   │ source ./configs.sh
   │
prefetch.sh ─┬─ run-bench.sh ─┬─ run-quality.sh
             │                │
        all use llama.cpp's own `-hf` resolver to download/cache, so the cache layout is
        identical across the three (no mismatch). Never swap this for huggingface-cli.
```

- **The models.ini registry is the central abstraction.** One `[section]` per model; the section
  name is the label used in CSV rows and output dirs. Reserved keys: `hf = repo:quant` (resolved
  via llama.cpp's `-hf` downloader — never a local path), `type = dense|moe`, `sys`, `template`.
  Every other `key = value` passes through to the llama command line; `[*]` holds shared defaults
  (per-section keys win), and a `bench.` / `quality.` prefix scopes a key to one script.
  `type=moe` is the only thing that lets `n-cpu-moe` through (expert offload to RAM); dense
  models never see it. **Adding a model = adding one section** (plus, if needed, a `.jinja` in
  `templates/`), and it flows through prefetch → bench → quality unchanged. This is the
  model-neutral design: the scripts are a generic engine, model-specifics are data. Do **not**
  fork the scripts per-model.
- **All parsing lives in `ini.sh`** — the scripts never read the file directly. Rules the parser
  encodes (tested by `test-ini.sh`; keep tests in sync): an EMPTY value skips a key, but `0` does
  NOT (`ngl = 0` is a real setting — deliberate divergence from launchers that drop zeroes);
  keys in `SHORT_FLAGS` get one dash (`-ngl`), all others two (`--n-cpu-moe`), because
  llama.cpp's parsers match option strings exactly and reject the wrong dash count.
- **`run-bench.sh`** uses `llama-bench` (no chat template, it doesn't do templating), composes per-model flags via `ini_flags <label> bench` and sweeps each
  model across every `DEPTHS` value, samples peak VRAM/RAM in a background `sample_mem` loop, and
  appends one CSV row per `config × depth`. An OOM writes `status=FAIL` for that row and continues,
  so the CSV doubles as the fit map (which quant dies at which context depth).
- **`run-quality.sh`** uses `llama-server` (which *does* need a template, if any). Per model it
  resolves the system prompt (`sys` key, `[*]` as fallback) and chat template (`template` key,
  `[*]` as fallback, else the model's built-in; relative paths anchor at the repo root), composes
  server flags via `ini_flags <label> quality` (context size = `quality.c`), boots a server, POSTs each prompt to
  `/v1/chat/completions` at coding sampling settings (temp 0.6 / top_p 0.95 / top_k 20), writes one
  markdown file per answer, then kills the server and moves on. The system message is omitted
  entirely when no system prompt is set. Seeds 3 example prompts on first run if `prompts/` is empty.

## Load-bearing facts the scripts encode (don't "fix" these without reading the methodology)

- **MoE ≠ small footprint.** Active-param names (e.g. "A3B") describe compute per token, not memory;
  all expert weights must be resident. A MoE Q4 GGUF often exceeds 16 GB VRAM, so experts are
  offloaded to RAM via `--n-cpu-moe`. The real ceiling for MoE is **system RAM**, not VRAM.
- **`--n-cpu-moe` / MoE flag is applied only when `type=moe`.** The gate lives in exactly one
  place — `ini_flags` in `ini.sh` — not in the runner scripts. Change it there or nowhere.
- **System prompt and chat template are per-model, not global constants.** They are the `sys` and
  `template` keys in `models.ini` (with the `[*]` section as the global fallback). Some models need a specific system
  prompt to perform, and some need a fixed template so tool-call XML isn't mangled. The Qwen3.6
  specifics (its required system prompt and the froggeric v20 template in `templates/`) are an
  **example**, documented in `model-benches/qwen36.md`, not a universal rule.
- **Low-bit quants can be runtime-sensitive.** A given CUDA / `llama.cpp` build can make a low-bit
  quant emit gibberish; pin a known-good toolchain and sanity-check output. (Example: Qwen3.6 4-bit
  breaks on CUDA 13.2, see `model-benches/qwen36.md`.)
- **Uncensored/abliterated configs** belong on an isolated-sandbox track and should be judged on the
  quality outputs, not vendor refusal/KLD numbers.

## Adding a model (the canonical recipe)

1. Add a `[label]` section to `models.ini`: `hf = repo:quant`, `type = dense|moe`, plus optional
   `sys` / `template` keys.
2. If the model needs a fixed template, drop it in `templates/` and set `template = templates/<file>.jinja`
   (repo-root-relative).
3. `./prefetch.sh && ./run-bench.sh && ./run-quality.sh`.
4. (Optional) Write up findings in `model-benches/<model>.md`, using `qwen36.md` as the template.

Quality prompts (`scripts/prompts/*.txt`) are **shared** across all models on purpose: same tasks,
fair comparison. Don't fork them per-model.

## Outputs

Everything lands in `bench_results/` (gitignored via `.gitignore`): `throughput_<stamp>.csv`,
per-run `json/` + `.log`, and `quality/<label>/<prompt>.md`. The CSV columns are
`label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status`.
