# 16 GB NVIDIA Local-LLM Benchmark Bundle

A repeatable way to benchmark **any GGUF model family** (dense or MoE) at 4-bit and beyond under
`llama.cpp` on a **16 GB NVIDIA GPU**, and pick a daily driver from real speed/fit/quality numbers.

The **engine is model-neutral** — the scripts are driven entirely by the `models.ini` registry, so
any model family drops in. The repo ships model-neutral: the Qwen3.6 example (the worked example
below) lives as a **commented-out** block in `models.ini`, so uncomment it to reproduce the example,
or add your own `[section]`s. The
numbers and fit ceilings in the included worked example were measured on one **reference
configuration**, a Proxmox LXC container with a passed-through **RTX 4070 Ti Super (16 GB)** on a
**7800X3D** host, with **24 GB RAM and 8 cores allocated to the container**. Those are the
container's allocated resources, so adapt the limits to whatever you run on.

## Start here

**Open `INSTALL.md` and follow it top to bottom.** That's the whole job: ordered, copy-paste
commands with checkboxes. Everything else is support material.

## Serve a model

Once `llama.cpp` is built and the model is fetched (per `INSTALL.md`), these launch `llama-server`
with the **exact flags each config was benchmarked at**, so the running server behaves like the thing
that was measured. They serve an OpenAI-compatible API on `http://127.0.0.1:8080`. _(Picks are from
the included Qwen3.6 worked example — see `model-benches/qwen36.md`.)_

```bash
cd scripts

# Daily driver — best all-round quality + long context.
# Dense Heretic NEO-CODE IQ3_M, ~80k ctx, ~40 tok/s, q8_0 KV.
./serve-27b-uncensored.sh

# Max dense quality with real context.
# Same model at higher-fidelity IQ4_XS; q4_0 KV unlocks ~49k ctx (q8_0 caps ~16k).
./serve-27b-iq4xs.sh

# Max context + fastest generation.
# 35B MoE UD-Q3_K_M, ~256k ctx at ~6 GB VRAM (RAM-bound), ~58 tok/s.
./serve-35b-moe.sh
```

Every tunable is env-overridable, e.g. `CTX=32768 PORT=9000 HOST=0.0.0.0 ./serve-27b-uncensored.sh`
(set `HOST=0.0.0.0` to reach it from another machine). Each script's header documents why that config
was picked and its caveats.

## What's in the box

```
INSTALL.md                  ← the runbook. Do this.
benchmark-methodology.md    ← the "why": the loop, how to read the CSV, dense-vs-MoE, how to add a model.
model-benches/              ← per-model worked writeups (analysis, quant ranking, caveats).
  qwen36.md                 ←   a full Qwen3.6 example you can copy for your own model.
templates/                  ← optional per-model chat templates (drop your model's .jinja here).
  qwen36-froggeric-v20.jinja←   Qwen3.6 tool-call template (referenced by the Qwen example config).
scripts/
  models.ini         ← the ONE file you edit: register models ([*] defaults + per-model overrides)
  configs.sh         ← infra + measurement protocol (LLAMA_DIR, depths, reps, output paths)
  ini.sh             ← models.ini parser, sourced by all scripts (tested by test-ini.sh)
  prefetch.sh        ← download all models up front (run once) -> ./models by default
  run-bench.sh       ← unattended speed + fit sweep -> CSV
  run-quality.sh     ← unattended quality outputs -> markdown per model
  serve-*.sh         ← launch llama-server on a picked config (see "Serve a model" above)
```

## 60-second version

```bash
git clone https://github.com/ryanilano/localai-16gb-bench.git
cd localai-16gb-bench

chmod +x scripts/*.sh
sudo apt install -y jq
nano scripts/configs.sh        # set LLAMA_DIR (skip if llama.cpp is at ~/llama.cpp)
nano scripts/models.ini        # register your models (one [section] each)

cd scripts
./prefetch.sh                  # downloads your models (run once, slow)
./run-bench.sh                 # speed + fit sweep -> ../bench_results/<run>/throughput.csv
./run-quality.sh               # add your own prompts/*.txt first; outputs -> ../bench_results/<run>/quality/
./publish-results.sh           # optional: push bench_results/ to a private GitHub repo (gh CLI)
```

Open the CSV in a spreadsheet, sort by `tg_tok_s`, read the quality `.md` files, pick a winner.

Results live at the repo root under `bench_results/`, and **each run is a self-contained folder**
named `<date>_<time>-<host>` (e.g. `bench_results/2026-07-02_143000-debianbox/`) holding that run's
`throughput.csv`, `json/`, `quality/`, and its provenance — so re-runs and different model sets never
overwrite each other.

Every run is **version-stamped**: each run folder includes a `versions.txt` provenance record (NVIDIA
driver, CUDA toolkit, resolved `llama.cpp` binary + git commit, GPU/VRAM, power cap, timestamp) and a
human-readable `RUN.md` report — so a result always says exactly which toolchain produced it, and a run
that lands on the known-bad CUDA 13.2 warns loudly.

### Registering a model

`scripts/models.ini` is the only place a model lives. One section per model:

```ini
[my-model-q4]
hf = org/My-Model-GGUF:Q4_K_M
type = dense            ; or moe — the only key that triggers --n-cpu-moe
```

Shared defaults live in the `[*]` section (KV quant, threads, ngl, MoE offload) and any
key can be overridden per model. Keys other than the reserved four (`hf`, `type`, `sys`,
`template`) pass straight through to the llama.cpp command line, so a model that needs
`temp = 1.0` or `quality.b = 2048` just says so in its section — no script edits, ever.
Prefix a key with `bench.` or `quality.` to apply it to only one script. Full format
docs are in the header of `models.ini`.

GGUFs download to the repo's `./models/` by default (gitignored); point `MODELS_DIR`
at a bigger disk to relocate everything.

## Two things that will save you a headache

1. **Sanity-check low-bit output.** A specific CUDA / `llama.cpp` version can make a given low-bit
   quant emit gibberish. Pin a known-good toolchain and read the actual output; don't just check
   that it loads. (Example: Qwen3.6 4-bit breaks on CUDA 13.2, see `model-benches/qwen36.md`.)
2. **MoE is bound by system RAM, not VRAM.** A MoE model parks its idle experts in RAM via
   `--n-cpu-moe`; set your RAM limit high enough (the Qwen 35B-A3B example needs ~22 GB) or the
   OOM-killer takes it mid-load. Dense models won't hit this.

Full detail in `INSTALL.md` Step 0 and `benchmark-methodology.md`.
