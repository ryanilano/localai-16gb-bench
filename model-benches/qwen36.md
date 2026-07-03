# Qwen3.6 4-bit Benchmark: Worked Example

> This is a **model-bench**: a worked example of the general process in
> [`../benchmark-methodology.md`](../benchmark-methodology.md), applied to Qwen3.6. The reasoning,
> quant ranking, and Qwen-specific caveats (system prompt, tool-call template, the CUDA 13.2 ban)
> live here. Use it as the template for benchmarking your own model: copy it to
> `model-benches/<your-model>.md`.

**Test environment:** every number here was measured inside a **Proxmox LXC container** with a passed-through **RTX 4070 Ti Super (16 GB)**, on a **7800X3D** host, with **24 GB RAM and 8 cores allocated to the container** (≈22 GB usable before the cgroup OOM-killer trips). Those are the **container's allocated resources**, not bare-metal specs; treat the fit ceilings and memory limits below as the reference figures for this configuration and adapt them to whatever you run on.

**Goal:** Empirically compare every viable 4-bit quant of **Qwen3.6-27B (dense)** and **Qwen3.6-35B-A3B (MoE)** under **llama.cpp**, and pick a daily driver.

**Date built:** June 2026. Models released April 2026 (27B on the 22nd, 35B-A3B on the 16th). Both are Apache 2.0, multimodal, 256K-context, and ship MTP heads.

---

## 0. Read this first: one load-bearing misconception to clear up

A common assumption is that the 35B-A3B *"fits fully in VRAM at Q4_K_M (active params only)."* **That is false, and it changes the whole plan.**

A MoE model must hold **all** expert weights resident: the router can pick any expert for any token. "A3B" describes *compute per token* (≈3 B active → fast), **not** memory footprint. The 35B-A3B Q4 GGUF is **~20-22 GB of weights**. It does **not** fit in 16 GB.

So the corrected mental model:

| | 27B **dense** | 35B-A3B **MoE** |
|---|---|---|
| Q4_K_M weights | ~15.2 GB | ~20-22 GB |
| Fits 16 GB fully? | Barely not (no room for KV) | No, not close |
| Offload behavior | **Hurts**: every param fires each token, bottlenecked by DDR5 (~90 GB/s vs 672 GB/s VRAM) | **Tolerant**: only ~3 B active, so idle expert FFN tensors sit in RAM cheaply (`--n-cpu-moe` / `-ot`) |
| MTP speedup | **Large (1.5-2×)**: heavy dense decode has lots to save | **Small (1.15-1.25×)**: baseline decode already cheap |

Two consequences worth internalizing:

1. **Both models offload on a 16 GB card.** The dense one offloads a *little* and pays a *lot* per GB; the MoE offloads a *lot* and pays *little*. That's why the MoE feels faster despite being bigger.
2. **MTP partially closes the gap the other way**: it accelerates the dense model far more than the MoE. Don't decide on raw baseline tok/s alone; decide on *MTP-on* numbers, since that's how you'll actually run them.

**The real ceiling is the 24 GB of RAM allocated to the container**, not VRAM. MoE expert offload parks ~16-18 GB in RAM; add OS + KV overflow and you're near the wall. Watch for swap.

---

## 1. Engine: llama.cpp only at this VRAM tier

For single-user local on 16 GB, **llama.cpp is the only engine that runs both models at 4-bit.** Its decisive feature is *expert-aware* CPU offload (`--n-cpu-moe`, `-ot exps=CPU`): it can park the MoE's idle experts in DDR5 RAM and keep attention on the GPU. Everything in this playbook targets it.

> ### Why vLLM was dropped
> vLLM (and SGLang) need the model resident in VRAM: they have **no expert-aware offload**, only a crude blanket `--cpu-offload-gb` that's slow and not MoE-smart. On 16 GB:
> - **27B dense, 4-bit AWQ (~15 GB)** + engine/KV overhead → boots only at trivial context, OOMs in practice.
> - **35B-A3B, 4-bit AWQ/GPTQ (~20 GB+)** → does not fit at all (a 32 GB RTX 5090 already burns ~31 GB on it).
>
> vLLM's real strengths are high-concurrency batching (~2× on parallel requests) and tensor/expert parallelism, and those only pay off with **≥24-32 GB VRAM or a second GPU**. Revisit it **only** if you later stand up a shared inference endpoint feeding several concurrent agent sandboxes from a bigger card; at that point it becomes the right tool. For this single-GPU 16 GB setup it loses to llama.cpp on every single-stream metric, so it's excluded from the bench rather than run for a foregone-conclusion result.

---

## 2. The quants to actually test ("all 4-bit"), ranked

Not all 4-bit quants are peers. Unsloth's own KL-divergence benchmarks put **UD-Q4_K_XL ahead of every other Q4 while being ~8 GB smaller**: it spends Q5_K on important matrices and Q4_K elsewhere, where plain `Q4_K_M` just uses uniform Q4_K with Q6_K on a few. So `Q4_K_M` is a *reference point*, not a real contender. Pull this spread for **each** model (sizes approximate, vary by builder; confirm with `ls -lh`):

| Quant tag | ~Size 27B | ~Size 35B-A3B | Role |
|---|---|---|---|
| **`UD-Q4_K_XL`** ⭐ | ~16 GB | ~22 GB | **Best 4-bit quality/size.** Default quality pick. Dynamic Q5_K on key layers. |
| **`UD-IQ4_NL_XL`** ⭐ | ~14.5 GB | ~19.5 GB | New Unsloth non-linear i-quant, **CPU/RAM-friendly**: matters for the MoE since its experts run on DDR5. Top candidate for the **MoE**. |
| `IQ4_XS` | ~13.8-15 GB | ~19 GB | Most compressed (~4.25 bpw), faster generation. **Best shot at the 27B dense fully on GPU.** |
| `Q4_K_M` | ~15.2 GB | ~22 GB | **Baseline comparison only — don't run locally.** A published reference point so your numbers line up with the world's; dominated by UD-Q4_K_XL on this hardware. |
| `Q4_K_S` *(optional)* | ~14.2 GB | ~20 GB | Bracketed by IQ4_XS and Q4_K_M; little unique value. Skip unless a fit gap appears. |

**Quick mapping:**
- **27B dense → lead with `IQ4_XS`** (fit it on the GPU), compare against `UD-Q4_K_XL` (quality).
- **35B-A3B MoE → lead with `UD-IQ4_NL_XL`** (CPU-friendly experts), compare against `UD-Q4_K_XL` (quality).
- Skip the legacy `Q4_0/Q4_1/Q4_NL` ARM/Apple formats: no benefit on CUDA.

**For MTP runs you need the MTP-specific GGUFs** (standard GGUFs lack the prediction heads):
- `unsloth/Qwen3.6-27B-MTP-GGUF` (e.g. `:UD-Q4_K_XL`)
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (e.g. `:UD-Q4_K_XL`)
- Canonical alternative: `ggml-org/Qwen3.6-27B-MTP-GGUF`, `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`

---

## 3. Uncensored / Heretic track (optional)

If you want a variant that won't false-refuse on benign agentic, security-research, or coding prompts inside your sandbox, several teams have run **Heretic** (an automated abliteration tool that orthogonalizes the model's refusal direction out of the weights) against both Qwen3.6 checkpoints. These are a legitimate fit for an *isolated* sandbox, but go in with eyes open:

- **Abliteration is not free.** Removing the refusal direction can introduce subtle coherence/instruction-following drift and occasional artifacts (some cards mention "crisis-substitution" deflections). **Lower KL-divergence vs base = less drift**, so use it as your quality-retention proxy.
- **"Stronger than base" claims are usually commonsense-MCQ benchmarks** (ARC/BoolQ/HellaSwag/PIQA/WinoGrande), **not coding**. Don't take them as evidence of better SWE/agentic performance; validate in the Phase C quality pass on your own tasks.
- It removes a safety layer; the model will answer what the base refuses. You own the outputs. Keep it on the sandbox VLAN, not anything LAN-exposed.

### The variants worth testing (CUDA / GGUF)

| Model | Base | Refusals↓ | KLD vs base | 4-bit options | MTP? | Notes for 16 GB |
|---|---|---|---|---|---|---|
| **Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored** | 27B dense, pure abliteration (no finetune) | low | **0.0282** (lowest) | Q4_K_M, Q3_K_M, Q2_K | ❌ | **Least capability drift** (lowest KLD): closest to stock 27B, just refusals removed. Best "minimal-intervention" 27B. |
| **llmfan46/…35B-A3B-uncensored-heretic-Native-MTP-Preserved** | 35B-A3B MoE | low | low | i1/K-quants incl. Q4_K | ✅ **MTP heads kept** | **The only uncensored MoE that keeps MTP** → fits the speculative-decoding plan. Top MoE pick for this track. |
| **fredrezones55/…35B-A3B-Uncensored-HauhauCS-Aggressive** | 35B-A3B MoE | **0/465** | n/a | `Q4_K_P` ("Perfect" custom quants) | ❌ | Most aggressive de-censor; K_P quants claim +1-2 quant levels of quality at +5-15% size. No MTP. |
| **HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced** | 27B dense, abliteration + K_P imatrix quants | low | n/a | `IQ4_XS`, `Q3_K_P`, `IQ3_M` (up to `Q8_K_P`) | ❌ (dense) | **Keeps the reasoning trace**: reasons out loud, adds a short disclaimer, then answers in full — the softest-touch 27B here. Ships an mmproj (vision). IQ4_XS (~15 GB) ≈ stock-27B offload regime; `Q4_K_P` (~18 GB) exceeds 16 GB — skip. |
| **Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic** | 35B-A3B MoE | 1/25 | low | Q4_K_M (+Q6/Q8) | ❌ | Straightforward MoE abliteration with matching mmproj. |
| **mradermacher/…35B-A3B-uncensored-heretic-i1** | 35B-A3B MoE | n/a | n/a | full i1 imatrix ladder | ❌ | Standard mradermacher i1 quants if you want the widest 4-bit spread. |

*(Skip MLX ports like `unn/qwen3.6-35b-a3b-heretic-4bit`; those are Apple-only.)*

### Three-way conflict to plan around
On llama.cpp, **MTP + vision (`--mmproj`) can't run together yet**, and most Heretic variants drop MTP entirely. So you pick **two of three**: uncensored / MTP / vision. Practical resolutions:
- **Uncensored + MTP** → llmfan46 MoE, drop vision.
- **Uncensored + vision** → any Heretic variant + its mmproj, drop MTP.

### Recommended uncensored picks
- **27B (quality-retention):** Youssofal abliterated (lowest KLD) at Q4_K_M.
- **35B-A3B (speed + MTP):** llmfan46 MTP-Preserved at Q4_K, expert-offload to RAM like the stock MoE.

> Run these through the **same scripts** as the stock models (just add their lines to `configs.sh`). The KLD numbers are the vendors' own; your Phase C quality pass is what actually decides whether abliteration cost you anything.

### Code-specialized finetunes (separate from the abliteration track)

`DavidAU/Qwen3.6-27B-NEO-CODE-Di-IMatrix-MAX-GGUF` is a code/creative **finetune** of the 27B dense, not an abliteration — so **there is no KLD-vs-base drift proxy** to reason about, and the card's "stronger than base" style claims are not coding evidence (see §closing note). Its only justification is your own Phase-C coding pass. Two practical notes for 16 GB: it's an "IMatrix-**MAX**" repo (embeddings/output kept high-precision), so files run *larger* than a stock tag of the same name — `IQ4_XS` is already ~15.4 GB (stock-27B offload regime), `IQ3_M` (~12.9 GB) fits with KV room, and `Q4_K_M` (~16.9 GB) exceeds 16 GB VRAM. Commented-out lines are in `configs.sh` under the "code-specialized finetune" block.

---

## 4. Phased plan

**Phase A: Environment (once).** Build llama.cpp main with CUDA (Section 5). **Pin CUDA 13.1 or 13.3, NOT 13.2** (known bug: low-bit Qwen3.6 quants emit gibberish on 13.2). Confirm `--n-cpu-moe` is present (`llama-bench --help | grep cpu-moe`).

**Phase B: Fit + throughput sweep.** Run `run-bench.sh` (Section 7). It auto-finds what loads vs OOMs and records prefill/generation tok/s plus peak VRAM/RAM for every quant across context depths. The `status=FAIL` rows are your fit map.

**Phase C: Quality pass.** Run `run-quality.sh` with your *own* coding/agentic prompts in `prompts/`. It generates every model's answers to disk for you to read later. This is the only quality signal that matters for your use; public benchmarks won't tell you if a quant quietly degraded on your tasks.

---

## 5. Build + run manually (text-only, no MTP)

Threads: `-t 8` (physical cores; the 7800X3D's 3D V-cache won't help, expert offload is DDR5-bandwidth-bound, not cache-bound). FA on for Ada (sm_89). `-hf` auto-downloads and caches the model on first run.

### Build once

```bash
sudo apt install -y build-essential cmake git libcurl4-openssl-dev   # + CUDA toolkit (pin 13.1/13.3, NOT 13.2)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j
# binaries in build/bin/ ; -DLLAMA_CURL=ON enables -hf auto-download
```

### 27B dense: push everything onto GPU

```bash
# Start at 8K to guarantee load; raise -c if VRAM allows, drop -ngl a few if it OOMs.
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-GGUF:IQ4_XS \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 8192 -t 8 \
  --jinja --chat-template-file ../templates/qwen36-froggeric-v20.jinja \
  --host 0.0.0.0 --port 8080
```

### 35B-A3B MoE: offload experts to RAM

```bash
# --n-cpu-moe 99 parks ALL expert layers in DDR5 RAM; attention stays on GPU.
# Lower the number to pull experts back onto the GPU until VRAM ~fills (= faster).
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL_XL \
  -ngl 99 -fa on --n-cpu-moe 99 \
  -ctk q8_0 -ctv q8_0 \
  -c 16384 -t 8 -b 2048 -ub 512 \
  --jinja --chat-template-file ../templates/qwen36-froggeric-v20.jinja \
  --host 0.0.0.0 --port 8080
```

### Fixed chat template (tool calls)

The `--chat-template-file ../templates/qwen36-froggeric-v20.jinja` flag above loads the **froggeric v20** template, which corrects the stock Qwen3.6 chat template's tool-call handling: the built-in one mangles the `<tool_call>` / `<function=…>` XML nesting and multi-call output, so agentic tool use fails or double-nests. The fixed template also tightens think-block handling and adds error-retry hints. It ships in this repo at `templates/qwen36-froggeric-v20.jinja`; to refetch it:

```bash
curl -L -o templates/qwen36-froggeric-v20.jinja \
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja
```

It requires `--jinja` (already set). It's not needed for `llama-bench` (which does no templating), but use it for **every `llama-server` run** and anywhere you do agentic/tool-calling work. In the automation, each Qwen line in `configs.sh` wires it via `CONFIGS` field 5 (the per-model chat-template path), which `run-quality.sh` passes to the server.

Then open `http://<box-ip>:8080` for the built-in chat UI, or POST to `/v1/chat/completions`. Use **temp 0.6 / top_p 0.95 / top_k 20** for coding, and always send the system prompt `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`

**Vision is off by default** (these commands don't load it), fine for a text/code/agent sandbox. If a task ever needs image/video input, download the repo's `mmproj-*.gguf` and add `--mmproj /path/to/mmproj.gguf`. Start text-only; add vision only when a real task demands it.

### Optimal-flags table (fill in your tuned values)

| Flag | 27B dense | 35B-A3B MoE |
|---|---|---|
| `-ngl` | 99 (drop if OOM) | 99 |
| `--n-cpu-moe` | n/a (dense) | 99 = all on CPU; lower to fill VRAM |
| `-fa` | on | on |
| `-c` | ___ (max stable) | ___ (max stable) |
| `-b` / `-ub` | 2048 / 512 | 2048 / 512 (raise ub if PP-bound) |
| `-t` | 8 | 8 |
| KV quant | q8_0 / q8_0 | q8_0 / q8_0 |

---

## 6. llama-bench: clean throughput numbers

`llama-bench` is the right tool for tok/s: it handles warmup + repetitions and prints structured output. It accepts `-hf` (so no manual downloads), `--n-cpu-moe`, and `-d` to sweep generation speed at different context depths. One manual example each:

```bash
# Dense, GPU-only: pp512 (prefill) and tg128 (generation)
llama-bench -hf unsloth/Qwen3.6-27B-GGUF:IQ4_XS \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128

# MoE, all experts on CPU, generation speed at several context depths in one call
llama-bench -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL_XL \
  -ngl 99 -fa on --n-cpu-moe 99 -ctk q8_0 -ctv q8_0 \
  -p 512 -n 128 -d 0,4096,8192,16384,32768
```

`-d N` prefills N tokens before measuring `tg`, so you see how generation slows as context fills. The next section wraps this in a script so you don't run it by hand for every quant.

---

## 7. Automate the whole sweep

Three scripts (in the `scripts/` bundle alongside this playbook) run everything unattended and leave you a CSV + a folder of model outputs to review later. They share one editable config file.

**`configs.sh`** is the single place you edit: path to your llama.cpp build, the test matrix (`label | hf-repo:quant | dense|moe`), depths to sweep, KV-quant, threads, and the MoE offload number. Add/remove/comment model lines here; all scripts read them.

**`prefetch.sh`** downloads every model in the matrix up front (via llama.cpp's own `-hf` resolver, so the cache matches what the sweep expects) and integrity-checks each. Run it once before the sweep so `run-bench.sh` never stalls on a ~100 GB download mid-run.

**`run-bench.sh`** loops every config × every depth, runs `llama-bench` (auto-downloads via `-hf`, applies `--n-cpu-moe 99` for MoE configs), samples peak VRAM + RAM in the background during each run, and writes one tidy CSV: `label,quant,type,depth,pp_tok_s,tg_tok_s,vram_peak_mib,ram_used_peak_mib,status`. OOM at a deep context just records `FAIL` for that row and continues, so the CSV doubles as your fit map.

**`run-quality.sh`** starts `llama-server` for each config with the **fixed froggeric chat template** (correct tool-call XML), runs every prompt in `prompts/*.txt` through `/v1/chat/completions` at coding sampling settings, saves each response as `quality/<label>/<prompt>.md`, then shuts the server down and moves on. Drop your own real coding/agentic tasks into `prompts/` and read the outputs whenever; that's your Phase C quality pass, done async.

**`templates/qwen36-froggeric-v20.jinja`** is the froggeric v20 fixed template, wired per-model via `CONFIGS` field 5 in `configs.sh` and passed to the server by `run-quality.sh`.

```bash
# one-time
chmod +x scripts/*.sh
sudo apt install -y jq            # used to parse llama-bench JSON
nano scripts/configs.sh           # set LLAMA_DIR, edit the CONFIGS matrix

# download everything first (run once; the slow part)
./scripts/prefetch.sh

# throughput + fit (unattended; grab coffee)
./scripts/run-bench.sh

# quality outputs for later review (also unattended)
./scripts/run-quality.sh
```

**Runtime:** roughly *configs × depths × (load + ~6 bench runs)*. With 5 configs × 5 depths and `REPS=3`, budget ~1-2 hours the first time (downloads dominate); reruns are far faster since GGUFs and the OS page cache are warm. Trim `DEPTHS` or `CONFIGS` to shorten it. Run it under `tmux`/`screen` (or as a systemd unit) so it survives an SSH drop. The result CSV opens directly in any spreadsheet: sort by `tg_tok_s` to rank, filter `status=OK` to see what actually fit.

---

## 8. Results matrix (auto-filled by `run-bench.sh`)

`run-bench.sh` writes these columns to CSV; the rows below are the configs it sweeps. `depth` is the context-fill the speed was measured at.

| label | quant | type | depths swept | notes |
|---|---|---|---|---|
| `27B_IQ4_XS` | IQ4_XS | dense | 0-32K | lead dense (fit) |
| `27B_UD-Q4_K_XL` | UD-Q4_K_XL | dense | 0-32K | quality compare |
| `35B_UD-IQ4_NL_XL` | UD-IQ4_NL_XL | moe | 0-32K | lead MoE (CPU-friendly) |
| `35B_UD-Q4_K_XL` | UD-Q4_K_XL | moe | 0-32K | quality compare |
| `35B_UD-Q3_K_M`† | UD-Q3_K_M | moe | 0-32K | mostly-in-VRAM option |
| `27B_Heretic_Youssofal` | Q4_K_M | dense | 0-32K | uncensored, lowest KLD *(commented out by default)* |
| `27B_Heretic_Youssofal_Q3_K_M` | Q3_K_M | dense | 0-32K | uncensored 3-bit; smaller, fits dense on-GPU *(commented out)* |
| `35B_Heretic_HauhauCS`‡ | Q4_K_P | moe | 0-32K | uncensored MoE; **gated repo — needs `HF_TOKEN`** *(commented out by default)* |
| `27B_HauhauCS_Balanced` | IQ4_XS | dense | 0-32K | uncensored 27B, keeps reasoning trace; ships mmproj *(commented out)* |
| `27B_HauhauCS_Balanced_Q3_K_P` | Q3_K_P | dense | 0-32K | same, ~14 GB for more on-GPU KV room *(commented out)* |
| `27B_NEO_CODE_IQ4_XS`§ | IQ4_XS | dense | 0-32K | code finetune (not abliteration); ~15.4 GB, stock-27B offload regime *(commented out)* |
| `27B_NEO_CODE_IQ3_M`§ | IQ3_M | dense | 0-32K | same finetune, ~12.9 GB fits with KV room *(commented out)* |

†Not a 4-bit quant, but `UD-Q3_K_M` (~16.6 GB) is the one config that gets the **MoE mostly into 16 GB VRAM**, worth keeping as the "max-speed, accept slight quality loss" option, and it directly answers the Q3_K_M-vs-Q4 question.

‡The `35B_Heretic_HauhauCS` repo (`fredrezones55/…HauhauCS-Aggressive`) is **gated/private**: the `-hf` resolver returns HTTP 401 and the model never downloads without a valid `HF_TOKEN` (put one in `.local/secrets.env` — see the Secrets block in `configs.sh`). Separately, `27B_Heretic_Youssofal_Q3_K_L` was removed: `Q3_K_L` is **not** published by Youssofal (§3 shows `Q4_K_M / Q3_K_M / Q2_K`), so it returned "no GGUF files found" and never loaded in the 2026-07-02 runs. Use `Q3_K_M` for on-GPU 3-bit.

§`NEO_CODE` is a **finetune**, not an abliteration — no KLD drift proxy, so it lives in its own `configs.sh` block and is justified only by your Phase-C coding pass (see §3, "Code-specialized finetunes"). Both its tags (`IQ4_XS`, `IQ3_M`) are on DavidAU's HF repo; `Q4_K_M` (~16.9 GB) exceeds 16 GB VRAM and is intentionally omitted.

---

## 9. Known caveats / gotchas

- **CUDA 13.2 = gibberish** on low-bit Qwen3.6. Pin 13.1 or 13.3.
- **The container's 24 GB RAM is the wall** for MoE offload. If you see swap thrash, drop a quant tier or lower `--n-cpu-moe` (more on GPU) only if VRAM allows.
- **Proxmox LXC passthrough:** match the host's NVIDIA driver in the container (`--no-kernel-module`); pass `nvidia-uvm` too (not just `nvidia0`/`nvidiactl`) or CUDA fails even when `nvidia-smi` works; and set the container Memory limit ≥ ~22 GB or the cgroup OOM-killer kills the MoE during expert load.
- **Flash attention** is fine on Ada; if you hit a Gated-DeltaNet quirk, test `-fa off` on the MoE as a fallback (some early reports).
- Set the system prompt (`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`); the model underperforms without it.
- **Tool calls:** use the fixed `templates/qwen36-froggeric-v20.jinja` (froggeric v20) via `--chat-template-file` + `--jinja` on the server. The stock Qwen3.6 template mis-formats `<tool_call>`/`<function=…>` XML, breaking agentic use.
- **Heretic/abliterated variants** (Section 3): use KLD-vs-base as your drift proxy (lower = closer to stock), and don't trust commonsense-MCQ "stronger than base" claims as coding evidence. Keep them on the isolated sandbox VLAN.

---

## 10. Expected outcome (predictions to validate, confidence noted)

- **27B dense** lands ~8-15 tok/s with small offload. *(Medium confidence: scaled from RTX 3090/Mac reports to this bandwidth; the offload penalty is the wildcard.)*
- **35B-A3B MoE** lands **~20-35 tok/s** with expert offload. **It will be the more responsive daily driver.** *(Medium-high confidence.)*
- **27B dense wins on coding quality** per Qwen's own benchmarks (every param fires); the MoE is close and much faster. *(Vendor-reported; the Phase C quality pass on your tasks is the tiebreaker.)*
- **27B offload turns "unusably slow" (<5 tok/s)** mainly as context grows and KV pushes weights off the GPU, likely past 32K once you're forced into heavier offload. The depth sweep pins the exact knee. *(Medium confidence.)*

**Bottom line going in:** daily driver = **35B-A3B at UD-IQ4_NL_XL with expert offload** (or UD-Q3_K_M if you want it mostly in VRAM); keep **27B IQ4_XS** as the high-quality coding model. Let the scripts confirm.
