# Install & Run: 16 GB NVIDIA Local-LLM Benchmark Runbook

Do these in order. Each step has a command, what to expect, and how to know it worked.
The model-neutral rationale lives in `benchmark-methodology.md`; the Qwen3.6 worked example
(used throughout this runbook to make things concrete) is `model-benches/qwen36.md`.

**Files you have:** `benchmark-methodology.md` (the "why") · `model-benches/` (per-model writeups) ·
`templates/` (optional per-model chat templates) · `scripts/` (`configs.sh`, `prefetch.sh`,
`run-bench.sh`, `run-quality.sh`).

---

## Step 0: Pre-flight (5 min)

- [ ] **Pin a known-good CUDA / llama.cpp build and sanity-check output.** Low-bit quants can be runtime-sensitive: a bad pairing makes them emit gibberish even though they load. (Example: Qwen3.6 4-bit breaks on CUDA 13.2; 13.1/13.3 are fine, see `model-benches/qwen36.md` §9.)
- [ ] Free disk for your GGUFs (4-bit GGUFs run ~14-22 GB each; budget for every model you register).
- [ ] **LXC GPU passthrough working** (if containerized); see the LXC notes block below. `nvidia-smi` must run _inside the container_ and CUDA compute must actually init.
- [ ] **If benchmarking a MoE model: RAM limit high enough for expert offload.** MoE parks idle experts in system RAM via `--n-cpu-moe`; too low a cap and the OOM-killer kills it mid-load. (The Qwen 35B-A3B example needs ~22 GB; size this to your largest MoE.) Dense models won't hit this.

### LXC GPU passthrough: make sure these are true

Running the _inference server_ in an LXC is fine (llama.cpp serves tokens, it isn't executing untrusted code; that's your agent-execution sandboxes, which still want the VM boundary). But LXC passthrough fails differently than VM/VFIO:

1. **Driver match.** Install the **same** NVIDIA driver version in the container as on the Proxmox host, with `--no-kernel-module` (kernel module stays on the host). A mismatch makes `nvidia-smi` fail in the container.
2. **Pass the right device nodes**, including `nvidia-uvm`. In `/etc/pve/lxc/<id>.conf` you typically need entries like:
   ```
   lxc.cgroup2.devices.allow: c 195:* rwm        # nvidia0, nvidiactl
   lxc.cgroup2.devices.allow: c 235:* rwm        # nvidia-uvm (major may differ; check `ls -l /dev/nvidia*`)
   lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
   lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
   lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
   lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
   ```
   **The `nvidia-uvm` trap:** `nvidia-smi` can work while CUDA still fails, because `nvidia-smi` doesn't need uvm but llama.cpp does. If CUDA init errors but `nvidia-smi` is fine, this is why. On the host, ensure uvm is loaded (`nvidia-modprobe -u -c=0`) so the device node exists.
3. **Verify both layers in the container:**
   ```bash
   nvidia-smi                                   # lists the 4070 Ti Super
   ./build/bin/llama-bench -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 -n 8   # tiny CUDA smoke test
   ```
   The second line proves CUDA compute works, not just device visibility.
4. **`free -m` shows the container limit** (lxcfs), so the bench CSV's `ram_used_peak_mib` is already the number that matters; watch it against your 22 GB cap on MoE rows.

---

## Step 1: Build llama.cpp (15-30 min, once)

```bash
sudo apt update && sudo apt install -y build-essential cmake git libcurl4-openssl-dev jq
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DLLAMA_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

- [ ] `./build/bin/llama-server --version` and `./build/bin/llama-bench --version` both run.
- [ ] `./build/bin/llama-bench --help | grep cpu-moe` shows `--n-cpu-moe` (confirms MoE offload support).

---

## Step 2: Drop in the scripts (2 min)

```bash
# from the localai-16gb-bench directory (git clone https://github.com/ryanilano/localai-16gb-bench.git):
cd scripts
chmod +x prefetch.sh run-bench.sh run-quality.sh
```

- [ ] **Chat templates are optional and per-model.** Most models run fine on their built-in template,
      so skip this. Only supply a template (in `../templates/`, referenced by `CONFIGS` field 5) when a
      model's stock template mangles tool-call XML. The bundled `templates/qwen36-froggeric-v20.jinja`
      is the example used by the Qwen3.6 config lines. If you need to refetch it:
  ```bash
  curl -L -o ../templates/qwen36-froggeric-v20.jinja \
    https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja
  ```

---

## Step 3: Point the config at your build (2 min)

Edit `configs.sh`:

- [x] Set `LLAMA_DIR` to your llama.cpp checkout (the folder containing `build/bin/`).
- [x] (Optional) Set `LLAMA_CACHE` to a big disk if you don't want GGUFs in `~/.cache`.
- [x] **Review the `CONFIGS` matrix.** Each line is 5 pipe-delimited fields (fields 4-5 optional):
      `label|hf_repo:quant|type(dense|moe)|system_prompt|chat_template_path`. `configs.sh` **ships with
      the Qwen3.6 16 GB sweep already active** (9 configs), so you can reproduce the worked example
      as-is — no edits needed. To go model-neutral, re-comment those lines; to add your own, append
      lines (see `benchmark-methodology.md`, "How to add a model"). The uncensored 27B lines are active
      too — the isolated-sandbox caveat still applies (Step 9). Only the **gated `35B_Heretic_HauhauCS`**
      stays commented: it needs an `HF_TOKEN` (Step 9).
- [ ] (Optional) Trim `DEPTHS` to `(0 8192 32768)` for a faster first pass.

Sanity check:

```bash
bash -n run-bench.sh && bash -n run-quality.sh && echo "scripts OK"
```

---

## Step 4: Pre-download all models (1-2 hrs, unattended)

Pull every model up front so the sweep never stalls mid-run on a download.

```bash
# optional, avoids HF rate limits on big pulls:
# export HF_TOKEN=hf_xxxxx
tmux new -s fetch
./prefetch.sh
# detach: Ctrl-b then d
```

**What happens:** loops `CONFIGS`, downloads each GGUF (all split parts) via llama.cpp's own
`-hf` resolver into the same cache the bench/server use, and does a 1-token CPU pass to verify
each file isn't corrupt. No GPU needed; mmap keeps RAM low. Already-cached models are skipped.

- [ ] Ends with "All N models cached." If any line FAILs, fix that quant tag in `configs.sh`
      (or check disk/network in `bench_results/prefetch_<label>.log`) and re-run; cached ones skip.
- [ ] Budget download size for every model you registered (~14-22 GB per 4-bit GGUF; the Qwen3.6 example block is ~100 GB total). This is the slow part; everything after is fast.

> Why this and not `huggingface-cli`? Using llama.cpp's resolver guarantees the cache layout
> matches what `run-bench.sh`/`run-quality.sh` look for, honoring whatever `LLAMA_CACHE`/`HF_HOME`
> you set, so there's no mismatch. (`huggingface-cli download <repo> --include "*<quant>*"` also works if you
> leave `LLAMA_CACHE` unset so both default to `~/.cache/huggingface`.)

---

## Step 5: Throughput + fit sweep (30-60 min, unattended)

```bash
tmux new -s bench          # so it survives an SSH drop
./run-bench.sh
# detach: Ctrl-b then d   |   reattach: tmux attach -t bench
```

**What happens:** for every quant × depth it loads from cache (Step 4 already downloaded them),
runs `llama-bench`, samples peak VRAM/RAM, and appends a row to `bench_results/throughput_<stamp>.csv`.
OOM at a depth just writes `status=FAIL` and continues.

- [ ] Script finishes and prints a results table.
- [ ] Open the CSV in a spreadsheet. **Decisions to make from it:**
  - Sort by `tg_tok_s` (desc) → your speed ranking.
  - Filter `status=FAIL` → the fit ceiling (which quant dies at which context depth).
  - Check `ram_used_peak_mib` on MoE rows → if it's near your container cap (~22000), you're at the RAM wall. A MoE row that `FAIL`s while `vram_peak_mib` looks fine is almost always the **cgroup OOM-killer**: raise the container Memory limit or drop to a smaller MoE quant. _(Playbook §0, §9; LXC notes in Step 0)_

---

## Step 6: Quality pass (30-60 min, unattended)

First, **replace the seed prompts with your real tasks**. This is the step that actually decides quality:

- [ ] Put 3-8 of your own coding/agentic prompts in `scripts/prompts/` as `*.txt`
      (one task per file; the first run seeds 3 examples you can overwrite).

```bash
./run-quality.sh
```

**What happens:** for each quant it starts `llama-server` (with the fixed tool-call template),
answers every prompt, and saves `bench_results/quality/<label>/<prompt>.md`, then moves on.

- [ ] Read the `.md` outputs side by side. Look for: correct code, coherent reasoning,
      clean tool-call XML, no degradation vs the larger quants.

---

## Step 7: Decide the daily driver (15 min)

Combine Step 5 (speed/fit) + Step 6 (quality) and pick the model that's fastest while still passing
your quality bar at your target context depth. As an illustration, the Qwen3.6 worked example landed
on _(see model-benches/qwen36.md §10)_:

- **Daily driver:** `35B_UD-IQ4_NL_XL` (MoE, expert-offload): fastest, most responsive.
- **Coding/quality model:** `27B_IQ4_XS` (fits on GPU).
- **Max-speed MoE fallback:** `35B_UD-Q3_K_M` (mostly in VRAM, slight quality loss).
- [ ] Pick the winner. If two are close, prefer the one with more context headroom (lower VRAM at your max depth).

---

## Step 8: Tune the chosen MoE config (optional, 20 min)

If you picked a MoE model, claw back speed by moving experts onto the GPU until VRAM ~fills:

- [ ] In `configs.sh` lower `NCMOE_ALL` from `99` in steps (e.g. 99 → 32 → 24 → 20).
- [ ] Re-run just that one config (comment out the others) and watch `vram_peak_mib`.
- [ ] Stop at the lowest number that still loads at your target context without OOM; that's your fastest stable setting.

---

## Step 9: (Optional) Uncensored / Heretic track

The uncensored 27B lines **ship active** and run in the default sweep _(rationale in
model-benches/qwen36.md §3)_: the Youssofal 27B (lowest KLD) and **`27B_HauhauCS_Balanced`**
(softest-touch 27B — keeps the reasoning trace, ships an mmproj). If you don't want them, re-comment
those lines. Quant guidance: `IQ4_XS` for the stock-27B offload regime, or a 3-bit tag for more on-GPU
KV room; the ~18 GB `Q4_K_P` overflows 16 GB — skip it.

- [ ] **Gated 35B (needs a token):** to add `35B_Heretic_HauhauCS`, put an `HF_TOKEN` with repo access
      in `.local/secrets.env` (see the Secrets block in `configs.sh`) and uncomment its line. Without a
      token the `-hf` resolver returns HTTP 401 and it never downloads.
- [ ] **Code finetune (separate, experimental):** `27B_NEO_CODE_*` (DavidAU) is a code-specialized
      _finetune_, not an abliteration — it has no KLD drift proxy, so judge it **only** on the Step 6
      coding outputs. `IQ4_XS` (~15.4 GB) matches the stock-27B offload regime; `IQ3_M` (~12.9 GB) fits
      with KV room; the ~16.9 GB `Q4_K_M` overflows 16 GB.
- [ ] Re-run `run-bench.sh` and `run-quality.sh`; they flow through the same pipeline.
- [ ] Judge on the **Step 6 quality outputs**, not the vendors' refusal/KLD numbers.
- [ ] Keep these models on the **isolated sandbox VLAN**, never LAN-exposed.

---

## Step 10: Lock it in

- [ ] Save the winning `llama-server` command (with tuned `--n-cpu-moe`, `-c`, KV flags,
      and `--chat-template-file ...` if your model needs one) as a systemd unit or a launch script.
- [ ] Roll it into your Ansible/provisioning so the sandbox rebuilds reproducibly.
- [ ] Archive the `bench_results/` CSV + quality outputs as your baseline for the next model generation.

---

### Quick reference: what each artifact is for

| File                       | Purpose                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `INSTALL.md`               | This runbook; do it in order.                                    |
| `benchmark-methodology.md` | The "why": the loop, reading the CSV, dense-vs-MoE, add a model. |
| `model-benches/qwen36.md`  | Worked example: analysis, quant ranking, caveats, results.       |
| `scripts/configs.sh`       | The one file you edit: models, depths, paths, offload.           |
| `scripts/prefetch.sh`      | Download all models up front (run once, before the sweep).       |
| `scripts/run-bench.sh`     | Unattended speed + fit sweep → CSV.                              |
| `scripts/run-quality.sh`   | Unattended quality outputs → markdown per model.                 |
| `templates/*.jinja`        | Optional per-model chat templates (e.g. the Qwen3.6 example).    |
