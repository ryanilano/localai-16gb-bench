# ===========================================================================
# versions.sh — provenance capture for the benchmark scripts.
# Sourced by run-bench.sh and run-quality.sh. Defines capture_versions(),
# which records the exact NVIDIA driver, CUDA toolkit and llama.cpp build that
# produced a run, so every result set is self-documenting.
#
# Everything is probed dynamically at runtime — nothing here is hardcoded.
# Missing tools degrade to "(unavailable)" rather than aborting the run.
# ===========================================================================

# run_slug — a human-readable, sortable, collision-resistant id for one run:
#   <date>_<time>-<host>, e.g. 2026-07-02_1432-debianbox
# Time-to-the-second + hostname means a later run never clobbers an earlier one.
run_slug() {
  printf '%s-%s' "$(date +%Y-%m-%d_%H%M%S)" "$(hostname -s 2>/dev/null || hostname)"
}

# csv_to_md <csvfile> — render a CSV as a GitHub-flavored markdown table on stdout.
# Assumes simple, unquoted comma-separated values (which the throughput CSV is).
csv_to_md() {
  local csv="$1"
  awk -F',' 'NR==1 {
      printf "|"; for (i=1;i<=NF;i++) printf " %s |", $i; print "";
      printf "|"; for (i=1;i<=NF;i++) printf " --- |"; print "";
      next
    }
    { printf "|"; for (i=1;i<=NF;i++) printf " %s |", $i; print "" }' "$csv"
}

# capture_versions <outfile> [llama_binary]
#   outfile      where to write the provenance stamp (created, never overwritten
#                by design since callers pass a per-run timestamped path).
#   llama_binary the binary actually driving this run (llama-bench / llama-server);
#                its resolved path is recorded. Defaults to $LLAMA_CLI.
# Emits a non-fatal WARNING (stderr + stamp file) if the CUDA toolkit is 13.2,
# the repo's known-bad low-bit-correctness version.
capture_versions() {
  local outfile="$1"
  local bin="${2:-${LLAMA_CLI:-}}"

  # --- llama.cpp binary + version --------------------------------------------
  local resolved_bin llama_ver llama_commit
  resolved_bin=$(readlink -f "$bin" 2>/dev/null || echo "$bin")
  # llama-cli --version prints to stderr; capture both streams. Prefer LLAMA_CLI
  # for the version string, fall back to the run binary if it isn't present.
  if [ -n "${LLAMA_CLI:-}" ] && [ -x "$LLAMA_CLI" ]; then
    llama_ver=$("$LLAMA_CLI" --version 2>&1 | head -20)
  elif [ -n "$resolved_bin" ] && [ -x "$resolved_bin" ]; then
    llama_ver=$("$resolved_bin" --version 2>&1 | head -20)
  else
    llama_ver="(unavailable)"
  fi
  # llama.cpp source git commit, if a checkout is discoverable at LLAMA_DIR.
  if git -C "${LLAMA_DIR:-.}" rev-parse HEAD >/dev/null 2>&1; then
    llama_commit=$(git -C "${LLAMA_DIR:-.}" rev-parse HEAD 2>/dev/null)
  else
    llama_commit="(no git checkout at ${LLAMA_DIR:-?})"
  fi

  # --- NVIDIA driver / GPU / VRAM / power ------------------------------------
  local gpu_line driver_userspace power_line
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line=$(nvidia-smi --query-gpu=driver_version,name,memory.total \
                 --format=csv,noheader 2>/dev/null | head -1)
    power_line=$(nvidia-smi --query-gpu=power.limit,power.max_limit \
                   --format=csv,noheader 2>/dev/null | head -1)
  fi
  [ -n "${gpu_line:-}" ]   || gpu_line="(nvidia-smi unavailable)"
  [ -n "${power_line:-}" ] || power_line="(nvidia-smi unavailable)"
  if [ -r /proc/driver/nvidia/version ]; then
    driver_userspace=$(cat /proc/driver/nvidia/version)
  else
    driver_userspace="(unavailable)"
  fi

  # --- CUDA toolkit ----------------------------------------------------------
  local nvcc_release cuda_link cuda_ver_num
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_release=$(nvcc --version 2>/dev/null | grep -i release | head -1)
  fi
  [ -n "${nvcc_release:-}" ] || nvcc_release="(nvcc unavailable)"
  cuda_link=$(readlink -f /usr/local/cuda 2>/dev/null || echo "(unavailable)")
  # Extract the "13.2"-style version for the known-bad check.
  cuda_ver_num=$(printf '%s' "$nvcc_release" | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')

  # --- write the stamp -------------------------------------------------------
  {
    echo "# Benchmark run provenance"
    echo "timestamp:            $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "hostname:             $(hostname)"
    echo
    echo "## NVIDIA driver / GPU"
    echo "gpu (driver,name,vram): $gpu_line"
    echo "power (limit,max):      $power_line"
    echo "driver userspace:       $driver_userspace"
    echo
    echo "## CUDA toolkit"
    echo "nvcc release:           $nvcc_release"
    echo "/usr/local/cuda ->:     $cuda_link"
    echo
    echo "## llama.cpp"
    echo "resolved binary:        $resolved_bin"
    echo "source git commit:      $llama_commit"
    echo "version output:"
    printf '%s\n' "$llama_ver" | sed 's/^/  /'
  } > "$outfile"

  # --- known-bad CUDA 13.2 warning (non-fatal) -------------------------------
  if [ "${cuda_ver_num:-}" = "13.2" ]; then
    local msg="WARNING: CUDA toolkit 13.2 detected — known to break low-bit quant correctness (see model-benches/qwen36.md). Results may be gibberish."
    echo "$msg" >&2
    echo >> "$outfile"
    echo "$msg" >> "$outfile"
  fi
}
