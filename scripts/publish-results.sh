#!/usr/bin/env bash
# ===========================================================================
# publish-results.sh — opt-in: sync your bench_results/ into a dedicated PRIVATE
# GitHub repo and push.
#
# Runs never auto-publish; you call this when you want to sync/back up results.
# Results are copied into a SEPARATE repo directory (a sibling of this project by
# default) so your model outputs live in their own history, apart from the tool.
# On first run it creates the private GitHub repo via the gh CLI if it's missing.
#
#   Results dir:  $RESULTS_DIR env var, else ../../localai-16gb-bench-results
#                 (a sibling of the benchmark project).
#   Repo name:    $RESULTS_REPO env var, else the basename of the results dir.
#   Usage:        ./publish-results.sh [commit message]
#                 RESULTS_DIR=~/data/results RESULTS_REPO=me/results ./publish-results.sh
#
# Requires: git, rsync, and gh (authenticated: `gh auth login`).
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
source ./configs.sh

command -v git   >/dev/null || { echo "Install git"; exit 1; }
command -v rsync >/dev/null || { echo "Install rsync"; exit 1; }
command -v gh    >/dev/null || { echo "Install the GitHub CLI: https://cli.github.com/"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in to GitHub — run: gh auth login"; exit 1; }
# Let git authenticate HTTPS pushes with the gh token — no SSH keys needed
# (idempotent; safe to run every time). This is what makes a fresh box with no
# SSH keys able to push.
gh auth setup-git >/dev/null 2>&1 || true

[ -d "$OUTDIR" ] || { echo "No results yet at $OUTDIR — run ./run-bench.sh first."; exit 1; }

# Resolve to absolute paths NOW, while still in scripts/, since we cd away below
# and $OUTDIR is typically relative (./bench_results).
SRC=$(cd "$OUTDIR" && pwd)
# Absolute path to the benchmark project root (scripts/ lives under it).
PROJECT_ROOT=$(cd .. && pwd)
RESULTS_DIR="${RESULTS_DIR:-$(dirname "$PROJECT_ROOT")/localai-16gb-bench-results}"
REPO="${RESULTS_REPO:-$(basename "$RESULTS_DIR")}"
MSG="${1:-results: sync $(date +%Y-%m-%d_%H%M%S) from $(hostname -s 2>/dev/null || hostname)}"

# Ensure the private GitHub repo exists (create if missing) and get its HTTPS URL.
if [[ "$REPO" == */* ]]; then full="$REPO"; else full="$(gh api user -q .login)/$REPO"; fi
if ! gh repo view "$full" >/dev/null 2>&1; then
  echo "Creating private repo $full ..."
  gh repo create "$full" --private --disable-wiki -d "Benchmark results" >/dev/null
fi
# HTTPS URL (matches gh's default protocol; authenticated via the gh token, so it
# works on a box with no SSH keys).
url="$(gh repo view "$full" --json url -q .url).git"

# Work from a checkout whose tree MATCHES the remote, so publishing only ADDS
# result files and never deletes anything already on the remote (README, prior
# run folders). Cloning/pulling is what makes this safe — a bare `git init` +
# `git add -A` would record every remote-only file as a deletion.
if [ ! -d "$RESULTS_DIR/.git" ]; then
  echo "Cloning $full -> $RESULTS_DIR ..."
  git clone -q "$url" "$RESULTS_DIR"   # creates the dir; works for empty repos too
fi
cd "$RESULTS_DIR"
git remote set-url origin "$url" 2>/dev/null || git remote add origin "$url"
# Fast-forward the working tree to the latest remote state before adding. This
# self-heals a stale checkout and guarantees add-only publishes.
git fetch -q origin 2>/dev/null || true
git checkout -q main 2>/dev/null || git checkout -q -b main
git merge -q --ff-only origin/main 2>/dev/null || true

# Copy result files in (never delete from the archive: --ignore-existing keeps
# the archive append-only, so a re-run can't clobber an earlier run's outputs).
echo "Syncing $SRC -> $RESULTS_DIR/results ..."
rsync -a --ignore-existing "$SRC/" "$RESULTS_DIR/results/"

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to publish."
  exit 0
fi
git commit -q -m "$MSG"
git push -u origin main
echo "Published -> $(git remote get-url origin)"
