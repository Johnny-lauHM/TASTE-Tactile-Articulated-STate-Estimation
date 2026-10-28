#!/usr/bin/env bash
# Run a command in its own systemd scope with a memory cap, so an OOM kills only this job (not tmux / the shell).
#   scripts/run_isolated.sh <log_file> <command...>      (env: MEM_MAX, default 40G)
set -euo pipefail
log="$1"; shift
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate sharpa
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$(dirname "$log")"
exec systemd-run --user --scope --quiet -p MemoryMax="${MEM_MAX:-40G}" -p MemorySwapMax=0 "$@" > "$log" 2>&1 < /dev/null
