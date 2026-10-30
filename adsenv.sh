#!/usr/bin/env bash
# Run a script in this project with the ADS bindings available.
#
#   ./adsenv.sh --setup      # once: build .venv from ADS's Python, install torch,
#                            # unpack the LTE stimulus
#   ./adsenv.sh amam.py      # then run any script through it
#
# Nothing here is tied to one machine. ADS is found from $ADS_HOME, else
# $HPEESOF_DIR, else the `ads` command on PATH. Python is $ADS_PY, else
# ./.venv, else the interpreter ADS bundles (enough for everything except the
# LSTM scripts, which need torch).
#
# LD_LIBRARY_PATH is set for THIS COMMAND ONLY, never exported: ADS ships
# OpenSSL 3.2, which shadows the system 3.5 and breaks curl and the ADS GUI if
# it leaks. The loader reads it at process launch, so setting it from inside
# Python would be too late.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ADS_HOME:-}" ]]; then
  if [[ -n "${HPEESOF_DIR:-}" ]]; then
    ADS_HOME="$HPEESOF_DIR"
  elif ADS_BIN="$(command -v ads)"; then
    ADS_HOME="$(cd "$(dirname "$(readlink -f "$ADS_BIN")")/.." && pwd)"
  else
    echo "ADS not found: set ADS_HOME=/path/to/ADS, or put its bin/ on PATH" >&2
    exit 1
  fi
fi
ADS_LIB="$ADS_HOME/lib/linux_x86_64"
ADS_PYTHON="$ADS_HOME/tools/python/bin/python3"
[[ -d "$ADS_LIB" ]] || { echo "missing ADS libs: $ADS_LIB" >&2; exit 1; }

if [[ "${1:-}" == "--setup" ]]; then
  # --system-site-packages keeps the keysight bindings, numpy, scipy, pandas
  # and matplotlib that ADS already ships; only torch is added on top.
  "$ADS_PYTHON" -m venv --system-site-packages "$HERE/.venv"
  "$HERE/.venv/bin/python3" -m pip install -r "$HERE/requirements.txt"
  # The LTE stimulus is committed xz-compressed (67 MB -> 11 MB)
  STIM="$HERE/ADS_workspace/data/LTE_DL_7p68MSPS_5M_SR_122.88.txt"
  [[ -f "$STIM" ]] || xz -dk "$STIM.xz"
  exit 0
fi

if [[ -n "${ADS_PY:-}" ]]; then
  VENV_PY="$ADS_PY"
elif [[ -x "$HERE/.venv/bin/python3" ]]; then
  VENV_PY="$HERE/.venv/bin/python3"
else
  VENV_PY="$ADS_PYTHON"
fi
[[ -x "$VENV_PY" ]] || { echo "missing interpreter: $VENV_PY" >&2; exit 1; }

exec env \
  LD_LIBRARY_PATH="$ADS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  HPEESOF_DIR="$ADS_HOME" \
  PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
  MPLBACKEND="Agg" \
  QT_QPA_PLATFORM="offscreen" \
  "$VENV_PY" "$@"
