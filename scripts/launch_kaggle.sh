#!/usr/bin/env bash
# Launch a Kaggle kernel from this repo, with the two settings that are easy to
# get wrong and expensive to get wrong.
#
# 1. --accelerator NvidiaTeslaT4 is MANDATORY and is why this script exists.
#    The flag is case-sensitive, invalid values are accepted silently, and
#    omitting it gives a P100 — which Kaggle's PyTorch (sm_70+) cannot run at
#    all, so every CUDA op dies with cudaErrorNoKernelImageForDevice. Launching
#    without it burns a GPU session slot on a run that cannot succeed. That has
#    now happened twice; hence this script rather than a documented incantation.
#
# 2. The kernels load src/ from the knee-src DATASET, not from git. Editing a
#    file here changes nothing on Kaggle until you re-publish the dataset:
#        ./scripts/launch_kaggle.sh --push-src
#
# Usage:
#   ./scripts/launch_kaggle.sh --push-src            # publish src/ to knee-src
#   ./scripts/launch_kaggle.sh train-crop            # launch a configured kernel
#   ./scripts/launch_kaggle.sh llm-s0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${KNEE_STAGE_DIR:-/c/Temp}"   # short path: the CLI mangles long ones
OWNER="sibusisokhumalo11"
KAGGLE="python -m kaggle"            # `kaggle` is not on PATH in Git Bash

push_src() {
  local dir="$STAGE/kg-src"
  rm -rf "$dir"; mkdir -p "$dir"
  cp -r "$REPO/src/"* "$dir/"
  find "$dir" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  # Written without a BOM on purpose: PowerShell's `Out-File -Encoding utf8`
  # adds one and the Kaggle CLI's JSON parser fails on it with a bare
  # "Expecting value: line 1 column 1".
  printf '{\n  "title": "knee-src",\n  "id": "%s/knee-src",\n  "licenses": [{"name": "other"}]\n}\n' \
    "$OWNER" > "$dir/dataset-metadata.json"
  $KAGGLE datasets version -p "$dir" -m "${1:-update src}" -r zip

  # WAIT for the new version to actually exist before returning.
  #
  # `datasets version` is ASYNCHRONOUS: it prints "version is being created"
  # and returns immediately. A kernel launched in that window mounts the
  # PREVIOUS version, so the code that just changed is silently not the code
  # that runs. That cost a TPU run: an xattn head experiment failed with
  # "SlotNet.__init__() got an unexpected keyword argument 'head'" because the
  # kernel had the old src. Every earlier run survived only because something
  # else happened between the push and the launch.
  echo "waiting for knee-src to become ready..."
  for _ in $(seq 1 60); do
    if $KAGGLE datasets status "$OWNER/knee-src" 2>&1 | grep -qi '^ready'; then
      echo "knee-src ready"
      return 0
    fi
    sleep 10
  done
  echo "knee-src did not become ready in 10 min; refusing to launch stale code" >&2
  return 1
}

launch() {
  local name="$1" dir="$STAGE/kg-$1"
  [ -d "$REPO/kaggle/$name" ] || { echo "no config: kaggle/$name" >&2; exit 1; }
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$REPO/kaggle/$name/kernel-metadata.json" "$dir/"
  bash "$REPO/kaggle/$name/build.sh" "$REPO" "$dir"

  # CPU kernels must NOT be given --accelerator. Passing it attaches a GPU and
  # bills the 30 h/week quota for work that never touches CUDA -- which is the
  # entire reason the preprocessing kernels exist. The metadata is the single
  # source of truth for this, so read it rather than passing a second flag that
  # could disagree with it.
  if grep -q '"enable_tpu"[[:space:]]*:[[:space:]]*true' "$dir/kernel-metadata.json"; then
    # TPU draws on a SEPARATE 20 h/week quota from the GPU's 30 h. Passing
    # --accelerator here would override the metadata and silently request a
    # GPU instead, which is both the wrong hardware and the exhausted budget.
    echo "launching $name on TPU (separate 20 h/week quota)"
    $KAGGLE kernels push -p "$dir"
  elif grep -q '"enable_gpu"[[:space:]]*:[[:space:]]*false' "$dir/kernel-metadata.json"; then
    echo "launching $name on CPU (no accelerator quota consumed)"
    $KAGGLE kernels push -p "$dir"
  else
    $KAGGLE kernels push -p "$dir" --accelerator NvidiaTeslaT4
  fi
}

case "${1:-}" in
  --push-src) shift; push_src "${1:-update src}" ;;
  "")         echo "usage: $0 [--push-src | <kernel-name>]" >&2; exit 1 ;;
  *)          launch "$1" ;;
esac
