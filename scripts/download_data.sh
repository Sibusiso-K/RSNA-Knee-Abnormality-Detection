#!/usr/bin/env bash
set -euo pipefail

COMP=rsna-knee-abnormality-detection
DEST=data

mkdir -p "$DEST"
kaggle competitions download -c "$COMP" -p "$DEST"
unzip -n "$DEST"/*.zip -d "$DEST"
