#!/usr/bin/env bash
# Pull ONLY the small CSV files from the competition.
#
# Do NOT run a bare `kaggle competitions download -c rsna-knee-abnormality-detection`:
# the full dataset is 569.76 GB / 819,640 files. The DICOMs stay on Kaggle, where they
# are pre-mounted in notebooks at /kaggle/input/. See docs/03-data-guide.md.

set -euo pipefail

COMP=rsna-knee-abnormality-detection
DEST=data

mkdir -p "$DEST"

for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
    echo "==> $f"
    kaggle competitions download -c "$COMP" -f "$f" -p "$DEST"
done

# Kaggle serves larger files zipped; unpack anything that arrived that way.
shopt -s nullglob
for z in "$DEST"/*.zip; do
    unzip -o "$z" -d "$DEST"
    rm "$z"
done

echo
echo "Done. Contents of $DEST:"
ls -lh "$DEST"
