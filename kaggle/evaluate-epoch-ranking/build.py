"""Build the CUDA epoch-ranking evaluator for the epoch-snapshot experiment
(docs/15-claude-score-improvement-strategy.md, experiment 1). Inference
only - no retraining, no submission.

Reads every checkpoint train-6slice-24ep-snap produced per fold: the
existing best-on-TPU file (knee_slot_fold{N}.pth) plus the snapshot epochs
(knee_slot_fold{N}_snap_e{16,20,24}.pth), scores each on CUDA against that
fold's held-out set, and reports whether CUDA agrees with TPU about which
epoch is best.

Usage: python kaggle/evaluate-epoch-ranking/build.py OUTPUT_DIRECTORY

**Push with `--accelerator NvidiaTeslaT4` explicit**, never a plain
`kernels push` - see kaggle/evaluate-fresh24/build.py for why (P100
incompatibility, hit three times in this project already).
"""
import ast
import json
from pathlib import Path
import sys


def build(output):
    config = Path(__file__).resolve().parent
    repo = config.parents[1]
    source = (repo / "notebooks/kaggle_06_train_slots.py").read_text(encoding="utf-8")
    source = source.split("\ndef find_dinov2():", 1)[0]
    source = source.replace('os.environ.get("USE_XLA", "auto")', '"0"')
    source = source.replace('os.environ.get("HEAD", "slot")', '"xattn"')

    start = source.index('def shard_paths(pattern):')
    end = source.index('\n\nindex_paths =', start)
    source = source[:start] + '''def shard_paths(pattern):
    import fnmatch
    hits = {}
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                path = os.path.join(root, name)
                if name in hits:
                    raise RuntimeError(f"Duplicate cache shard {name}: {hits[name]}, {path}")
                hits[name] = path
    return [hits[name] for name in sorted(hits)]
''' + source[end:]
    source += '\n' + (config / "ranking.py").read_text(encoding="utf-8")
    ast.parse(source)
    metadata = {
        "id": "sibusisokhumalo11/knee-evaluate-epoch-ranking",
        "title": "knee-evaluate-epoch-ranking",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "sibusisokhumalo11/knee-src",
            "sibusisokhumalo11/knee-labels",
            "sibusisokhumalo11/knee-slot-6slice-24ep-snap-v1",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": ["sibusisokhumalo11/knee-cache-6slice"],
        "model_sources": ["metaresearch/dinov2/pyTorch/small/1"],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
    print(f"Built epoch-ranking evaluator -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
