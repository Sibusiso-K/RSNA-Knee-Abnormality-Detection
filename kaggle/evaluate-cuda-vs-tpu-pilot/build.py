"""CUDA evaluator for docs/23: does the CUDA-trained small/xattn pilot beat
its matched TPU-trained control, under the exact production five-view TTA?

Folds 0 and 1 only (the pilot's scope). Both checkpoints share verified
bit-identical initial weights (docs/23) and the same recipe - labels, cache,
batch, optimizer, augmentation, 24-epoch schedule, per-fold seed - so the
only deliberate difference is the training backend (CUDA vs TPU/XLA). Two
comparisons, both paired, same-checkpoint-pair:
  1. Standalone: TTA(TPU-trained) vs TTA(CUDA-trained), the small family alone.
  2. Ensemble: 0.5*TTA(TPU)+0.5*TTA(deployed base) vs
     0.5*TTA(CUDA)+0.5*TTA(deployed base) - does swapping the small member
     for its CUDA-trained twin change the DEPLOYED 0.893 blend. A win in
     comparison 1 alone is not sufficient; comparison 2 is the one that
     could justify a submission.

Inference only - no training, no submission.

Usage: python kaggle/evaluate-cuda-vs-tpu-pilot/build.py OUTPUT_DIRECTORY

**Push with `--accelerator NvidiaTeslaT4` explicit**, never a plain
`kernels push` - see kaggle/evaluate-fresh24/build.py for why.
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
    source += '\n' + (config / "compare.py").read_text(encoding="utf-8")
    ast.parse(source)
    metadata = {
        "id": "sibusisokhumalo11/knee-evaluate-cuda-vs-tpu-pilot",
        "title": "knee-evaluate-cuda-vs-tpu-pilot",
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
            "sibusisokhumalo11/knee-slot-base-24ep-v1",
            "sibusisokhumalo11/knee-ema-pilot-v1",
            "sibusisokhumalo11/knee-cuda-vs-tpu-pilot-v1",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": ["sibusisokhumalo11/knee-cache-6slice"],
        "model_sources": [
            "metaresearch/dinov2/pyTorch/small/1",
            "metaresearch/dinov2/pyTorch/base/1",
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
    print(f"Built CUDA-vs-TPU pilot evaluator -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
