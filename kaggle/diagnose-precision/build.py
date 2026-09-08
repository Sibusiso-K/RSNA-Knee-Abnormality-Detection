"""Isolate whether the fresh24 mismatch is autocast precision: bf16 (what
these checkpoints were TRAINED and scored under, on TPU) vs fp16 (what a GPU
eval defaults to). small fold 0 ONLY, three precision modes, one run.

Usage: python kaggle/diagnose-precision/build.py OUTPUT_DIRECTORY

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
    source += '\n' + (config / "diagnose.py").read_text(encoding="utf-8")
    ast.parse(source)
    metadata = json.loads((repo / "kaggle/submit-fresh10-24ep/kernel-metadata.json").read_text())
    metadata.update(id="sibusisokhumalo11/knee-diagnose-precision", title="knee-diagnose-precision",
                    kernel_sources=["sibusisokhumalo11/knee-cache-6slice"])
    metadata["dataset_sources"].append("sibusisokhumalo11/knee-labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
    print(f"Built precision diagnostic -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
