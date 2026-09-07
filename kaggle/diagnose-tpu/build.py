"""Same isolation as diagnose-precision, but on the TPU/XLA backend these
checkpoints actually trained under - not just a different autocast dtype on
GPU. Precision alone (fp32/fp16/bf16, all on GPU) did NOT reproduce the
recorded score; this checks whether the backend itself is the variable.

Usage: python kaggle/diagnose-tpu/build.py OUTPUT_DIRECTORY
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
    # Deliberately NOT forcing USE_XLA off here - let it auto-detect the TPU.
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
    metadata.update(id="sibusisokhumalo11/knee-diagnose-tpu", title="knee-diagnose-tpu",
                    kernel_sources=["sibusisokhumalo11/knee-cache-6slice"],
                    enable_gpu=False, enable_tpu=True)
    metadata["dataset_sources"].append("sibusisokhumalo11/knee-labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
    print(f"Built TPU-backend diagnostic -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
