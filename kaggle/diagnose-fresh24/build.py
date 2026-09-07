"""Diagnostic-only build: same OOF harness as evaluate-fresh24, but the
±0.003 assertion is a LOG LINE, not an abort, and it runs every checkpoint
regardless of earlier mismatches. Not the shipped evaluator - exists only to
characterise the mismatch (magnitude, sign, which folds/families) before
fixing evaluate-fresh24 itself.

Usage: python kaggle/diagnose-fresh24/build.py OUTPUT_DIRECTORY
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

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one build marker: {old!r}")
        source = source.replace(old, new, 1)

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
    metadata.update(id="sibusisokhumalo11/knee-diagnose-fresh24", title="knee-diagnose-fresh24",
                    kernel_sources=["sibusisokhumalo11/knee-cache-6slice"])
    metadata["dataset_sources"].append("sibusisokhumalo11/knee-labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
    print(f"Built fresh24 mismatch diagnostic -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
