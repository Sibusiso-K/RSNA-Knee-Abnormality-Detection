"""Build an inference-only OOF/gold diagnostic; no retraining or submission.

**Push with `--accelerator NvidiaTeslaT4` explicit**
(`python -m kaggle kernels push -p OUTPUT_DIR --accelerator NvidiaTeslaT4`),
never a plain `kernels push`. Without it Kaggle can hand back a P100, and
this PyTorch build only supports sm_70+ (P100 is sm_60) - every CUDA op
fails immediately. Hit three times now on three different kernels in this
project (knee-train-pseudo-sel twice, this one once, 2026-09-07) - each
time the CLI's status just says the run "failed" with no obvious cause
until the web UI's Logs tab is checked and shows the accelerator was P100.
"""
import ast
import json
from pathlib import Path
import sys

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
source += '\n' + (config / "evaluate.py").read_text(encoding="utf-8")
ast.parse(source)
metadata = json.loads((repo / "kaggle/submit-fresh10-24ep/kernel-metadata.json").read_text())
metadata.update(id="sibusisokhumalo11/knee-evaluate-fresh24", title="knee-evaluate-fresh24",
                kernel_sources=["sibusisokhumalo11/knee-cache-6slice"])
metadata["dataset_sources"].append("sibusisokhumalo11/knee-labels")
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
(output / "script.py").write_text(source, encoding="utf-8")
(output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + '\n', encoding="utf-8")
print(f"Built inference-only fresh24 OOF/gold diagnostic -> {output}")
