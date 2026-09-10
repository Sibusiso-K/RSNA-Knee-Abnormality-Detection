"""Build the controlled probability-blend variant of fresh10.

The ten checkpoint roster and preprocessing are inherited from fresh10.  CUDA
OOF showed 50/50 probability averaging exceeds rank averaging in every one of
the five folds; this builder changes only that final combine operation.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys


def load_builder(path):
    spec = importlib.util.spec_from_file_location("fresh10_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(output):
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    fresh = load_builder(repo / "kaggle/submit-fresh10-24ep/build.py")
    fresh.build(output)
    output = Path(output)
    script_path = output / "script.py"
    source = script_path.read_text(encoding="utf-8")
    start = source.index("    # --- combine members by RANK, per column")
    end = source.index("\n    for i, target in enumerate(TARGETS):", start)
    replacement = '''    # --- combine members by PROBABILITY, per column ---------------------
    # CUDA OOF: 50/50 probability blending beat rank blending in all five
    # held-out folds. Members and grids remain the exact fresh10 roster.
    if scored.any():
        combined = np.zeros((int(scored.sum()), len(TARGETS)), dtype=np.float32)
        total_w = 0.0
        for member, (_net, slices, _head, _name) in zip(member_preds, models):
            w = float(GRID_WEIGHT.get(slices, 1.0))
            combined += w * member[scored]
            total_w += w
        preds[scored] = combined / max(total_w, 1e-6)
    log(f"combined {len(member_preds)} member(s) by per-column probability mean "
        f"over {int(scored.sum())} scored studies")
'''
    source = source[:start] + replacement + source[end:]
    ast.parse(source)
    script_path.write_text(source, encoding="utf-8")
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(id="sibusisokhumalo11/knee-submit-fresh10-24ep-prob",
                    title="knee-submit-fresh10-24ep-prob")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Built fresh10 probability blend: same ten 24-epoch members")


if __name__ == "__main__":
    build(sys.argv[1])
