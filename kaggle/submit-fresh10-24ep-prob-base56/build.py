"""Build an OOF-supported fixed-family-weight Fresh10 submission.

CUDA OOF selects a 56.25% DINOv2-base and 43.75% DINOv2-small raw-probability
blend.  This changes only the final ten-member combine operation.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys


BASE_WEIGHT = 0.5625
SMALL_WEIGHT = 1.0 - BASE_WEIGHT


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
    replacement = '''    # CUDA OOF selected this fixed global family ratio.  It is deliberately
    # global rather than fitted per label, avoiding a high-variance 12-vector.
    if scored.any():
        combined = np.zeros((int(scored.sum()), len(TARGETS)), dtype=np.float32)
        total_w = 0.0
        for member, (net, _slices, _head, _name) in zip(member_preds, models):
            hidden = int(net.encoder.config.hidden_size)
            if hidden == 768:
                weight = 0.5625 / 5.0
            elif hidden == 384:
                weight = 0.4375 / 5.0
            else:
                write_and_exit(f"unexpected Fresh10 encoder width: {hidden}")
            combined += weight * member[scored]
            total_w += weight
        if abs(total_w - 1.0) > 1e-6:
            write_and_exit(f"invalid family blend weight: {total_w}")
        preds[scored] = combined
    log("combined fresh10 by raw probability: base=0.5625, small=0.4375")
'''
    source = source[:start] + replacement + source[end:]
    ast.parse(source)
    script_path.write_text(source, encoding="utf-8")
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(id="sibusisokhumalo11/knee-submit-fresh10-24ep-prob-base56",
                    title="knee-submit-fresh10-24ep-prob-base56")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1])
