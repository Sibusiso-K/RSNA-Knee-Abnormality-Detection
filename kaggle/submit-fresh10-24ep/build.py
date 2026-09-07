"""Build the controlled small/base 24-epoch ensemble, with a strict roster.

Usage: python kaggle/submit-fresh10-24ep/build.py OUTPUT_DIRECTORY
Reuse the established mixed submission inference and preprocessing unchanged.
Only the roster changes: five small + five base, all trained for 24 epochs.
"""
import ast
import json
from pathlib import Path
import sys


def build(output):
    config = Path(__file__).resolve().parent
    repo = config.parents[1]
    source = (repo / "notebooks/kaggle_08_submit_mixed.py").read_text(encoding="utf-8")

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one build marker: {old!r}")
        source = source.replace(old, new, 1)

    # Infrastructure errors must fail the notebook, not produce a plausible
    # constant submission. Individual undecodable studies retain the shared
    # pipeline's documented fallback behavior.
    replace_once('raise SystemExit(0)', 'raise SystemExit(1)')
    replace_once('    models, skipped = [], []', '''    models, skipped = [], []
    roster = set()
    expected_roster = {(hidden, fold) for hidden in (384, 768) for fold in range(5)}
    if device.type != "cuda":
        write_and_exit("fresh10 requires a GPU")
    if len(checkpoints) != 10:
        write_and_exit(f"expected exactly ten checkpoints, found {len(checkpoints)}")''')
    replace_once('        dinov2 = find_dinov2(hidden)', '''        key = (hidden, blob.get("fold"))
        if blob.get("epochs") != 24 or slices != 6 or key not in expected_roster or key in roster:
            write_and_exit(f"invalid fresh24 member: {path}, width/fold={key}, epochs={blob.get('epochs')}, slices={slices}")
        roster.add(key)
        log(f"fresh24 member: {path}, width/fold={key}, best_epoch={blob.get('epoch')}, CV={blob.get('score')}")
        dinov2 = find_dinov2(hidden)''')
    replace_once('    TAKES = tuple(sorted({s for _net, s, _h, _n in models}))', '''    if skipped or roster != expected_roster or len(models) != 10:
        write_and_exit(f"incomplete fresh24 ensemble: roster={sorted(roster)}, skipped={skipped}")
    TAKES = tuple(sorted({s for _net, s, _h, _n in models}))''')
    # Retain raw member outputs for diagnostics; ranks are still calculated
    # across the complete scored test set, never independently per batch.
    replace_once('    # --- combine members by RANK, per column ----------------------------', '''    np.savez_compressed("member_predictions.npz", predictions=np.stack(member_preds),
                        scored=scored, study_ids=test[ID].to_numpy(dtype=str))
    # --- combine members by RANK, per column ----------------------------''')
    ast.parse(source)
    metadata = json.loads((config / "kernel-metadata.json").read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "script.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built fresh10: 5 small + 5 base; 24 epochs; 6 slices; uniform rank mean -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
