"""0.893 TTA submission plus a second independent five-fold small/xattn run.

Usage: python kaggle/submit-fresh15-seedavg-tta/build.py OUTPUT_DIRECTORY
       python kaggle/submit-fresh15-seedavg-tta/build.py OUTPUT_DIRECTORY --rehearse [N_STUDIES]

Chains onto kaggle/submit-fresh10-24ep-prob-tta (same decode, same five-view
TTA, same hard output checks) and changes only the roster and the weights:

    members  S   = deployed small, knee-slot-6slice-24ep-v1, folds 0-4
             E24 = snapshot run's epoch-24 small checkpoint,
                   knee-slot-6slice-24ep-snap-v1/*_snap_e24.pth, folds 0-4
             B   = deployed base, knee-slot-base-24ep-v1, folds 0-4
    blend    0.25 * mean(S) + 0.25 * mean(E24) + 0.5 * mean(B)
             (per-member weights 0.05 / 0.05 / 0.10)

Evidence: scripts/eval_small_seed_average.py on the saved TTA predictions
(kaggle/evaluate-snap-tta/results/small_seed_average.json): mean-fold OOF
0.861336 vs anchor 0.859991, +0.001345, 95% CI [+0.00066, +0.00269], 5/5 folds
positive. Post hoc: those folds and checkpoints were inspected in docs/21.

Rehearsal mode adds parity for the E24 members (folds 0 and 1) against the
snapshot evaluator's saved predictions (dataset knee-snap-tta-ref), on top of
the existing S/B parity against knee-tta-eval-ref.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys

SNAP_DATASET = "knee-slot-6slice-24ep-snap-v1"
N_PARITY = 96


def load_builder(path):
    spec = importlib.util.spec_from_file_location("tta_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(output, rehearse=False, n_studies=None):
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    tta = load_builder(repo / "kaggle/submit-fresh10-24ep-prob-tta/build.py")
    if rehearse:
        tta.build(output, rehearse=True, n_studies=n_studies or tta.N_REHEARSAL)
    else:
        tta.build(output)
    output = Path(output)
    script_path = output / "script.py"
    source = script_path.read_text(encoding="utf-8")

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one build marker: {old!r} (found {source.count(old)})")
        source = source.replace(old, new, 1)

    # 1. Keep only the epoch-24 snapshot from the snapshot dataset, and never a
    #    snapshot file from anywhere else. Every other file in that dataset
    #    (its best-epoch knee_slot_foldN.pth, e16, e20) also matches the glob.
    replace_once("    checkpoints = sorted(checkpoints)\n", f'''    checkpoints = sorted(checkpoints)

    def _in_snap(p):
        return "{SNAP_DATASET}" in p.replace("\\\\", "/").split("/")

    _dropped = [p for p in checkpoints
                if (_in_snap(p) and not p.endswith("_snap_e24.pth"))
                or (not _in_snap(p) and "_snap_" in os.path.basename(p))]
    checkpoints = [p for p in checkpoints if p not in _dropped]
    log(f"roster filter: kept {{len(checkpoints)}}, dropped {{len(_dropped)}} non-roster files")
''')

    # 2. Roster is (tag, fold) for three five-fold families.
    replace_once(
        "    expected_roster = {(hidden, fold) for hidden in (384, 768) for fold in range(5)}",
        '    expected_roster = {(tag, fold) for tag in ("S", "E24", "B") for fold in range(5)}\n'
        '    MEMBER_WEIGHT = {"S": 0.25 / 5, "E24": 0.25 / 5, "B": 0.5 / 5}')
    replace_once(
        '    if len(checkpoints) != 10:\n'
        '        write_and_exit(f"expected exactly ten checkpoints, found {len(checkpoints)}")',
        '    if len(checkpoints) != 15:\n'
        '        write_and_exit(f"expected exactly fifteen checkpoints, found {len(checkpoints)}: {checkpoints}")')
    replace_once(
        '        key = (hidden, blob.get("fold"))\n',
        '        _tag = "E24" if path.endswith("_snap_e24.pth") else ("B" if hidden == 768 else "S")\n'
        '        if _tag == "E24" and (hidden != 384 or blob.get("snapshot_epoch") != 24\n'
        '                              or not _in_snap(path)):\n'
        '            write_and_exit(f"invalid E24 member: {path} hidden={hidden} "\n'
        '                           f"snapshot_epoch={blob.get(\'snapshot_epoch\')}")\n'
        '        if _tag != "E24" and _in_snap(path):\n'
        '            write_and_exit(f"snapshot-dataset file outside the E24 roster: {path}")\n'
        '        key = (_tag, blob.get("fold"))\n')
    replace_once("        member_keys.append((hidden, blob.get('fold')))",
                 "        member_keys.append((hidden, blob.get('fold'), _tag))")
    replace_once("    if skipped or roster != expected_roster or len(models) != 10:",
                 "    if skipped or roster != expected_roster or len(models) != 15:")

    # 3. Weighted probability mean by family, not by slice grid.
    replace_once(
        "        for member, (_net, slices, _head, _name) in zip(member_preds, models):\n"
        "            w = float(GRID_WEIGHT.get(slices, 1.0))\n",
        "        for member, _mk in zip(member_preds, member_keys):\n"
        "            w = float(MEMBER_WEIGHT[_mk[2]])\n")
    replace_once('    log(f"combined {len(member_preds)} member(s) by per-column probability mean "',
                 '    log(f"family weights {MEMBER_WEIGHT} (sum over members = '
                 '{sum(MEMBER_WEIGHT[k[2]] for k in member_keys):.3f})")\n'
                 '    log(f"combined {len(member_preds)} member(s) by per-column probability mean "')

    # 4. Output checks expect fifteen members.
    replace_once('    if len(models) != 10:\n'
                 '        _problems.append(f"{len(models)} members loaded, expected 10")',
                 '    if len(models) != 15:\n'
                 '        _problems.append(f"{len(models)} members loaded, expected 15")')

    if rehearse:
        replace_once(
            'REF = _np.load(os.path.join(_ref_dir, "tta_reference.npz"), allow_pickle=True)\n',
            'REF = _np.load(os.path.join(_ref_dir, "tta_reference.npz"), allow_pickle=True)\n'
            '_snap_dir = find_dir("snap_tta_reference.npz")\n'
            'if _snap_dir is None:\n'
            '    raise SystemExit("snapshot reference dataset (knee-snap-tta-ref) not attached")\n'
            'SNAPREF = _np.load(os.path.join(_snap_dir, "snap_tta_reference.npz"), allow_pickle=True)\n'
            'for _f in (0, 1):\n'
            '    assert (SNAPREF[f"ids_fold{_f}"] == REF[f"ids_fold{_f}"]).all(), "reference id order differs"\n')
        a = source.index('        _fam = {384: "small", 768: "base"}\n')
        b = source.index('        _stats["parity_vs_evaluator"] = _parity\n')
        source = source[:a] + f'''        _fam = {{"S": "small", "B": "base"}}
        _parity = {{}}
        for i, (hidden, fold, tag) in enumerate(member_keys):
            if fold not in (0, 1):
                continue
            lo = fold * {N_PARITY}
            sel = slice(lo, lo + {N_PARITY})
            if not scored[sel].all():
                _problems.append(f"parity studies unscored ({{tag}} fold {{fold}})")
                continue
            if tag == "E24":
                pairs = (("tta", member_preds[i], SNAPREF[f"E24_tta_prob_fold{{fold}}"][:{N_PARITY}]),)
            else:
                fam = _fam[tag]
                pairs = tuple((kind, mine, REF[f"{{fam}}_{{kind}}_prob_fold{{fold}}"][:{N_PARITY}])
                              for kind, mine in (("tta", member_preds[i]), ("identity", ident_preds[i])))
            for kind, mine, ref in pairs:
                got = mine[sel]
                _parity[f"{{tag}}_fold{{fold}}_{{kind}}"] = {{
                    "max_abs_diff": float(np.abs(got - ref).max()),
                    "mean_abs_diff": float(np.abs(got - ref).mean())}}
''' + source[b:]
        replace_once('        if len(_parity) != 8:\n'
                     '            _problems.append(f"expected 8 parity comparisons, ran {len(_parity)}")',
                     '        if len(_parity) != 10:\n'
                     '            _problems.append(f"expected 10 parity comparisons, ran {len(_parity)}")')

    ast.parse(source)
    script_path.write_text(source, encoding="utf-8")

    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources = list(metadata["dataset_sources"])
    sources.append(f"sibusisokhumalo11/{SNAP_DATASET}")
    if rehearse:
        sources.append("sibusisokhumalo11/knee-snap-tta-ref")
        metadata.update(id="sibusisokhumalo11/knee-rehearse-fresh15-seedavg-tta",
                        title="knee-rehearse-fresh15-seedavg-tta")
    else:
        metadata.update(id="sibusisokhumalo11/knee-submit-fresh15-seedavg-tta",
                        title="knee-submit-fresh15-seedavg-tta")
    metadata["dataset_sources"] = sources
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built fresh15 seed-average TTA ({'REHEARSAL' if rehearse else 'submission'}) -> {output}")


if __name__ == "__main__":
    args = sys.argv[1:]
    rehearse = "--rehearse" in args
    rest = [a for a in args if a != "--rehearse"]
    build(rest[0], rehearse=rehearse, n_studies=int(rest[1]) if len(rest) > 1 else None)
