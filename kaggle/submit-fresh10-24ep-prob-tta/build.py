"""Five-view translation-TTA version of the 0.891 fresh10 probability ensemble.

Usage: python kaggle/submit-fresh10-24ep-prob-tta/build.py OUTPUT_DIRECTORY
       (rehearsal variant: kaggle/rehearse-fresh10-tta/build.py)

Chains onto the probability builder (same ten checkpoints, 50/50 small/base
uniform probability blend) and changes only the per-member inference:

    per member:  logits per (slice group, view) -> mean over groups per view
                 -> mean over the five views -> sigmoid          (src.model.tta)
    ensemble:    uniform mean of member probabilities           (unchanged)

Each study is decoded once (build_study_multi) and its tensors, plus their
translated views, are shared by all ten members.

Unlike the base notebook, failures here are hard: no fallback submission is
written for infrastructure problems, and the finished predictions are checked
(IDs, columns, shape, finiteness, range, spread, decode-failure rate) before
submission.csv exists at all.

Rehearsal mode (rehearse=True) runs the identical inference code over ~1,400
TRAINING studies, writes rehearsal_predictions.csv (never submission.csv, so it
cannot be submitted), times decode and forward separately, and checks parity of
the fold-0/fold-1 members against the evaluator's saved predictions on the
studies each of those checkpoints held out. The ten-model ensemble is NOT
scored on training studies: it includes models trained on them.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys

N_REHEARSAL = 1400
N_PARITY = 96
PARITY_TOL = 5e-3          # fp16-autocast/batching noise; real bugs are >>0.01


def load_builder(path):
    spec = importlib.util.spec_from_file_location("prob_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REHEARSAL_STUDIES = """\
import numpy as _np
_ref_dir = find_dir("tta_reference.npz")
if _ref_dir is None:
    raise SystemExit("evaluator reference dataset (knee-tta-eval-ref) not attached")
REF = _np.load(os.path.join(_ref_dir, "tta_reference.npz"), allow_pickle=True)
_train = pd.read_csv(f"{COMP}/train.csv")
_pinned = [str(s) for f in (0, 1) for s in REF[f"ids_fold{f}"][:@@PAR@@]]
_others = sorted(set(_train["StudyInstanceUID"].astype(str)) - set(_pinned))
_others = list(_np.random.RandomState(0).permutation(_others)[:@@TOT@@ - len(_pinned)])
test = pd.DataFrame({"StudyInstanceUID": _pinned + _others})
test_series = pd.read_csv(f"{COMP}/train_series.csv")
""".replace("@@PAR@@", str(N_PARITY))

EVAL_CHECK_DEF = """\
    REHEARSAL_EVAL_CHECK = @@ON@@
    REHEARSAL_EVAL_CHECK_RESULT = {}

    def eval_style_check(net, xk, m, got):
        \"\"\"The evaluator's predict_views loop (float64 weighted sum, numpy-style
        sigmoid) on the same tensors, independent of tta_probs.\"\"\"
        acc = [None] * len(TTA_VIEWS)
        n_g = xk.shape[2] // GROUP
        for g in range(n_g):
            x0 = xk[:, :, g * GROUP:(g + 1) * GROUP]
            for vi, (tx, ty) in enumerate(TTA_VIEWS):
                with torch.autocast(device_type="cuda", enabled=False):   # evaluator translated outside autocast
                    xv = translate_images(x0, tx, ty)
                lg = net(xv.clone() if xv.is_floating_point() else xv, m).float()
                acc[vi] = lg if acc[vi] is None else acc[vi] + lg
        per_view = torch.stack([a / n_g for a in acc], dim=1).double()
        w = torch.tensor(TTA_WEIGHTS, dtype=torch.float64, device=per_view.device)
        want = torch.sigmoid((per_view * w[None, :, None]).sum(1)).float()
        return {"max_abs_diff_tta_probs_vs_evaluator_formula": float((want - got).abs().max())}

"""

POST_CHECKS = """
    # ---- hard output checks: nothing below may be silently degraded --------
    import json
    _n = len(test)
    _problems = []
    _stats = {"n_studies": _n, "scored": int(scored.sum()), "decode_failures": int(failures),
              "no_usable_slot": int(empty), "members": len(models),
              "member_keys": [list(map(str, k)) for k in member_keys],
              "wall_seconds_total": time.time() - T0,
              "loop_seconds": time.time() - T_LOOP0,
              "decode_seconds": T_DECODE, "forward_seconds": T_FORWARD,
              "decode_s_per_study": T_DECODE / max(_n, 1),
              "forward_s_per_study": T_FORWARD / max(int(scored.sum()), 1),
              "views": [list(v) for v in TTA_VIEWS], "weights": list(TTA_WEIGHTS),
              "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    if len(models) != 10:
        _problems.append(f"{len(models)} members loaded, expected 10")
    if device.type != "cuda":
        _problems.append("not running on CUDA")
    if (failures + empty) > int(0.01 * _n):
        _problems.append(f"{failures} decode failures + {empty} unusable of {_n} studies")
    _cols = list(submission.columns)
    if _cols != [ID] + list(TARGETS):
        _problems.append(f"columns {_cols}")
    if submission.shape != (_n, 1 + len(TARGETS)):
        _problems.append(f"shape {submission.shape}")
    if not submission[ID].astype(str).equals(test[ID].astype(str)):
        _problems.append("IDs differ from the study list / order")
    _vals = np.stack([preds[:, i] for i in range(len(TARGETS))], axis=1).astype(np.float64)
    if not np.isfinite(_vals).all():
        _problems.append("non-finite predictions")
    if _vals.min() < 0 or _vals.max() > 1:
        _problems.append("predictions outside [0, 1]")
    if _n > 1 and float(np.nanstd(_vals, axis=0).mean()) < 1e-4:
        _problems.append("predictions are (nearly) constant")
    _diff = [float(np.abs(member_preds[i][scored] - ident_preds[i][scored]).mean())
             for i in range(len(models))] if scored.any() else []
    _stats["mean_abs_tta_minus_identity_per_member"] = _diff
    if _diff and min(_diff) <= 0.0:
        _problems.append("a member's TTA output equals its identity output - TTA not applied")
    if _diff and max(_diff) > 0.1:
        _problems.append(f"implausibly large TTA shift {max(_diff):.3f}")
    if @@REHEARSE@@:
        _stats["eval_formula_check"] = REHEARSAL_EVAL_CHECK_RESULT
        if not REHEARSAL_EVAL_CHECK_RESULT or REHEARSAL_EVAL_CHECK_RESULT[
                "max_abs_diff_tta_probs_vs_evaluator_formula"] > 1e-4:
            _problems.append(f"tta_probs != evaluator formula: {REHEARSAL_EVAL_CHECK_RESULT}")
        _fam = {384: "small", 768: "base"}
        _parity = {}
        for i, (hidden, fold) in enumerate(member_keys):
            if fold not in (0, 1):
                continue
            lo = fold * @@PAR@@
            sel = slice(lo, lo + @@PAR@@)
            fam = _fam[int(hidden)]
            if not scored[sel].all():
                _problems.append(f"parity studies unscored ({fam} fold {fold})")
                continue
            for kind, mine in (("tta", member_preds[i]), ("identity", ident_preds[i])):
                ref = REF[f"{fam}_{kind}_prob_fold{fold}"][:@@PAR@@]
                got = mine[sel]
                _parity[f"{fam}_fold{fold}_{kind}"] = {
                    "max_abs_diff": float(np.abs(got - ref).max()),
                    "mean_abs_diff": float(np.abs(got - ref).mean())}
        _stats["parity_vs_evaluator"] = _parity
        _stats["parity_tol"] = @@TOL@@
        if len(_parity) != 8:
            _problems.append(f"expected 8 parity comparisons, ran {len(_parity)}")
        for k, v in _parity.items():
            log(f"  parity {k}: max {v['max_abs_diff']:.2e} mean {v['mean_abs_diff']:.2e}")
            if v["max_abs_diff"] > @@TOL@@:
                _problems.append(f"parity {k}: max diff {v['max_abs_diff']:.3e}")
        np.savez_compressed("rehearsal_member_predictions.npz",
                            tta=np.stack(member_preds), identity=np.stack(ident_preds),
                            ids=test[ID].to_numpy(dtype=str))
    np.savez_compressed("member_predictions_tta.npz", tta=np.stack(member_preds),
                        identity=np.stack(ident_preds), scored=scored,
                        study_ids=test[ID].to_numpy(dtype=str))
    _stats["problems"] = _problems
    with open("run_stats.json", "w") as _fh:
        json.dump(_stats, _fh, indent=2)
    log(f"run stats: decode {T_DECODE:.0f}s forward {T_FORWARD:.0f}s total {time.time() - T0:.0f}s "
        f"({_stats['decode_s_per_study']:.3f} + {_stats['forward_s_per_study']:.3f} s/study)")
    if _problems:
        for _p in _problems:
            log(f"!! CHECK FAILED: {_p}")
        raise SystemExit(1)
    log("all output checks passed")
""".replace("@@PAR@@", str(N_PARITY)).replace("@@TOL@@", str(PARITY_TOL))


def build(output, rehearse=False, n_studies=N_REHEARSAL):
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    prob = load_builder(repo / "kaggle/submit-fresh10-24ep-prob/build.py")
    prob.build(output)
    output = Path(output)
    script_path = output / "script.py"
    source = script_path.read_text(encoding="utf-8")

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one build marker: {old!r} (found {source.count(old)})")
        source = source.replace(old, new, 1)

    replace_once("from src.model.members import member_fingerprint, refuse_reason  # noqa: E402",
                 "from src.model.members import member_fingerprint, refuse_reason  # noqa: E402\n"
                 "from src.model.tta import TTA_VIEWS, TTA_WEIGHTS, translate_images, tta_probs  # noqa: E402")
    replace_once("    models, skipped = [], []\n", "    models, skipped = [], []\n    member_keys = []\n")
    replace_once("        models.append((net, int(slices), head, name))",
                 "        models.append((net, int(slices), head, name))\n"
                 "        member_keys.append((hidden, blob.get('fold')))")

    if rehearse:
        replace_once('test = pd.read_csv(f"{COMP}/test.csv")\ntest_series = pd.read_csv(f"{COMP}/test_series.csv")\n',
                     REHEARSAL_STUDIES.replace("@@TOT@@", str(int(n_studies))))
        if source.count('f"{COMP}/test_series"') != 3:
            raise ValueError("expected three test_series decode paths")
        source = source.replace('f"{COMP}/test_series"', 'f"{COMP}/train_series"')
        source = source.replace('submission.to_csv("submission.csv"', 'submission.to_csv("rehearsal_predictions.csv"')
        source = source.replace('pd.read_csv("submission.csv")', 'pd.read_csv("rehearsal_predictions.csv")')

    replace_once("    BATCH = 8\n",
                 "    BATCH = 8\n" + EVAL_CHECK_DEF.replace("@@ON@@", repr(bool(rehearse))) +
                 "    T_DECODE = T_FORWARD = 0.0\n"
                 "    ident_preds = [np.full((len(test), len(TARGETS)), 0.5, dtype=np.float32) for _ in models]\n"
                 "    T_LOOP0 = time.time()\n")
    replace_once("        chunk = test.iloc[start : start + BATCH]\n",
                 "        chunk = test.iloc[start : start + BATCH]\n        _t_dec = time.time()\n")
    replace_once("        if not rows:\n            continue\n",
                 "        T_DECODE += time.time() - _t_dec\n        if not rows:\n            continue\n")

    a = source.index("        with torch.no_grad(), autocast:")
    b = source.index("        if start % 80 == 0:")
    inference = """\
        view_caches = {}                      # per batch: translate each grid's views once
        if device.type == "cuda":
            torch.cuda.synchronize()
        _t_fwd = time.time()
        with torch.no_grad(), autocast:
            for member, (net, slices, head, _name) in enumerate(models):
                xk = x[slices]
                if head == "gattn":
                    raise RuntimeError("gattn members are not part of the TTA path")
                probs, ident = tta_probs(net, xk, m, GROUP,
                                         view_cache=view_caches.setdefault(slices, {}),
                                         return_identity=True)
                member_preds[member][rows] = probs.cpu().numpy()
                ident_preds[member][rows] = ident.cpu().numpy()
                if REHEARSAL_EVAL_CHECK and start == 0 and member == 0:
                    REHEARSAL_EVAL_CHECK_RESULT.update(eval_style_check(net, xk, m, probs))
        if device.type == "cuda":
            torch.cuda.synchronize()
        T_FORWARD += time.time() - _t_fwd
        del view_caches

"""
    source = source[:a] + inference + source[b:]

    replace_once('    log(f"  physical crop applied on {crop_ok}/{crop_total} slots")\n',
                 '    log(f"  physical crop applied on {crop_ok}/{crop_total} slots")\n'
                 + POST_CHECKS.replace("@@REHEARSE@@", repr(bool(rehearse))))
    ast.parse(source)
    script_path.write_text(source, encoding="utf-8")

    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if rehearse:
        metadata.update(id="sibusisokhumalo11/knee-rehearse-fresh10-tta", title="knee-rehearse-fresh10-tta")
        metadata["dataset_sources"] = list(metadata["dataset_sources"]) + ["sibusisokhumalo11/knee-tta-eval-ref"]
    else:
        metadata.update(id="sibusisokhumalo11/knee-submit-fresh10-24ep-prob-tta",
                        title="knee-submit-fresh10-24ep-prob-tta")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built fresh10 five-view TTA ({'REHEARSAL' if rehearse else 'submission'}) -> {output}")


if __name__ == "__main__":
    build(sys.argv[1])
