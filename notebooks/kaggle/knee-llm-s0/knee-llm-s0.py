import os
os.environ["N_SHARDS"] = "3"
os.environ["SHARD"] = "0"
"""KAGGLE LLM LABEL EXTRACTION — paste as a single cell.

Settings: Accelerator GPU T4 x2 | Internet ON (to pull weights from HF)

Why this exists, in one line: **label noise is now the binding constraint, not
model capacity.** Session 11 measured Synovitis at 0.836 from imaging while its
*text* labels score only 0.630 — the imaging model learns the finding better
than the labels teaching it. Session 7 established regex fundamentally cannot
close that gap (only 3/27 gold-positive Synovitis reports contain any
thickening word at all). So the ceiling moves by improving labels, and that is
what this notebook is for.

Runs an open-weights multilingual LLM locally on the Kaggle GPU. Open weights
on Kaggle is deliberate over a hosted API: it is free, offline at inference,
and sidesteps the "minimal cost / reasonably accessible" test the host reserved
the right to judge after the fact (docs/08-model-and-rules.md). The rules now
*permit* commercial APIs — we simply don't need one.

Two-stage by design:
  Stage 1 (default) — the **58 gold studies only**. That is the decisive
    number: does the LLM beat the rule extractor's 0.757? ~5 minutes.
  Stage 2 (RUN_FULL=True) — all 4,407 studies, only worth the GPU hours if
    stage 1 wins.

The prompt is imported from src.extract.llm rather than copied, so the Ollama
path and this one cannot silently diverge.
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

# --- src bootstrap (see kaggle_01_smoke.py) ------------------------------
PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"


def find_dir(marker, max_depth=5):
    if not os.path.isdir(INPUT):
        return None
    stack = [(INPUT, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        if marker in entries:
            return directory
        for entry in entries:
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                stack.append((path, depth + 1))
    return None


print("=== resolving input paths ===")
_src = find_dir("labels.py")
COMP = find_dir("train_series.csv")
print("  src :", _src)
print("  comp:", COMP)
if _src is None or COMP is None:
    raise SystemExit("Attach knee-src and the competition in the sidebar.")
if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)
# -------------------------------------------------------------------------

from src.extract import RuleExtractor                    # noqa: E402
from src.extract.evaluate import evaluate, format_report  # noqa: E402
from src.extract.llm import SYSTEM_PROMPT, _coerce_score  # noqa: E402
from src.labels import ID_COLUMN, TARGETS                 # noqa: E402

# Qwen2.5-7B-Instruct: strong multilingual instruction-following, and this
# corpus spans 11+ languages including Greek, Bulgarian and Croatian, which
# smaller models handle poorly. 4-bit keeps it ~5 GB so it fits a T4 with room
# for activations; fp16 would be ~15 GB of 15 and OOM mid-run.
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Stage 1 verdict (session 13, n=58): LLM 0.7806 vs rule 0.7565 — inside the
# noise band alone, BUT the plain mean of the two scores 0.8234, +0.067 over
# the rule extractor. The errors are complementary: the LLM wins Effusion by
# +0.178 (our worst label) while the rule extractor wins MCL by +0.204. So the
# full corpus run is justified — we need LLM scores for all 4,407 studies to
# build ensemble labels.
RUN_FULL = True
MAX_NEW_TOKENS = 300
BATCH = 4

# --- sharding: why this exists ------------------------------------------
# The first full-corpus run wrote labels_ensemble_v1.csv only after all 4,407
# studies finished. On 2026-08-10 that run died at ~7.5 h with status ERROR and
# a 0-byte log, and it cost the entire corpus pass.
#
# MEASURED, and it corrects an earlier assumption written here: that failed run
# **did** commit its /kaggle/working files (llm_run_info.txt and
# llm_scores_gold.csv both came back). So on an *exception* Kaggle still
# commits, and chunked writes alone would have saved most of the corpus. It is
# only a hard timeout/kill where partials are lost. Both mechanisms below earn
# their place: CHUNK flushing survives the error case that actually happened,
# sharding bounds runtime so the timeout case cannot arise.
#
# The fix that holds is to make each run small enough to finish:
# shard the corpus, run each shard as its own kernel, publish each partial as
# a private Dataset, and let later shards resume past whatever is already done.
#
# N_SHARDS=1 reproduces the old single-run behaviour.
N_SHARDS = int(os.environ.get("N_SHARDS", "3"))
SHARD = int(os.environ.get("SHARD", "0"))
# How often to flush a partial CSV. This does not survive a hard timeout (see
# above) but it does survive an exception, and it lets a watching human see
# real progress in the log rather than silence.
CHUNK = 200

train = pd.read_csv(f"{COMP}/train.csv")
present = [c for c in TARGETS if c in train.columns]
gold = train[train[present].notna().any(axis=1)][[ID_COLUMN, "Report", *present]]
print(f"\ngold studies: {len(gold)}  |  full corpus: {len(train)}")

# --- rule-extractor baseline (the number to beat) ------------------------
print("\n--- rule extractor (baseline) ---")
rule_scores = RuleExtractor().extract_frame(gold, id_column=ID_COLUMN)
rule_reports, rule_macro = evaluate(gold, rule_scores, 0.5, ID_COLUMN)
print(format_report(rule_reports, rule_macro))

# --- load the LLM --------------------------------------------------------
print(f"\n--- loading {MODEL} ---", flush=True)
import subprocess  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# bitsandbytes is NOT in the Kaggle image (measured — the first run died on
# exactly this). Install it, but treat 4-bit as best-effort rather than a hard
# requirement: a pip failure or a CUDA mismatch should cost us model size, not
# the whole run. FALLBACK is 3B in fp16 (~6 GB), which fits a T4 unquantised;
# 7B fp16 would be ~15 GB of 15 and OOM during generation.
FALLBACK = "Qwen/Qwen2.5-3B-Instruct"
t0 = time.time()
model = None

try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"],
        check=True, timeout=600,
    )
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
    print(f"  loaded {MODEL} in 4-bit")
except Exception as exc:
    print(f"  4-bit path unavailable ({type(exc).__name__}: {exc})")
    print(f"  falling back to {FALLBACK} in fp16")
    MODEL = FALLBACK
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )

model.eval()
print(f"  loaded in {time.time() - t0:.0f}s | "
      f"GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def extract_batch(reports):
    """One generation pass over a list of report strings -> list of score dicts."""
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"REPORT:\n{r}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in reports
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy: label extraction is not creative
            pad_token_id=tokenizer.pad_token_id,
        )
    texts = tokenizer.batch_decode(
        out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )

    import json
    import re

    results = []
    for text in texts:
        parsed = None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except ValueError:
                parsed = None
        # A parse failure scores every label 0.0 rather than guessing. Counted
        # and reported below — a high failure rate invalidates the comparison.
        results.append(
            {label: _coerce_score((parsed or {}).get(label)) for label in TARGETS}
        )
    return results


def load_done():
    """Every study already scored by a previous shard or run.

    Reads any llm_partial_*.csv / labels_llm_v1.csv reachable under /kaggle/input
    (attach earlier shards as Datasets) plus anything in /kaggle/working from
    this session. Returns a frame that may be empty but always has the columns.
    """
    found = []
    for root in (INPUT, "/kaggle/working"):
        if not os.path.isdir(root):
            continue
        for directory, _dirs, files in os.walk(root):
            for name in files:
                if name.startswith("llm_partial_") or name == "labels_llm_v1.csv":
                    try:
                        found.append(pd.read_csv(os.path.join(directory, name)))
                    except Exception as exc:
                        print(f"  skipped unreadable {name}: {exc}")
    if not found:
        return pd.DataFrame(columns=[ID_COLUMN, *TARGETS])
    prior = pd.concat(found, ignore_index=True).drop_duplicates(
        subset=[ID_COLUMN], keep="last"
    )
    print(f"  resuming past {len(prior)} already-scored studies")
    return prior[[ID_COLUMN, *TARGETS]]


def run(frame, partial_path=None):
    """Score `frame`, skipping studies already present in a partial, flushing
    to `partial_path` every CHUNK studies so an exception costs one chunk."""
    prior = load_done() if partial_path else pd.DataFrame(columns=[ID_COLUMN])
    done_uids = set(prior[ID_COLUMN]) if len(prior) else set()
    todo = frame[~frame[ID_COLUMN].isin(done_uids)]
    if len(todo) < len(frame):
        print(f"  {len(frame) - len(todo)} of {len(frame)} already done — "
              f"scoring the remaining {len(todo)}")

    rows, failures = [], 0
    reports = todo["Report"].fillna("").tolist()
    uids = todo[ID_COLUMN].tolist()
    start = time.time()

    def flush():
        if not (partial_path and rows):
            return
        frame_so_far = pd.DataFrame(rows, columns=[ID_COLUMN, *TARGETS])
        combined = pd.concat(
            [prior[[ID_COLUMN, *TARGETS]], frame_so_far], ignore_index=True
        ) if len(prior) else frame_so_far
        combined.to_csv(partial_path, index=False)

    crashes = 0
    for i in range(0, len(reports), BATCH):
        chunk = reports[i : i + BATCH]
        # Two full-corpus runs have now died mid-pass with a 0-byte Kaggle log,
        # taking the GPU hours with them. Whatever the cause (OOM on a long
        # report, CUDA fragmentation), losing one batch of 4 is enormously
        # cheaper than losing the run. Score the batch zero, clear the cache,
        # and carry on — the count is printed loudly below, so this degrades
        # visibly rather than silently.
        try:
            scores = extract_batch(chunk)
        except Exception as exc:
            crashes += 1
            print(f"  !! batch at {i} failed ({type(exc).__name__}: {exc}) — "
                  f"scored 0, continuing", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            scores = [{label: 0.0 for label in TARGETS} for _ in chunk]
        for uid, score in zip(uids[i : i + BATCH], scores):
            if all(v == 0.0 for v in score.values()):
                failures += 1
            rows.append({ID_COLUMN: uid, **score})
        done = min(i + BATCH, len(reports))
        if done % CHUNK < BATCH:
            flush()
        rate = (time.time() - start) / done
        print(f"  {done}/{len(reports)}  {rate:.1f}s/study "
              f"(eta {rate * (len(reports) - done) / 60:.0f}m)", flush=True)
    flush()
    print(f"  all-zero outputs: {failures}/{len(reports)} "
          f"(parse failures or genuinely normal studies)")
    if crashes:
        print(f"  !! {crashes} batch(es) CRASHED and were scored 0 — up to "
              f"{crashes * BATCH} studies carry junk labels. Re-run this shard "
              f"after deleting those rows if the count is material.")

    scored = pd.DataFrame(rows, columns=[ID_COLUMN, *TARGETS])
    if len(prior):
        scored = pd.concat([prior[[ID_COLUMN, *TARGETS]], scored], ignore_index=True)
        scored = scored[scored[ID_COLUMN].isin(frame[ID_COLUMN])]
    return scored


print(f"\n--- LLM extractor on {len(gold)} gold studies ---", flush=True)
llm_scores = run(gold)
llm_reports, llm_macro = evaluate(gold, llm_scores, 0.5, ID_COLUMN)
print(format_report(llm_reports, llm_macro))

print("\n=== VERDICT ===")
print(f"  rule extractor : {rule_macro:.4f}")
print(f"  LLM ({MODEL.split('/')[-1]}) : {llm_macro:.4f}")
if llm_macro and rule_macro:
    delta = llm_macro - rule_macro
    print(f"  delta          : {delta:+.4f}")
    print(
        "  -> LLM wins; set RUN_FULL=True and relabel the corpus"
        if delta > 0.02 else
        "  -> not a clear win on n=58. Do NOT spend GPU hours relabelling "
        "4,407 studies on this margin; 58 studies is too small to resolve it."
    )

llm_scores.to_csv("/kaggle/working/llm_scores_gold.csv", index=False)

# Record WHICH model produced these scores. The first successful run's Kaggle
# log came back empty (0 bytes), leaving no way to tell whether the 4-bit 7B
# path or the 3B fp16 fallback generated the numbers — fine for a one-off
# comparison, useless for reproducing it. A sidecar file survives that.
with open("/kaggle/working/llm_run_info.txt", "w") as fh:
    fh.write(f"model={MODEL}\ngold_macro_auc={llm_macro}\nrule_macro_auc={rule_macro}\n")
print(f"\nrecorded model={MODEL} in llm_run_info.txt")

if RUN_FULL:
    corpus = train[[ID_COLUMN, "Report"]]
    # Strided rather than contiguous slicing: every shard then draws from the
    # whole corpus, so a shard's rate and all-zero count generalise instead of
    # reflecting whatever site happens to sit in that block of rows.
    shard = corpus.iloc[SHARD::N_SHARDS] if N_SHARDS > 1 else corpus
    partial = f"/kaggle/working/llm_partial_{SHARD:02d}of{N_SHARDS:02d}.csv"
    print(f"\n--- full corpus: shard {SHARD + 1}/{N_SHARDS}, "
          f"{len(shard)} of {len(corpus)} studies ---", flush=True)
    print(f"    partial -> {partial}", flush=True)

    full = run(shard, partial_path=partial)
    print(f"wrote {os.path.basename(partial)} ({len(full)} studies)")

    # The ensemble needs the WHOLE corpus, so build it only once every shard is
    # in hand. Writing a partial labels_ensemble_v1.csv would be worse than
    # writing nothing: kaggle_02_train.py prefers that filename automatically
    # and would silently train on a fraction of the labels.
    everything = load_done()
    missing = len(corpus) - everything[ID_COLUMN].isin(corpus[ID_COLUMN]).sum()
    if missing > 0:
        print(f"\n=== SHARD DONE — corpus still {missing} studies short ===")
        print(f"  Publish {os.path.basename(partial)} as a PRIVATE Kaggle Dataset,")
        print("  attach it to the next run, and set SHARD to the next index.")
        print("  The ensemble is NOT built yet — nothing to retrain on.")
    else:
        everything.to_csv("/kaggle/working/labels_llm_v1.csv", index=False)
        print(f"\nwrote labels_llm_v1.csv ({len(everything)} studies)")

        # Plain unweighted mean — no per-label selection, which on n=58 gold
        # studies would be fitting the only ground truth we have.
        rule_full = RuleExtractor().extract_frame(train, id_column=ID_COLUMN)
        merged = rule_full.merge(everything, on=ID_COLUMN, suffixes=("_r", "_l"))
        ens = pd.DataFrame({ID_COLUMN: merged[ID_COLUMN]})
        for label in TARGETS:
            ens[label] = (merged[f"{label}_r"] + merged[f"{label}_l"]) / 2
        ens.to_csv("/kaggle/working/labels_ensemble_v1.csv", index=False)
        print(f"wrote labels_ensemble_v1.csv ({len(ens)} studies) — "
              "publish as a Dataset and retrain against it")
