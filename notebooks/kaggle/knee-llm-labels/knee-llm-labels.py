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


def run(frame):
    rows, failures = [], 0
    reports = frame["Report"].fillna("").tolist()
    uids = frame[ID_COLUMN].tolist()
    start = time.time()
    for i in range(0, len(reports), BATCH):
        chunk = reports[i : i + BATCH]
        scores = extract_batch(chunk)
        for uid, score in zip(uids[i : i + BATCH], scores):
            if all(v == 0.0 for v in score.values()):
                failures += 1
            rows.append({ID_COLUMN: uid, **score})
        done = min(i + BATCH, len(reports))
        rate = (time.time() - start) / done
        print(f"  {done}/{len(reports)}  {rate:.1f}s/study "
              f"(eta {rate * (len(reports) - done) / 60:.0f}m)", flush=True)
    print(f"  all-zero outputs: {failures}/{len(reports)} "
          f"(parse failures or genuinely normal studies)")
    return pd.DataFrame(rows, columns=[ID_COLUMN, *TARGETS])


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
    print(f"\n--- full corpus: {len(train)} studies ---", flush=True)
    full = run(train[[ID_COLUMN, "Report"]])
    full.to_csv("/kaggle/working/labels_llm_v1.csv", index=False)
    print("wrote labels_llm_v1.csv")

    # Build the ensemble labels here, where both score sets already exist.
    # Plain unweighted mean — no per-label selection, which on n=58 gold
    # studies would be fitting the only ground truth we have.
    rule_full = RuleExtractor().extract_frame(train, id_column=ID_COLUMN)
    merged = rule_full.merge(full, on=ID_COLUMN, suffixes=("_r", "_l"))
    ens = pd.DataFrame({ID_COLUMN: merged[ID_COLUMN]})
    for label in TARGETS:
        ens[label] = (merged[f"{label}_r"] + merged[f"{label}_l"]) / 2
    ens.to_csv("/kaggle/working/labels_ensemble_v1.csv", index=False)
    print(f"wrote labels_ensemble_v1.csv ({len(ens)} studies) — "
          "publish as a Dataset and retrain against it")
