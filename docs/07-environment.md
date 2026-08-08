# 07 — Environment & workflow

How to pick this project up on **any** machine, and how the local/Kaggle split works.

---

## 1. The split, in one table

| | Local machine | Kaggle notebook |
|---|---|---|
| **Holds** | Code, docs, CSVs, label experiments | The 570 GB of DICOMs (pre-mounted, no download) |
| **Does** | EDA on text/tabular, report label extraction, writing code | DICOM decoding, preprocessing, training, inference, submission |
| **Needs** | Python + a few hundred MB | GPU quota (~30 h/week free, T4×2 or P100) |
| **Source of truth** | This git repo | Notebooks pushed back to the repo under `notebooks/` |

**Never download the DICOMs.** 569.76 GB / 819,640 files. See [03-data-guide.md](03-data-guide.md).

---

## 2. Setting up from scratch on a new machine

```bash
git clone https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection.git
cd RSNA-Knee-Abnormality-Detection
python -m venv .venv
```

Activate it — Windows PowerShell: `.venv\Scripts\Activate.ps1` · Git Bash: `source .venv/Scripts/activate` · macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

Then read [00-state.md](00-state.md) to find out where the project actually is.

### Kaggle API credentials

1. Join the competition (accept the rules) — downloads fail with a 403 until you do.
2. Kaggle → avatar → **Settings** → **API** → **Create New Token** → saves `kaggle.json`.
3. Place it:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`
4. Verify: `kaggle competitions list` should return results, not an auth error.

`kaggle.json` is gitignored. **Never commit it** — it's an account credential.

### Get the small files

```bash
bash scripts/download_data.sh
```

This pulls only `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, and
`sample_submission.csv` into `data/` (gitignored).

---

## 3. Working on Kaggle

1. Competition page → **Code** → **New Notebook**. The competition data is attached automatically at
   `/kaggle/input/rsna-knee-abnormality-detection/`.
2. Settings → **Accelerator: GPU T4 ×2** (or P100). Watch the weekly quota — it's finite and it
   resets weekly, not daily.
3. **Internet must be OFF for submission notebooks.** You can enable it while developing, but the
   final submission runs offline. Anything you need (pretrained weights, your extracted labels, a
   preprocessing cache) must be attached as a **Kaggle Dataset**, not downloaded at runtime.
4. Save a copy of every meaningful notebook back into `notebooks/` in this repo so the work survives
   outside Kaggle.

### Kaggle Datasets we'll create

| Dataset | Contents | Why |
|---|---|---|
| `labels-v*` | Report-derived labels, one row per study | Consumed offline by training notebooks |
| `preproc-cache-v*` | Preprocessed `.npy` volumes | So training never re-decodes 800k DICOMs |
| `model-weights-v*` | Trained checkpoints | Inference notebook loads these offline |

Version them by name (`v1`, `v2`, …). A submission that silently changes underneath you is
unreproducible.

### Submission requirements

- Notebook, ≤ 9 hours runtime (CPU or GPU)
- Internet **disabled**
- Output file named exactly `submission.csv`
- Columns: `StudyInstanceUID` + the twelve label columns, in the documented order

---

## 4. Repo layout

```
docs/          ← all documentation; 00-state.md is the living one
notebooks/     ← Jupyter notebooks (copies of Kaggle work)
scripts/       ← shell helpers (data download)
src/           ← reusable Python
data/          ← gitignored. CSVs only, never DICOMs
submissions/   ← gitignored. Generated CSVs
```

## 5. Keeping the docs alive

The point of this doc set is that you can stop for three weeks and resume without reconstructing
anything from memory. That only works if the state file is honest.

**At the end of every working session:**
1. Update [00-state.md](00-state.md) — current position, next three actions, a session-log entry.
2. Add a decision-log row if a real choice was made.
3. Record any *measured* number (a CV score, a runtime) in the relevant doc — measured numbers beat
   remembered ones.
4. `git add -A && git commit && git push`.

The push is what makes it available from anywhere. An uncommitted insight is a lost insight.
