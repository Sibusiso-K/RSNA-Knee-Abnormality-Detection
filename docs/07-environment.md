# 07 — Environment & workflow

How to pick this project up on **any** machine, and how the local/Kaggle split works.

---

## 1. The split, in one table

| | Local machine | Kaggle notebook |
|---|---|---|
| **Holds** | Code, docs, CSVs, label experiments | The 570 GB of DICOMs (pre-mounted, no download) |
| **Does** | EDA on text/tabular, report label extraction, writing code | DICOM decoding, preprocessing, training, inference, submission |
| **Needs** | Python + a few hundred MB | GPU quota (~30 h/week free). **T4 only — P100 cannot run our model** |
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

1. Competition page → **Code** → **New Notebook**. The competition data is attached automatically —
   but **not** at `/kaggle/input/<slug>/`. Measured session 10: this account gets a nested tree,
   `/kaggle/input/competitions/rsna-knee-abnormality-detection/`. Resolve it by search, never by
   hardcoded path (see §6).
2. Settings → **Accelerator: GPU T4 ×2**. **Not P100** — Kaggle's PyTorch build supports sm_70+
   and a P100 is sm_60, so every CUDA op fails with `cudaErrorNoKernelImageForDevice`. Watch the
   weekly quota — it's finite and resets weekly, not daily.
3. **Internet must be OFF for submission notebooks.** You can enable it while developing, but the
   final submission runs offline. Anything you need (pretrained weights, your extracted labels, a
   preprocessing cache) must be attached as a **Kaggle Dataset**, not downloaded at runtime.
4. Save a copy of every meaningful notebook back into `notebooks/` in this repo so the work survives
   outside Kaggle.

### Kaggle Datasets we'll create

| Dataset | Contents | Why |
|---|---|---|
| `labels-v*` | Report-derived labels, one row per study | Consumed offline by training notebooks |
| `model-weights-v*` | Trained checkpoints | Inference notebook loads these offline |

Version them by name (`v1`, `v2`, …). A submission that silently changes underneath you is
unreproducible.

(A `preproc-cache` dataset was planned but is **not needed**: measured load is 1.3 s/study, so an
epoch is under an hour and training decodes DICOMs directly.)

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

---

## 6. Kaggle CLI workflow (session 10)

Everything is driven from the CLI, not the browser. `kernels push` is reproducible and
version-controlled; file-picker automation is where browser-driven setups break.

```bash
# datasets (both PRIVATE — labels_v1.csv is competition-derived)
kaggle datasets create -p <dir>          # first time
kaggle datasets version -p <dir> -m "…" --dir-mode tar
```

```bash
# kernels — ALWAYS pass the accelerator explicitly
kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4
kaggle kernels status  sibusisokhumalo11/knee-train
kaggle kernels output  sibusisokhumalo11/knee-train -p out/
```

### Three traps, all hit for real

- **`--accelerator` is case-sensitive and silently ignores invalid values.** `nvidiaTeslaT4` is
  accepted without error and falls back to a **P100**, which Kaggle's PyTorch (sm_70+) cannot run
  at all. Use **`NvidiaTeslaT4`**. Omitting the flag entirely also gives a P100.
- **Kaggle mounts inputs as a nested tree**: `/kaggle/input/competitions/<slug>/` and
  `/kaggle/input/datasets/<owner>/<slug>/`. Most public example code assumes flat
  `/kaggle/input/<slug>/` and will not work here. Resolve paths by searching for marker files.
- **Kaggle strips the top-level folder of an uploaded dataset**, so a dataset built from `src/`
  arrives as the *contents* of `src/`. Rebuild the package before importing.

Kernel logs come back as JSON, not plain text:

```bash
python -c "import json,glob;[print(e.get('data','').rstrip()) for f in glob.glob('out/*.log') for e in json.load(open(f))]"
```
