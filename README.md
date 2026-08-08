# RSNA Knee Abnormality Detection

Working repo for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
— detecting twelve clinically important abnormalities on knee MRI.

**→ Start at [docs/00-state.md](docs/00-state.md).** It says where the project is right now and what
the next action is. Everything else is stable reference.

---

## The competition in six lines

- **Predict** twelve binary findings per knee MRI study (ligaments, menisci, three compartments of
  osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture).
- **Metric**: macro-averaged AUC-ROC — all twelve labels weighted equally.
- **Submit** a Kaggle Notebook, ≤ 9 h, **no internet**, writing `submission.csv`.
- **Deadline**: entry 2026-10-15, final submission **2026-10-22**.
- **Prizes**: $77,000 — ten leaderboard prizes plus a separate three-prize Efficiency track.
- **Field**: 7,504 entrants / 640 teams as of day 3.

## The three things that shape everything

1. **`Report` does not exist at test time.** The host confirmed it. Despite the "multimodal"
   framing, inference is pure imaging — the reports are a *training-time label source only*.
2. **Only ~58 studies have real labels.** Every other label must be extracted from multilingual
   free-text reports. **Label quality is the competition**; the vision model is secondary.
3. **There is no metadata shortcut.** DICOM headers alone give 0.598 macro AUC under
   scanner-grouped folds. Random folds overstate by ~0.053 — so **site-grouped CV from day one**.

## Documentation

| Doc | What's in it |
|---|---|
| **[00-state.md](docs/00-state.md)** | **Living file.** Current position, next actions, blockers, decision log, session log |
| [01-competition.md](docs/01-competition.md) | The rules, the data spec, forum intel, prizes, constraints |
| [02-domain-primer.md](docs/02-domain-primer.md) | Knee anatomy, how MRI works, all twelve findings explained, which plane/sequence shows what |
| [03-data-guide.md](docs/03-data-guide.md) | Study/series/instance structure, the CSVs, DICOM internals, the 570 GB problem |
| [04-method.md](docs/04-method.md) | Why each technical choice: the metric, label extraction, validation, architecture, efficiency |
| [05-plan.md](docs/05-plan.md) | Phased schedule to 2026-10-22, with risks |
| [06-glossary.md](docs/06-glossary.md) | Every term, plain language — anatomy, MRI, DICOM, ML |
| [07-environment.md](docs/07-environment.md) | Setup on any machine, local↔Kaggle workflow, doc upkeep |

New to the project? Read **00 → 02 → 03 → 04**. The rest is reference.

## Quickstart

```bash
git clone https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection.git
```

```bash
python -m venv .venv && pip install -r requirements.txt
```

Then: join the competition on Kaggle, put `kaggle.json` in place, and

```bash
bash scripts/download_data.sh
```

Full instructions in [07-environment.md](docs/07-environment.md).

> ⚠️ Only the CSVs are ever pulled locally. The imaging data is **569.76 GB / 819,640 files** and
> stays on Kaggle, where notebooks mount it for free. Never run a bare
> `kaggle competitions download -c ...` without `-f`.

## Layout

```
docs/          all documentation (start at 00-state.md)
notebooks/     Jupyter notebooks, incl. copies of Kaggle work
scripts/       shell helpers
src/           reusable Python
data/          gitignored — CSVs only
submissions/   gitignored — generated submission.csv
```
