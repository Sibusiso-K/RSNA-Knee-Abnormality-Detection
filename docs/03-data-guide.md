# 03 — Data guide

Everything about what the data *is*, how it's shaped, and how to touch it without melting your
laptop.

---

## 1. The hierarchy: Study → Series → Instance

Medical imaging has a three-level structure, and all three appear in the file paths and the CSVs.

```
Study            one patient, one scanning session      → StudyInstanceUID   ← what you predict on
└── Series       one acquisition (e.g. "sagittal PD")   → SeriesInstanceUID
    └── Instance one image slice, one .dcm file         → SOPInstanceUID
```

A knee study contains **several series** (typically sagittal + coronal + axial, in a couple of
sequence types each). Each series contains **20–45 slices** (median 30), with a long tail out to a
few hundred.

**You predict at the study level** — one row per `StudyInstanceUID`, twelve probabilities. So every
model needs a way to aggregate: slices → series → study. That aggregation design is a core modelling
choice, covered in [04-method.md](04-method.md).

On disk:

```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

---

## 2. The files

### `train.csv` — one row per training study

| Column | Notes |
|---|---|
| `StudyInstanceUID` | Matches the folder under `train_series/` |
| `PatientSex` | Documented as Male/Female/blank — **but a forum thread reports it is missing from the actual file.** Verify before relying on it. |
| `Report` | Free-text radiology report, multilingual. **Train only — never present at test time.** |
| 12 label columns | `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture` — binary 0/1, **but populated for only ~58 studies** |

That last point is the whole competition. See [01-competition.md](01-competition.md).

### `train_series.csv` — one row per training series

| Column | Notes |
|---|---|
| `StudyInstanceUID` | Parent study |
| `SeriesInstanceUID` | Folder name |
| `Fluid_Sensitive` | 1 if T2/PD/STIR-like (water bright), else 0 |
| `Fat_Suppression` | 1 if fat signal is suppressed |
| `Anatomical_Plane` | `Sagittal` \| `Coronal` \| `Axial` |

These four columns are gold — they're pre-extracted protocol metadata that would otherwise take real
work to derive from DICOM headers, and they drive series routing. Note the forum probe found series
composition *alone* scores 0.5954 macro AUC, i.e. **which sequences a radiographer chose to acquire
already leaks a little about what they were looking for.**

### `test.csv` / `test_series.csv` / `test_series/`

Same schemas minus `Report` and minus labels. What ships with the competition is a **3-study
example**; at scoring time it's swapped for the real ~1,300-study test set. Your notebook must
therefore never hardcode anything about size or contents.

### `sample_submission.csv`

Every label set to 0.5. It's also the **efficiency benchmark** you must beat to qualify for that
track.

---

## 3. What a DICOM file actually is

A `.dcm` file is **one image slice plus a large tag dictionary**. The tags describe the patient, the
scanner, the acquisition physics, and the geometry. In this dataset they've been stripped to an
**allowlisted set of 86 tags** for anonymisation.

Tags that matter here:

| Tag | Why you care |
|---|---|
| `Rows`, `Columns` | Image dimensions — vary across series |
| `PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices` | Real-world millimetres per pixel. **Needed to resample everything to a common physical scale** — otherwise a model sees the same anatomy at different sizes |
| `ImagePositionPatient`, `ImageOrientationPatient` | Where the slice sits in 3D. **Use these to sort slices into correct anatomical order** — filename order is not reliable |
| `RescaleSlope`, `RescaleIntercept` | Linear transform to apply to raw pixel values |
| `Manufacturer`, `ManufacturerModelName`, `SoftwareVersions`, `ImagingFrequency`, `ReceiveCoilName` | The **scanner fingerprint** used for grouped CV (see below) |
| `SeriesDescription` | Free-text sequence name — useful sanity check on the `Fluid_Sensitive` flags |
| `RepetitionTime` (TR), `EchoTime` (TE), `InversionTime` (TI), `FlipAngle`, `MagneticFieldStrength` | The acquisition physics that determine image contrast |

### Transfer syntaxes — a real gotcha

The files come in **four different encodings**: uncompressed Explicit VR Little Endian, Implicit VR
Little Endian, **JPEG Lossless**, and **JPEG 2000**. The two compressed ones need extra codec
libraries or `pydicom` will read the header fine and then throw when you ask for `.pixel_array`.

Install `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, and `gdcm` (or `python-gdcm`) to
cover all four. **Test decoding on all four syntaxes early** — this is a classic week-8 disaster
when it should be a week-4 twenty-minute fix.

### Reading headers cheaply

To scan metadata across 800k files without decoding pixels:

```python
ds = pydicom.dcmread(path, stop_before_pixels=True)
```

That's orders of magnitude faster and is how you build the fingerprint table.

---

## 4. Intensity: the thing that trips up people coming from natural images

**MRI pixel values are not absolute.** Unlike CT (where Hounsfield units mean fixed tissue types) or
photographs (where RGB is RGB), an MRI intensity of 400 means nothing by itself — it depends on
scanner, coil, gain, and sequence. Two images of identical anatomy from different machines can have
wildly different value ranges.

So: **normalize per series**, not globally. Percentile clipping (e.g. 1st–99th) followed by scaling
to [0,1] is the standard robust choice, since it survives the bright outliers that MRI produces.

The same heterogeneity is why **site-grouped validation is mandatory** — a model can quietly learn
"this looks like a 2014 Siemens 1.5T from site 12, and that site images a lot of arthritic knees"
instead of learning to read a knee. That inflates random-fold CV by ~0.053 macro AUC, per the forum
probe.

**The scanner fingerprint** for grouping:
`Manufacturer + ManufacturerModelName + SoftwareVersions + ImagingFrequency + ReceiveCoilName`
→ 265 distinct fingerprints, the top 20 covering 45.5% of studies.

---

## 5. Size, and why we never download the DICOMs

**569.76 GB. 819,640 files.**

This machine has ~182 GB free and no NVIDIA GPU. So the workflow is split:

| Where | What happens there |
|---|---|
| **Locally** (this repo) | CSVs only. Report text, label extraction development, EDA on tabular data, all source code, all docs. Total footprint: a few hundred MB at most. |
| **Kaggle notebooks** | Everything involving DICOMs: decoding, preprocessing, caching, training, inference, submission. Data is pre-mounted at `/kaggle/input/` — **no download needed at all.** |

Pull just the small files with the Kaggle API's `-f` flag:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/
```

`scripts/download_data.sh` does exactly this for the four CSVs. Never run a bare
`kaggle competitions download -c ...` without `-f` — that starts the full 570 GB pull.

Also note: **`data/` and `submissions/` are gitignored.** Competition data must not be committed —
it's against the rules and would destroy the repo anyway.

---

## 6. The preprocessing pipeline you'll eventually need

Sketched here so the shape is clear; implementation is Phase 3 in [05-plan.md](05-plan.md).

1. **Index** — walk `train_series.csv`, read headers with `stop_before_pixels=True`, build one
   parquet table of every series with its geometry, physics, and scanner fingerprint.
2. **Select** — per study, pick the best series for each label group by plane + sequence flags. Not
   every study has every plane; you need a fallback ordering.
3. **Decode & sort** — read pixels, sort slices by `ImagePositionPatient` projected onto the slice
   normal (not by filename).
4. **Normalize** — percentile clip, scale per series.
5. **Resample** — to a fixed physical size and a fixed slice count, so tensors are uniform.
6. **Cache** — write `.npy`/`.npz` and publish as a **Kaggle Dataset**. Do this once. Re-decoding
   800k DICOMs on every training run will burn your entire weekly GPU quota on I/O.

Step 6 is the difference between iterating ten times and iterating twice.
