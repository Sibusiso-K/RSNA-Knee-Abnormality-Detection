# 06 — Glossary

Every term used across these docs, in plain language. Skim it once; come back when something reads
like jargon.

## Anatomy

| Term | Meaning |
|---|---|
| **Femur / Tibia / Patella** | Thigh bone / shin bone / kneecap — the three bones of the knee |
| **Condyle** | The rounded knuckle at the end of the femur. There's a medial and a lateral one |
| **Tibial plateau** | The flat top surface of the shin bone that the femur sits on |
| **Trochlea** | The groove on the femur that the kneecap slides in |
| **Medial** | The inner side, toward the other leg |
| **Lateral** | The outer side, away from the body's midline |
| **Anterior / Posterior** | Front / back |
| **Compartment** | One of the knee's three load-bearing zones: medial tibiofemoral, lateral tibiofemoral, patellofemoral. OA is scored separately in each |
| **ACL / PCL** | Anterior / posterior cruciate ligament — the crossed pair inside the joint controlling front-back stability |
| **MCL / LCL** | Medial / lateral collateral ligament — the side bands controlling sideways stability |
| **Meniscus** (pl. menisci) | C-shaped cartilage wedge on the tibial plateau; one medial, one lateral. Shock absorber |
| **Horn** | End of a meniscus — anterior or posterior. "Posterior horn of the medial meniscus" is the most commonly torn structure |
| **Articular cartilage** | The smooth cap on bone ends; losing it *is* osteoarthritis |
| **Synovium** | Lining of the joint capsule that makes lubricating fluid |
| **Hoffa's fat pad** | Fat pad behind the patellar tendon; its swelling is a clue to synovitis |
| **Suprapatellar recess** | Space above the kneecap where joint fluid pools — where effusion is most visible |
| **Popliteal fossa** | The hollow behind the knee, where a Baker's cyst forms |
| **Valgus / Varus** | Force pushing the knee inward / outward |

## Pathology (the twelve labels and related)

| Term | Meaning |
|---|---|
| **Tear** | A structural break in a ligament or meniscus |
| **Sprain** | A stretched-but-intact ligament injury |
| **Osteoarthritis (OA)** | Wear of articular cartilage, plus osteophytes and bone changes |
| **Osteophyte** | Bone spur at a joint margin; a hallmark of OA |
| **Subchondral** | Just beneath the cartilage — where OA-related cysts and marrow changes appear |
| **Effusion** | Excess fluid inside the joint |
| **Synovitis** | Inflammation/thickening of the joint lining |
| **Baker's cyst** | Fluid collection behind the knee, between semimembranosus and medial gastrocnemius |
| **Bone contusion / bruise** | Bleeding and swelling inside bone marrow with no break |
| **Bone marrow edema** | Fluid signal inside bone — visible only on fat-suppressed fluid-sensitive images |
| **Fracture** | An actual break in bone |
| **Pivot-shift pattern** | The paired bruise of lateral femoral condyle + posterolateral tibia that strongly implies an ACL tear |

## MRI

| Term | Meaning |
|---|---|
| **Sequence** | The recipe of radio pulses that determines image contrast (T1, T2, PD, STIR…) |
| **T1-weighted** | Fat bright, fluid dark. Good anatomy |
| **T2-weighted / PD / STIR** | Fluid bright — the **fluid-sensitive** family. Where pathology shows |
| **Fat suppression (FS)** | Cancelling fat's bright signal so edema underneath becomes visible |
| **Fluid_Sensitive** | Competition-provided flag: 1 if the sequence makes water bright |
| **Fat_Suppression** | Competition-provided flag: 1 if fat signal is suppressed |
| **Sagittal** | Slices side-to-side. Best for ACL and menisci |
| **Coronal** | Slices front-to-back. Best for MCL and medial/lateral OA |
| **Axial** | Horizontal slices. Best for patellofemoral OA and Baker's cyst |
| **Signal** | Brightness in an MR image. "High signal" = bright |
| **TR / TE / TI** | Repetition / echo / inversion time — the timing parameters that set contrast |
| **Field strength** | Magnet power in Tesla (usually 1.5T or 3T). Higher = more detail |
| **Coil** | The receiver antenna. Affects image appearance — part of the scanner fingerprint |

## Data & DICOM

| Term | Meaning |
|---|---|
| **DICOM** | The medical imaging file format: one image slice plus a dictionary of metadata tags |
| **Study** | One patient, one scanning session. **What you predict on** |
| **Series** | One acquisition within a study (e.g. "sagittal PD"). A stack of slices |
| **Instance** | One slice = one `.dcm` file |
| **UID** | Unique identifier. `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID` |
| **Transfer syntax** | How pixel data is encoded inside the DICOM. This dataset has four, two of them compressed (JPEG Lossless, JPEG 2000) — they need extra codec libraries |
| **PixelSpacing / SliceThickness** | Real-world millimetres per pixel / per slice. Needed to resample to a common scale |
| **ImagePositionPatient** | 3D location of a slice — use it to sort slices correctly |
| **Scanner fingerprint** | Our derived grouping key (manufacturer + model + software + frequency + coil) used for grouped CV |

## Machine learning

| Term | Meaning |
|---|---|
| **AUC-ROC** | Probability the model ranks a random positive above a random negative. 0.5 = random, 1.0 = perfect. **Ranking only — calibration is irrelevant** |
| **Macro-average** | Plain unweighted mean across the twelve labels. Rare labels count as much as common ones |
| **Multi-label** | Each study can have several findings at once (unlike multi-class, where exactly one applies) |
| **GroupKFold** | Cross-validation where a group (here: a scanner) appears in only one fold — prevents leakage |
| **Leakage** | When the model learns something that won't transfer (e.g. "this site has arthritic patients"), inflating validation scores |
| **Pseudo-labelling** | Generating training labels automatically — here, from report text |
| **Soft labels** | Targets between 0 and 1 expressing confidence, rather than hard 0/1 |
| **2.5D** | Run a 2D CNN per slice, then pool features across slices. A cheap stand-in for full 3D |
| **Attention pooling** | Learning *which* slices matter when combining them, instead of averaging |
| **Backbone** | The pretrained feature-extracting body of a network |
| **BCE** | Binary cross-entropy — the standard multi-label loss |
| **TTA** | Test-time augmentation: predict on several transformed copies and average. Boosts accuracy, costs runtime |
| **Class imbalance** | Rare positives (e.g. Fracture) — needs positive weighting so the loss isn't dominated by common labels |
| **Held-out / gold labels** | The ~58 studies with real expert labels. Used to *measure*, never to train |

## Competition mechanics

| Term | Meaning |
|---|---|
| **Code competition** | You submit a *notebook* that runs on Kaggle's machines, not a CSV from your laptop |
| **Public / Private LB** | The test set is split: public scores show during the competition, private decides the winners |
| **Efficiency track** | A parallel prize scored on runtime as well as accuracy |
| **Shakeup** | Rank changes between public and private LB — expected here, given ~1,300 test studies and warned-about prevalence shifts |
