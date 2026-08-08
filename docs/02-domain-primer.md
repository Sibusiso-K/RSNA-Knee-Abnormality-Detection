# 02 — Domain primer: knees, MRI, and the twelve findings

> Background knowledge, not competition-specific facts. You don't need to become a radiologist, but
> you *do* need this to make good modelling choices — especially **which imaging plane and sequence
> actually shows each finding**, which drives the series-routing design in
> [04-method.md](04-method.md).
>
> The competition host also posted detailed, finding-by-finding label criteria on the forum
> ("Knee Abnormality Detection AI Challenge Overview", pinned). **Read it once you've joined** — it
> defines exactly what counts as positive for each label, which matters enormously for Phase 1.

---

## 1. The knee, in one page

The knee is a **synovial hinge joint** made of three bones:

- **Femur** — thigh bone, above. Its lower end splits into two rounded knuckles, the **medial** and
  **lateral femoral condyles**.
- **Tibia** — shin bone, below. Its flat upper surface is the **tibial plateau**.
- **Patella** — kneecap, in front, riding in a groove on the femur called the **trochlea**.

Two orientation words you'll use constantly:

- **Medial** = the inner side, toward the other leg.
- **Lateral** = the outer side, away from the body's midline.

Six of the twelve labels are laterality-specific (medial vs lateral meniscus, medial vs lateral OA),
so getting medial/lateral right is not optional — in the report text *and* in the image.

**Soft tissue structures:**

| Structure | What it is | What it does |
|---|---|---|
| **ACL** (anterior cruciate ligament) | Cord running diagonally inside the joint, from the lateral femoral condyle down and forward to the tibia | Stops the tibia sliding forward; resists rotation |
| **PCL** (posterior cruciate) | The other cruciate, behind the ACL | Stops the tibia sliding backward |
| **MCL** (medial collateral ligament) | Flat band on the inner side | Resists the knee being pushed inward (valgus stress) |
| **LCL** (lateral collateral) | Cord on the outer side | Resists varus stress *(not a competition label)* |
| **Menisci** | Two C-shaped fibrocartilage wedges sitting on the tibial plateau, one medial one lateral. Each has an anterior horn, body, and posterior horn | Shock absorption, load distribution, joint congruity |
| **Articular cartilage** | Smooth glassy layer capping the bone ends | Near-frictionless motion; its loss *is* osteoarthritis |
| **Synovium** | Membrane lining the joint capsule | Makes lubricating fluid. Too much fluid = **effusion**; inflamed lining = **synovitis** |

---

## 2. How knee MRI works, practically

### Planes

A study is made of several **series**. Each series is one acquisition — a stack of parallel image
slices through the knee in one orientation.

- **Sagittal** — slices from side to side, like slicing a loaf front-to-back. The workhorse plane.
  Best for **ACL** and **menisci**.
- **Coronal** — slices front to back, like a slice through a standing person facing you. Best for
  **MCL** and for **medial vs lateral compartment OA**, because it puts both compartments side by
  side in one image.
- **Axial** — horizontal slices, looking up through the leg. Best for **patellofemoral** structures
  and for tracking a **Baker's cyst** to its neck.

`train_series.csv` gives you `Anatomical_Plane` directly. Use it.

### Sequences and the two flags you're given

MRI contrast depends on the pulse sequence. You don't get the sequence name — you get two engineered
flags in `train_series.csv`, which is actually more useful:

- **`Fluid_Sensitive = 1`** — T2, proton density (PD), STIR and similar. **Water appears bright.**
  Anything involving fluid, swelling, or inflammation lights up here: effusion, bone contusion,
  Baker's cyst, synovitis, acute ligament injury.
- **`Fluid_Sensitive = 0`** — T1-weighted and similar. Fluid is dark, fat is bright. Good anatomy
  and good for cortical bone detail and marrow replacement.
- **`Fat_Suppression = 1`** — the bright signal from fat is deliberately cancelled. Combined with
  fluid sensitivity, this is what makes **bone marrow edema** visible: normal fatty marrow goes
  dark, and any edema within it glows. **Bone contusion is essentially invisible without this.**

The rule of thumb that matters for modelling: **fluid-sensitive + fat-suppressed sequences carry
most of the pathology signal**; non-fat-suppressed sequences carry anatomy and structure.

### Why the images are so heterogeneous

The dataset spans many countries, scanners, field strengths, and protocols. That means varying
intensity scales, resolutions, slice thicknesses, and orientations. **MRI intensity values are not
absolute** — unlike CT's Hounsfield units, a pixel value of 400 means nothing on its own. It's only
meaningful relative to other tissue in the same image. This is why per-series normalization is
mandatory, and why a model can accidentally learn "which scanner made this" instead of "what's wrong
with this knee". See the site-grouped CV discussion in [04-method.md](04-method.md).

---

## 3. The twelve findings

For each: what it is, where it's best seen, and what makes it hard.

### Ligaments

**1. ACL — anterior cruciate ligament injury**
A normal ACL appears as a dark, continuous, straight band running parallel to a line along the roof
of the intercondylar notch. A tear shows as discontinuity, a wavy or horizontal course, or bright
signal replacing the dark fibres. Powerful **secondary signs** exist: the classic "pivot-shift" bone
contusion pattern (bruising of the lateral femoral condyle *and* the posterolateral tibial plateau
together), forward displacement of the tibia, and buckling of the PCL. A model that learns these
secondary signs will do well — and note they appear on *fluid-sensitive* images, not just the
sagittal ACL slice.
*Best seen:* sagittal, fluid-sensitive. *Difficulty:* moderate — one of the more learnable labels.

**2. MCL — medial collateral ligament injury**
Graded by severity: sprain (surrounding swelling, intact fibres) → partial tear → complete
disruption. On coronal fluid-sensitive images it shows as fluid and high signal around or through
the flat medial band.
*Best seen:* coronal, fluid-sensitive with fat suppression. *Difficulty:* moderate; low-grade
sprains are subtle and are where reader disagreement lives.

### Menisci

**3. Medial Meniscus tear** and **4. Lateral Meniscus tear**
A normal meniscus is a uniformly dark triangle in cross-section. A tear is a **linear bright signal
that reaches an articular surface** — that surface-contact requirement is the actual diagnostic
criterion, and internal signal that doesn't reach a surface is degeneration, not a tear. The medial
meniscus posterior horn is the most commonly torn structure in the knee.
*Best seen:* sagittal primarily, confirmed on coronal. *Difficulty:* **hard.** Tears can occupy two
or three slices out of thirty. This is the label most likely to sink your macro AUC, and the one
that most punishes aggressive slice downsampling.

### Osteoarthritis — three compartments

The knee is split into three compartments, and OA is scored separately in each:

**5. Medial OA** — medial tibiofemoral (inner femur-on-tibia).
**6. Lateral OA** — lateral tibiofemoral (outer femur-on-tibia).
**7. PF OA** — patellofemoral (kneecap against its femoral groove).

All three are the same disease process: **cartilage thinning or loss**, **osteophytes** (bone spurs
at the joint margins), **subchondral changes** (cysts and marrow signal change under the cartilage),
and joint space narrowing. Medial compartment OA is by far the most common.
*Best seen:* medial/lateral on coronal; PF on **axial**. *Difficulty:* moderate, but graded severity
means the positive/negative threshold is a judgement call — check the host's criteria post.

### Fluid and inflammation

**8. Effusion — excess joint fluid**
Bright fluid distending the joint, collecting most visibly in the **suprapatellar recess** above the
kneecap. Conceptually the easiest of the twelve: a big bright pool that's hard to miss on
fluid-sensitive images. The judgement is *how much* counts as abnormal — a small amount is normal.
*Best seen:* sagittal or axial, fluid-sensitive. *Difficulty:* **easy** — expect this to be your
best AUC.

**9. Synovitis — inflammation of the joint lining**
Thickened synovium, often with swelling and signal change in **Hoffa's fat pad** (the fat behind the
patellar tendon). Usually diagnosed with intravenous contrast, which most of these studies almost
certainly won't have — so it must be inferred from thickening and secondary signs.
*Best seen:* axial and sagittal, fluid-sensitive. *Difficulty:* **hard**, and probably the noisiest
label, since without contrast radiologists disagree about it a lot.

**10. Baker's cyst (popliteal cyst)**
A fluid collection behind the knee, in a very specific location: between the semimembranosus tendon
and the medial head of the gastrocnemius muscle. That anatomical neck is what distinguishes it from
any other fluid back there.
*Best seen:* axial (shows the neck), fluid-sensitive. *Difficulty:* **easy** — distinctive shape and
location. The forum metadata probe found this was the most predictable label even from headers alone
(0.765 random-fold AUC), which hints it correlates with patient population.

### Bone

**11. Contusion — bone bruise**
Bleeding and edema *inside* the bone marrow with no cortical break. Appears as a fuzzy,
ill-defined bright cloud within the bone on fluid-sensitive **fat-suppressed** images — and is
nearly invisible on anything else. Their pattern is diagnostically loaded: certain bruise
distributions imply specific ligament injuries (see ACL above).
*Best seen:* any plane, but **must** be fat-suppressed fluid-sensitive. *Difficulty:* moderate —
easy to see, but only on the right sequence. **Series selection matters more than model capacity
here.**

**12. Fracture**
An actual break in bone. Shows as a low-signal line through the marrow with surrounding edema, plus
interruption of the dark cortical rim.
*Best seen:* fluid-sensitive fat-suppressed for the edema; T1 for the fracture line.
*Difficulty:* **hard because it's rare.** The forum probe scored it worst under site-grouped folds
(0.519 — essentially random). Rarity plus macro-averaging means this single label can cost you as
much as ACL does. Plan for positive-class weighting.

---

## 4. What this implies for the model

| Signal | Plane that shows it | Sequence needed |
|---|---|---|
| ACL, Medial/Lateral Meniscus | Sagittal | Fluid-sensitive |
| MCL, Medial OA, Lateral OA | Coronal | Fluid-sensitive (+FS for MCL) |
| PF OA, Synovitis | Axial | Fluid-sensitive |
| Effusion, Baker's | Axial/Sagittal | Fluid-sensitive |
| Contusion, Fracture | Any | Fluid-sensitive **+ fat-suppressed** |

Three consequences:

1. **Don't feed every series into one model.** Route series to label groups by plane and sequence
   flags. You get those flags for free in `train_series.csv`.
2. **Expected difficulty ordering**, easiest to hardest: Effusion, Baker's ≫ ACL, MCL, the three OA
   labels, Contusion ≫ menisci, Synovitis, Fracture. Since the metric is the *unweighted mean* of
   twelve AUCs, effort spent on the hard tail is worth more than polishing Effusion.
3. **Slice resolution matters unevenly.** Effusion survives aggressive downsampling; a meniscal tear
   spanning three slices does not. This is the central tension in the Efficiency track.

---

## 5. Reading a radiology report

Reports are free text in roughly nine languages, and are the *only* label source for the vast
majority of studies. Three linguistic traps to design the extractor around — see
[04-method.md](04-method.md):

- **Negation.** "No evidence of ACL tear" and "ACL tear" share almost every keyword. Naive matching
  gets these exactly backwards.
- **Uncertainty.** "Possible", "cannot exclude", "suspicious for" — you must decide whether hedged
  findings count as positive, and apply that consistently.
- **Laterality.** "Posterior horn of the medial meniscus" vs "lateral meniscus" — six of twelve
  labels depend on resolving this correctly, sometimes across a sentence boundary.

Reports also have structure — typically a Technique/Findings/Impression layout. The **Impression**
is the radiologist's summary and is usually the highest-signal, lowest-noise section to parse.
