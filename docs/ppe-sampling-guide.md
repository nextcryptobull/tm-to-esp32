# PPE Sampling Guide: Step-by-Step Procedure for Capturing Training Data

This document is the procedure to hand to whoever collects the image
dataset for PPE binary classifiers. It assumes the **Option B** approach
described in `ppe-approach.md`: 3 (or 4) binary TM models, one per PPE
item (hat / vest / glasses [/ boots]).

## Boots coverage note

The default 3-model setup (hat / vest / glasses) does **not** cover
safety boots. Two paths to add boots:

**Path A — Swap one item out (recommended for v1)**
Keep `hat + vest + boots` as the three models. Drop glasses. Rationale:
hat + vest + boots are the "site entry trio" required by virtually every
construction safety standard; glasses can be checked at a separate station
or by a supervisor. You still get 3 binary models × 150 ms ≈ 450 ms per
frame on ESP32-S3.

**Path B — Add a 4th binary model**
Keep `hat + vest + glasses + boots` = 4 binary models. Total inference:
4 × 150 ms ≈ 600 ms per frame on ESP32-S3 — still real-time-ish (1.6 fps),
acceptable for compliance logging rather than blocking access.

**Camera constraint to flag now**: if your ESP32-CAM is mounted at head
height, **you physically won't see boots**. You'll need either:
- A second ESP32-CAM at ~50 cm height aimed at the floor / doorway, OR
- A single wide-FOV camera at ~2.5 m height tilted down ~30° to capture
  head-to-toe. The OV2640 with the standard lens gives ~65° FOV; at 2.5 m
  height tilted 30° down you cover roughly 1.4 m × 1.4 m at floor level —
  enough for one person passing through a doorway.

---

## Phase 0 — Pre-capture setup (do once)

1. **Mount the ESP32-CAM at the production angle/height** — not a phone,
   not a DSLR. The OV2640 has different noise, color, and dynamic range
   characteristics; the model must see them during training.
2. **Set fixed resolution** — 224×224 (matches MobileNetV2 input) or
   320×240 then center-crop. Lock it; don't change mid-capture.
3. **Save 5 reference photos** of an empty scene and a person standing in
   the canonical position. Print these on paper so data collectors can
   compare "is the subject in frame like the reference?"
4. **Verify OV2640 focus** — fixed-focus lens; optimal subject distance is
   ~1.5–2.5 m. Closer = blurry face, farther = boots become <20 px.
5. **Write a capture script** that grabs a JPEG from the ESP32-CAM every
   ~1.5 s and saves with a timestamp filename. 1.5 s gives the subject
   time to reposition between shots.

## Phase 1 — Subject recruitment

6. Recruit **at least 3 people** (different heights, body types, skin
   tones, hair coverage). One subject = the model memorizes the subject,
   not the PPE.
7. Brief each on the **PPE states they'll model**: for each of the 4 items,
   they need to be photographed with it on and off, while the other items
   vary.
8. For boots specifically: recruit at least one subject wearing dark
   sneakers and one wearing actual steel-toe boots — the model needs to
   see both to learn the difference.

## Phase 2 — Per-item capture (do this 4 times, once per item)

For each item (hat, vest, glasses, boots):

### 2.1 Positive samples — item correctly worn
- Capture **50–100 photos per subject** × 3 subjects = 150–300 positives.
- For each photo, vary one of:
  - **Distance**: 1 m, 2 m, 3 m from camera
  - **Angle**: front, ¾ left, ¾ right, side
  - **Pose**: standing, walking, bending, reaching
  - **Lighting**: morning, noon, dusk, fluorescent indoor
  - **Background**: at least 3 different (concrete wall, machinery, outdoors)
- Don't reuse the same pose/lighting combination for more than 3 frames.

### 2.2 Negative samples — item missing, others mixed
- Same 3 subjects, same distance/angle/pose/lighting matrix.
- 150–300 negatives.
- **Crucial**: keep the OTHER items in mixed states (sometimes present,
  sometimes missing). If every "hat missing" photo also has vest missing,
  the model learns the shortcut "vest missing → hat missing" and fails in
  the field.

### 2.3 Hard negatives — item "looks" present but isn't
- Minimum **30–50 hard negatives per item**:
  - Hat → beanie, hoodie hood, baseball cap (not hard hat)
  - Vest → yellow t-shirt, orange hoodie (not ANSI class 2/3 vest)
  - Glasses → sunglasses, reading glasses (not ANSI Z87 safety glasses)
  - Boots → dark sneakers, low-top work shoes (not steel-toe)
- These prevent embarrassing false positives in production.

### 2.4 Empty / no-subject frames
- 30–50 frames of the empty scene (no person in frame).
- Lets the model output "no person detected" instead of falsely claiming
  all PPE is missing.

## Phase 3 — Folder organization

Organize on disk like this so the converter script can use them as
`--rep-data`:

```
samples/
  hat/
    present/   # 150-300 imgs
    missing/   # 150-300 imgs
    hard_neg/  # 30-50 imgs
    empty/     # 30-50 imgs (shared across items, can symlink)
  vest/
    present/
    missing/
    hard_neg/
  glasses/
    present/
    missing/
    hard_neg/
  boots/
    present/
    missing/
    hard_neg/
```

## Phase 4 — Quality control (before training)

10. **Manually scan every image.** Delete: motion-blurred, partially
    occluded, subject out of frame, JPEG artifacts heavy enough to obscure
    the item.
11. **Verify class balance**: positives ≈ negatives; hard_neg ≥ 10% of
    positives; empty ≥ 10% of positives.
12. **Subject-disjoint split**: put Subject 3's entire set aside as the
    test set. Train/val on Subjects 1 + 2 only (80/10 split). This catches
    "the model learned Subject 1's haircut, not hats" failures.
13. **Resolution check**: confirm all images are the same resolution you
    set in step 2. Mixed resolutions silently break training.

## Phase 5 — Train per item in Teachable Machine

14. For each item, create a new TM project, upload `present/` and
    `missing/` folders as two classes (skip `hard_neg` and `empty` for
    now — TM's interface doesn't have a "hard negative" concept, but you
    can add them as a third class called `negative_other` and just not
    use it at inference).
15. Train, then check accuracy on the held-out Subject 3 test set.
    **Target: >90% accuracy.**
16. If accuracy < 90%: capture more hard negatives (double them),
    retrain.

## Phase 6 — Convert to ESP32

Run the converter once per item:

```bash
python tm_to_esp32.py -i ./hat_model   -o ./out_hat   -q int8 -r ./samples/hat/present   --c-header
python tm_to_esp32.py -i ./vest_model  -o ./out_vest  -q int8 -r ./samples/vest/present  --c-header
python tm_to_esp32.py -i ./glasses_model -o ./out_glasses -q int8 -r ./samples/glasses/present --c-header
python tm_to_esp32.py -i ./boots_model -o ./out_boots -q int8 -r ./samples/boots/present --c-header
```

The `--rep-data` flag matters: int8 quantization calibrates activation
ranges on real ESP32-CAM images, not random data, which preserves accuracy
in deployment.

## Phase 7 — Field validation (post-deploy)

18. Flash all 4 `.tflite` files to the ESP32-S3, run inference on live
    frames for 1 hour.
19. Log every prediction + manually label the same stream as ground truth.
20. Build a per-item confusion matrix. If field accuracy < test-set
    accuracy by more than 5%, your training data doesn't cover real
    lighting/angles — go back to Phase 2 and capture more variety.

---

## Time budget

Per item, expect ~400–700 photos. For 4 items, that's ~1,600–2,800 photos
total. At 1.5 s per capture, that's ~45–75 minutes of pure shutter time,
plus ~3–4 hours total including subject repositioning and lighting changes.
Plan a full day for one person to collect the dataset.
