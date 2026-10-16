# PPE Compliance Classification: How to Fit "What's Missing" into 3 Classes

This document explains why a 3-class single-label softmax model cannot
represent "full PPE compliance + which item is missing," and recommends
the practical workarounds used in this repo's PPE example pipeline.

## The math problem

If you have *N* PPE items to check, the number of present/missing
combinations is **2^N**:

| N (PPE items) | combinations | min. single-label classes needed |
|---|---|---|
| 2 (hat, vest) | 4 | 4 |
| 3 (hat, vest, glasses) | 8 | 8 |
| 4 (+ gloves) | 16 | 16 |
| 5 (+ boots) | 32 | 32 |

A 3-class softmax output can only pick 1 of 3 mutually-exclusive labels per
frame. It cannot represent "hat on, vest off, glasses on" as distinct from
"hat off, vest on, glasses on." So no matter how cleverly you name those 3
classes, you lose information about *what's* missing.

## Three real options, ranked

### Option A — Multi-label single model (proper ML solution, recommended)

Train one model with **N sigmoid binary output heads** instead of one
softmax over 3 classes. For PPE with hat/vest/glasses/gloves/boots you'd
have 5 sigmoid outputs, each independently `0/1`. Output vector
`[1,0,1,1,0]` means "hat ✓, vest ✗, glasses ✓, gloves ✓, boots ✗" → tells
the worker exactly what to put on.

- One inference, all items at once, ~200 ms on ESP32-S3 int8.
- Teachable Machine can't do this directly (TM only does softmax multiclass).
- **Workflow**: train in TM as a 3-class prototype to get the backbone,
  then export to Keras and rewrite the last layer to
  `Dense(N, activation="sigmoid")` + recompile with
  `loss="binary_crossentropy"`. Roadmap: extend `tm_to_esp32.py` with a
  `--multilabel N` flag to do this transform automatically.

### Option B — Three binary TM models, run sequentially (no code change to TM)

Stay inside TM's 3-class limitation by training **three separate 2-class
models**, one per item:

| Model | Class 0 | Class 1 | Output answers |
|---|---|---|---|
| M1 (head) | `hat_missing` | `hat_present` | "is hard hat on?" |
| M2 (torso) | `vest_missing` | `vest_present` | "is hi-vis vest on?" |
| M3 (eyes) | `glasses_missing` | `glasses_present` | "are safety glasses on?" |

Run them in series on each captured frame. Total: 3 × ~150 ms = ~450 ms
per frame on ESP32-S3, still real-time-ish. Output is the 3-bit vector →
8 possible states, which **does** cover all combinations of those 3 items.

**This is what this repo ships as the reference PPE example.**

### Option C — If you truly must have exactly one 3-class single-label model

The best 3-class breakdown for PPE work, ranked by criticality:

- **Class 1 — `compliant`** : all required PPE present (positive baseline)
- **Class 2 — `head_protection_missing`** : hard hat OR glasses missing
- **Class 3 — `body_protection_missing`** : vest OR gloves OR boots missing

Worker feedback becomes actionable at the region level ("fix your head
protection" vs "fix your body protection") but doesn't tell them *which*
specific item is missing. Useful as a triage classifier; not useful as a
checklist enforcer.

## Recommended path for PPE work

Go with **Option B**: three (or four) binary TM models (hat / vest / glasses
[/ boots]), captured with your actual ESP32-CAM, run sequentially on the
ESP32-S3. It stays inside TM's workflow, runs in real-time on S3, and gives
you the exact "which item is missing" answer with no information loss.

See `docs/ppe-sampling-guide.md` for the full sampling procedure.
