# tm-to-esp32 — Teachable Machine → ESP32-S3 Converter

Convert a Google Teachable Machine export into an **int8-quantized TFLite
package** ready to drop into your **xiaozhi-esp32** firmware build. No
Raspberry Pi, no Arduino IDE — just Python on your laptop, then flash the
resulting `.tflite` (or embedded `model_data.cc`) onto the ESP32-S3.

## What this repo contains

| Path | Purpose |
|---|---|
| `tm_to_esp32.py` | The converter script (Python 3, TF 2.x) |
| `docs/architecture.md` | How the ESP32-CAM + ESP32-S3 + xiaozhi stack fits together |
| `docs/ppe-approach.md` | Why 3-class softmax can't do "what's missing" PPE compliance, and what to do instead |
| `docs/ppe-sampling-guide.md` | Step-by-step data capture procedure (Phases 0–7) |
| `examples/ppe_inference.h` | C API reference for running multi-model PPE inference on the S3 |
| `examples/README.md` | How to wire the converted models into xiaozhi-esp32 firmware |

## Quick start

```bash
# Install deps
pip install tensorflow-cpu==2.20.0 pillow numpy

# Verify your install end-to-end with a built-in self-test
python tm_to_esp32.py --self-test

# Convert a TM export to int8 TFLite + C headers
python tm_to_esp32.py \
    --input  ~/my_tm_export \
    --output ./out \
    --quantize int8 \
    --rep-data ~/captured_samples \
    --c-header
```

## What the converter produces

| File | Purpose |
|---|---|
| `model.tflite` | Quantized int8 model — flash to SPIFFS or partition |
| `model_data.h` + `model_data.cc` | Model bytes as a C array (xxd -i style) — embed directly in firmware |
| `labels.h` | `kLabels[]` and `kLabelCount` for label lookup |
| `labels.txt` | Plain text labels (for runtime loading from SPIFFS) |
| `model_info.json` | Metadata: input shape, dtype, classes, size, quantization mode |

## PPE compliance example (recommended use case)

For "what's missing" PPE checks, train **3 (or 4) separate binary TM models** —
one per item (hat / vest / glasses [/ boots]). Run them sequentially on each
frame on the ESP32-S3. The combined output gives you the exact "which item
is missing" answer with no information loss.

See:
- `docs/ppe-approach.md` for why a 3-class softmax model can't do this
- `docs/ppe-sampling-guide.md` for the data capture procedure
- `examples/ppe_inference.h` for the C API contract on the S3

## CLI reference

```
-i, --input      Path to TM export folder (contains keras_model.h5 or saved_model.pb)
-o, --output     Output folder (default: ./out)
-q, --quantize   int8 | float16 | none   (default: int8, recommended for ESP32-S3)
-r, --rep-data   Folder of representative sample images (improves int8 accuracy)
    --c-header   Also emit model_data.h/.cc + labels.h for static embedding
    --header-name  Base name for generated C headers (default: model_data)
    --target     tflite-micro | esp-dl  (default: tflite-micro)
    --self-test  Build a synthetic TM export, run the whole pipeline, then exit
```

## Accuracy vs. size on ESP32-S3

| Quantize | Size (MobileNetV2 224² 3-class) | Inference time on S3 @ 240 MHz | Accuracy |
|---|---|---|---|
| float32 (none) | ~9.0 MB | 600–800 ms | baseline |
| float16 | ~4.5 MB | 400–600 ms | ~baseline |
| **int8 (rep-data)** | **~2.5 MB** | **150–250 ms** | -1 to -3% top-1 |
| int8 (no rep-data) | ~2.5 MB | 150–250 ms | -3 to -8% top-1 |

The S3's vector instructions give a 4–8× speedup on int8 ops, which is why
**int8 with representative data is the recommended default**.

## Troubleshooting

**"Could not find a Teachable Machine model"**
Make sure your `--input` folder contains either `keras_model.h5` *or*
`saved_model.pb` (+ `variables/`). The TM web export gives you a folder
with both, plus a `labels.txt`.

**"You need to provide either a dictionary..." during calibration**
Fixed in this revision. Run `--self-test` to confirm your environment is OK.

**Model > 4 MB after int8 quantization**
You're probably using a model that's too big for ESP32. Switch to
MobileNetV2 with `alpha=0.35` or `0.50` in Teachable Machine's
"Advanced → Export" options, or train on a smaller backbone.

**int8 accuracy drops > 5%**
Always pass `--rep-data` with at least 50 real images per class captured
from your ESP32-CAM. Random-data calibration is a fallback, not the
production path.

## Roadmap

- [ ] `--multilabel N` flag: convert a TM softmax head to N sigmoid binary
      heads (Option A from `docs/ppe-approach.md`)
- [ ] On-device few-shot last-layer trainer for ESP32-S3 (Path A from
      `docs/architecture.md`)
- [ ] Reference `ppe_inference.cc` implementation against TFLite-Micro
- [ ] ESP32-CAM capture script (Arduino + ESP-IDF variants) for the
      sampling procedure in `docs/ppe-sampling-guide.md`

## License

MIT — see `LICENSE`.
