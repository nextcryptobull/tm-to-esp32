# Architecture: ESP32-CAM + ESP32-S3 + xiaozhi for Vision-Grounded Voice AI

This document captures the architectural reasoning behind the project. It
answers: *how do you combine a Google Teachable Machine vision model with
the xiaozhi-esp32 voice stack, given ESP32 hardware constraints?*

## Hardware reality check

| | ESP32-CAM (classic ESP32) | ESP32-S3 (xiaozhi board) |
|---|---|---|
| CPU | Dual-core 240 MHz (no vector ext.) | Dual-core 240 MHz **+ vector instructions for NN** |
| RAM | 520 KB SRAM (+4 MB PSRAM typical) | 512 KB SRAM + PSRAM (often 8 MB) |
| ML support | TFLite-Micro works but slow | First-class TFLite-Micro + ESP-DL support |
| Camera | Onboard OV2640 | External via DVP/USB |
| Role in your project | **Sample capture + lightweight inference** | **Voice + LLM gateway + heavier inference** |

**Key insight**: the ESP32-S3 is the right brain, the ESP32-CAM is the eye.
Don't try to make the CAM do heavy work.

## Two training-path options compared

### Path A — On-device training (Teachable-Machine-like on ESP32)

Training (backprop, gradient updates) on 4 MB of PSRAM is impractical and
will take hours per epoch for any real image dataset. What *can* be done
on-device is **few-shot last-layer adaptation** — closest thing to
"Teachable Machine on ESP32" that actually works.

The ESP32-S3 has enough RAM to hold a frozen MobileNetV2/ESP-DL backbone
(~1–2 MB quantized) and update a small classification head via stochastic
gradient descent on a handful of new samples. This is the same trick Edge
Impulse uses for "learning on the edge."

**Status**: experimental, not implemented in this repo. Roadmap item.

### Path B — Train in Teachable Machine, adapt for ESP32 (this repo)

Teachable Machine exports a Keras `.h5` or SavedModel. The `tm_to_esp32.py`
script in this repo performs the conversion pipeline:

```
Teachable Machine (web) → Keras model
        ↓
TensorFlow Model Optimization Toolkit (post-training quantization → int8)
        ↓
TFLite Converter → .tflite (often 200 KB – 2 MB)
        ↓
ESP-DL / TFLite-Micro interpreter on ESP32-S3
        ↓
ESP32-CAM captures frames → UART/SPI to S3 → inference → result fed to xiaozhi LLM prompt
```

No Raspberry Pi, no Arduino required. The only "middleman" is the
conversion toolchain, which runs once on your laptop.

## Recommended runtime architecture

```
┌──────────────┐    frames over UART    ┌──────────────────────┐
│  ESP32-CAM   │ ────────────────────► │   ESP32-S3 (xiaozhi) │
│  OV2640 only │                       │  ───────────────────  │
│  capture +   │                       │  • TFLite-Micro int8  │
│  basic preproc│                      │    vision model(s)    │
└──────────────┘                       │  • ESP-DL (optional) │
                                       │  • Wake word + VAD    │
                                       │  • WiFi → cloud LLM  │
                                       │    (Qwen/GLM/etc.)    │
                                       └──────────┬────────────┘
                                                  │
                                          inferred label +
                                          transcript → prompt
```

**Why this split**:
- The CAM module's job is just `capture → JPEG/RGB → send`. Cheap,
  low-power, no model.
- All inference (vision + wake word + speech) lives on the S3 because it
  has the vector extensions and ESP-DL support.
- xiaozhi already handles the cloud LLM round-trip; you just inject the
  vision model's predicted class into the conversation context.

## Why this beats alternatives

- **No Pi dependency**: the ESP-IDF toolchain on your laptop is all you
  need for flashing. Arduino IDE is fine for prototyping but limits you
  when integrating with xiaozhi's ESP-IDF-based build.
- **No cloud round-trip for vision**: vision inference happens on the S3,
  only the predicted label (a few bytes) goes into the LLM prompt.
  Privacy + latency + bandwidth all improve.
- **Model size budget on ESP32-S3 with xiaozhi**:
  - xiaozhi firmware + FreeRTOS: ~1.5 MB flash
  - Wake word model (e.g., Hi-ESP-Skittles): ~150 KB
  - Your vision model: keep under 2 MB quantized
  - PSRAM is plenty for activations

## Bottom line

- **For shipping v1**: Train in Teachable Machine (or Edge Impulse) →
  quantize → convert to TFLite → flash to ESP32-S3. ESP32-CAM is only the
  camera. No Pi, no Arduino required.
- **For the "Teachable Machine on ESP32" dream v2**: Implement last-layer
  few-shot learning on the S3 — feasible, but only for the final
  classifier, not full network training. Roadmap.
