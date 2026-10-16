#!/usr/bin/env python3
"""
tm_to_esp32.py
==============

Convert a Google Teachable Machine export into an ESP32-S3 / ESP32-CAM
deployable package for use with the xiaozhi-esp32 firmware stack.

Pipeline
---------
  Teachable Machine export folder
        (keras_model.h5  OR  saved_model.pb + variables/)
        + labels.txt
            |
            v
  [1] Load Keras / SavedModel into TF
  [2] (optional) Build representative dataset from sample images folder
  [3] Quantize:
        --quantize int8      -> full integer int8 (best for ESP32-S3 vector ext.)
        --quantize float16   -> float16 (smaller, no representative data needed)
        --quantize none      -> no quantization (debug only, not recommended)
  [4] Convert to TFLite, write .tflite
  [5] (optional) Generate C header files:
        model_data.h / model_data.cc   (model as byte array, xxd -i style)
        labels.h                       (labels as char* array)
  [6] Write model_info.json (input shape, classes, quant, sizes)
  [7] Print human-readable summary

Usage
-----
  # Most common case: int8 quantization, no representative data (uses random data)
  python tm_to_esp32.py \\
      --input  ./my_tm_export \\
      --output ./out \\
      --quantize int8 \\
      --c-header

  # Better int8 quant: use real ESP32-CAM sample images as representative data
  python tm_to_esp32.py \\
      --input  ./my_tm_export \\
      --output ./out \\
      --quantize int8 \\
      --rep-data ./captured_samples \\
      --c-header

  # Float16 fallback (no representative data needed, slightly less accurate on S3)
  python tm_to_esp32.py --input ./my_tm_export --output ./out --quantize float16

  # Run built-in self-test (creates tiny synthetic model, runs whole pipeline)
  python tm_to_esp32.py --self-test

Author: GLM assistant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Generator, Iterable

# TensorFlow logs are noisy; suppress INFO.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

keras.utils.disable_interactive_logging()


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def log(msg: str, prefix: str = "  ") -> None:
    print(f"{prefix}{msg}")


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n {title}\n{bar}")


# --------------------------------------------------------------------------- #
# Step 1: Load TM model (auto-detect format)
# --------------------------------------------------------------------------- #
def detect_format(input_dir: Path) -> Tuple[str, Optional[Path]]:
    """
    Detect whether the TM export is a Keras .h5 or a SavedModel directory.

    Returns: (format_name, model_path)
    """
    h5 = input_dir / "keras_model.h5"
    pb = input_dir / "saved_model.pb"
    # Some TM exports put saved model under saved_model/ subdir
    pb_sub = input_dir / "saved_model" / "saved_model.pb"

    if h5.exists():
        return ("keras_h5", h5)
    if pb.exists():
        return ("saved_model", input_dir)
    if pb_sub.exists():
        return ("saved_model", input_dir / "saved_model")
    # Fallback: any .h5 in dir
    h5_files = list(input_dir.glob("*.h5"))
    if h5_files:
        return ("keras_h5", h5_files[0])
    # Fallback: any folder with saved_model.pb inside
    for sub in input_dir.iterdir():
        if sub.is_dir() and (sub / "saved_model.pb").exists():
            return ("saved_model", sub)
    raise FileNotFoundError(
        f"Could not find a Teachable Machine model in {input_dir}. "
        f"Expected one of: keras_model.h5, saved_model.pb, or <sub>/saved_model.pb"
    )


def load_model(input_dir: Path) -> Tuple[tf.Module, str]:
    fmt, path = detect_format(input_dir)
    log(f"Detected format: {fmt}")
    log(f"Model path: {path}")
    if fmt == "keras_h5":
        model = keras.models.load_model(path, compile=False)
    else:
        model = tf.saved_model.load(str(path))
    return model, fmt


def load_labels(input_dir: Path) -> List[str]:
    """Load labels.txt. Returns ['class_a', 'class_b', ...]."""
    candidates = [input_dir / "labels.txt"]
    # Some TM exports nest it
    for sub in input_dir.iterdir():
        if (sub / "labels.txt").exists():
            candidates.append(sub / "labels.txt")
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            log(f"Loaded {len(labels)} labels from {c}")
            return labels
    log("WARNING: no labels.txt found, using placeholder labels", "  ! ")
    return ["class_0", "class_1"]


def get_input_shape(model) -> Tuple[int, ...]:
    """Try to extract the model's input tensor shape."""
    # Keras Model
    if hasattr(model, "input_shape"):
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        return tuple(int(x) if x else 1 for x in shape)
    if hasattr(model, "inputs") and model.inputs:
        shape = model.inputs[0].shape
        return tuple(int(x) if x else 1 for x in shape)
    # SavedModel (signatures)
    if hasattr(model, "signatures"):
        sig = list(model.signatures.values())[0]
        spec = list(sig.structured_input_signature[1].values())[0]
        shape = spec.shape
        return tuple(int(x) if (hasattr(x, "value") and x.value is not None) or (isinstance(x, int)) else 1
                     for x in shape)
    raise RuntimeError("Cannot determine model input shape")


# --------------------------------------------------------------------------- #
# Step 2: Build representative dataset
# --------------------------------------------------------------------------- #
def iter_image_files(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def make_representative_data_generator(
    rep_data_dir: Optional[Path],
    input_shape: Tuple[int, ...],
    num_samples: int = 100,
) -> Generator[np.ndarray, None, None]:
    """
    Yield batches of input tensors matching the model's input shape.

    If rep_data_dir is provided, load and preprocess real images from it.
    Otherwise, generate uniformly distributed random data of the same shape.

    Each yielded item must be a numpy array of shape == (batch, *input_shape).
    """
    # Drop batch dim (index 0) for per-sample shape
    sample_shape = input_shape[1:]  # e.g. (224, 224, 3)
    hwc = sample_shape  # treat as HWC
    is_chw = False
    if len(sample_shape) == 3 and sample_shape[0] in (1, 3) and sample_shape[1] > 4 and sample_shape[2] > 4:
        # Could be CHW; flip to HWC for image loading
        is_chw = True
        hwc = (sample_shape[1], sample_shape[2], sample_shape[0])

    if rep_data_dir is not None and rep_data_dir.exists():
        log(f"Using representative dataset from {rep_data_dir}")
        files = list(iter_image_files(rep_data_dir))
        if not files:
            log("WARNING: rep_data_dir has no images, falling back to random data", "  ! ")
            rep_data_dir = None
        else:
            log(f"Found {len(files)} candidate images")
            for i, img_path in enumerate(files):
                if i >= num_samples:
                    break
                try:
                    img = Image.open(img_path).convert("RGB").resize((hwc[1], hwc[0]))
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    if is_chw:
                        arr = np.transpose(arr, (2, 0, 1))
                    # Yield a LIST of tensors (one per input), each with batch dim.
                    yield [np.expand_dims(arr, axis=0)]
                except Exception as e:
                    log(f"skip {img_path.name}: {e}", "  ! ")
                    continue

    if rep_data_dir is None:
        log(f"No representative dataset; using {num_samples} random uniform samples")
        for _ in range(num_samples):
            yield [np.random.uniform(low=0.0, high=1.0, size=(1, *sample_shape)).astype(np.float32)]


# --------------------------------------------------------------------------- #
# Step 3: TFLite conversion + quantization
# --------------------------------------------------------------------------- #
def convert_to_tflite(
    model,
    fmt: str,
    quantize: str,
    rep_data_dir: Optional[Path],
    input_shape: Tuple[int, ...],
    supported_ops: Optional[set] = None,
) -> bytes:
    """
    Convert the loaded model to a TFLite flatbuffer, applying the requested
    quantization. Returns raw bytes.

    quantize: 'int8' | 'float16' | 'none'
    """
    log(f"Converting -> TFLite (quantize={quantize})")

    if fmt == "keras_h5":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(getattr(model, "_saved_model_path", "")) or "")
        # Fallback path below uses signature, but from_saved_model is robust
        # If from_saved_model fails because we used tf.saved_model.load above,
        # we need to re-discover the path.
        if not getattr(converter, "_saved_model_dir", None):
            # The model loaded via tf.saved_model.load doesn't expose path;
            # we instead rely on caller having the path. Workaround: use signature.
            pass

    # Input/output specs
    if quantize in ("int8",):
        # Full int8 quantization requires representative data + int8 in/out
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if supported_ops:
            converter.target_spec.supported_ops = list(supported_ops)
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda: make_representative_data_generator(
            rep_data_dir, input_shape, num_samples=100
        )
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    elif quantize == "none":
        # No optimization at all
        converter.optimizations = []
    else:
        raise ValueError(f"Unknown quantize mode: {quantize}")

    t0 = time.time()
    tflite_bytes = converter.convert()
    log(f"Conversion done in {time.time() - t0:.1f}s, output size = {len(tflite_bytes):,} bytes")
    return tflite_bytes


def convert_saved_model_path(input_dir: Path, model, fmt: str) -> Optional[str]:
    """
    If the model was loaded via tf.saved_model.load, we need the path on disk
    to pass to TFLiteConverter.from_saved_model. The path was discovered by
    detect_format() earlier — pass it back in via attribute injection so the
    converter can find it.
    """
    if fmt != "saved_model":
        return None
    # detect_format already found a dir with saved_model.pb. We re-detect here.
    pb = input_dir / "saved_model.pb"
    pb_sub = input_dir / "saved_model" / "saved_model.pb"
    if pb.exists():
        return str(input_dir)
    if pb_sub.exists():
        return str(input_dir / "saved_model")
    return None


# --------------------------------------------------------------------------- #
# Step 4: Generate C header files (xxd -i style)
# --------------------------------------------------------------------------- #
def c_var_name(filename: str) -> str:
    """Mimic xxd -i: replace non-alphanumeric with underscore."""
    out = []
    for ch in filename:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def write_c_header(
    tflite_bytes: bytes,
    labels: List[str],
    out_dir: Path,
    header_name: str = "model_data",
    input_shape: Optional[Tuple[int, ...]] = None,
    quantize: str = "int8",
) -> Tuple[Path, Path, Path]:
    """
    Write three files in out_dir:
      <header_name>.h    - declarations + extern
      <header_name>.cc   - byte array definition
      labels.h           - labels string array
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    var = c_var_name(header_name) + "_tflite"
    len_var = c_var_name(header_name) + "_tflite_len"

    # .h declaration
    h_path = out_dir / f"{header_name}.h"
    guard = f"_{c_var_name(header_name).upper()}_H_"
    h_lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        f"// Auto-generated by tm_to_esp32.py",
        f"// Quantization: {quantize}",
        f"// Input shape: {input_shape}",
        f"// Labels: {len(labels)}",
        "",
        f"extern const unsigned char {var}[];",
        f"extern const unsigned int  {len_var};",
        "",
        "#endif",
        "",
    ]
    h_path.write_text("\n".join(h_lines), encoding="utf-8")

    # .cc definition (byte array, 12 bytes per line)
    cc_path = out_dir / f"{header_name}.cc"
    lines = [
        f'// Auto-generated from TFLite model. Quantization = {quantize}',
        f'#include "{header_name}.h"',
        "",
        f"const unsigned char {var}[] = {{",
    ]
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_vals},")
    # Remove trailing comma on last line
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines += [
        "};",
        f"const unsigned int {len_var} = sizeof({var}) / sizeof({var}[0]);",
        "",
    ]
    cc_path.write_text("\n".join(lines), encoding="utf-8")

    # labels.h
    labels_path = out_dir / "labels.h"
    lbl_lines = [
        "#ifndef _LABELS_H_",
        "#define _LABELS_H_",
        "",
        "static const char* const kLabels[] = {",
    ]
    for lbl in labels:
        # Escape quotes/backslashes
        esc = lbl.replace("\\", "\\\\").replace('"', '\\"')
        lbl_lines.append(f'  "{esc}",')
    lbl_lines += [
        "};",
        f"static const int kLabelCount = {len(labels)};",
        "",
        "#endif",
        "",
    ]
    labels_path.write_text("\n".join(lbl_lines), encoding="utf-8")

    return h_path, cc_path, labels_path


# --------------------------------------------------------------------------- #
# Step 5: Inspect the .tflite for ops + sizes (sanity check)
# --------------------------------------------------------------------------- #
def inspect_tflite(tflite_bytes: bytes) -> dict:
    """Return metadata about the .tflite file."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    # Convert numpy ints to plain Python ints for JSON safety
    def _py(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return x
    return {
        "input_dtype": str(np.dtype(in_det["dtype"])) if hasattr(in_det["dtype"], "itemsize") else str(in_det["dtype"]),
        "input_shape": [_py(x) for x in in_det["shape"]],
        "input_name": str(in_det.get("name", "")),
        "output_dtype": str(np.dtype(out_det["dtype"])) if hasattr(out_det["dtype"], "itemsize") else str(out_det["dtype"]),
        "output_shape": [_py(x) for x in out_det["shape"]],
        "output_name": str(out_det.get("name", "")),
        "size_bytes": int(len(tflite_bytes)),
        "size_kib": round(float(len(tflite_bytes)) / 1024.0, 2),
        "size_mib": round(float(len(tflite_bytes)) / (1024.0 * 1024.0), 3),
    }


# --------------------------------------------------------------------------- #
# Self-test: build a tiny Keras model and run the whole pipeline
# --------------------------------------------------------------------------- #
def build_synthetic_teachable_machine_export(out_dir: Path) -> Path:
    """
    Build a tiny MobileNetV2-like Keras model (3 classes, 224x224 input),
    save it in the same layout as a Teachable Machine export:
        keras_model.h5
        labels.txt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Building synthetic TM-style export in {out_dir}")

    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        pooling="avg",
    )
    x = keras.layers.Dense(3, activation="softmax")(base.output)
    model = keras.Model(base.input, x)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    h5_path = out_dir / "keras_model.h5"
    model.save(h5_path)
    (out_dir / "labels.txt").write_text("cat\ndog\nbird\n", encoding="utf-8")
    log(f"Saved {h5_path} ({h5_path.stat().st_size:,} bytes)")
    return out_dir


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_conversion(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    section(f"Step 1/5  Loading Teachable Machine export")
    log(f"Input:  {input_dir}")
    log(f"Output: {out_dir}")
    model, fmt = load_model(input_dir)
    labels = load_labels(input_dir)
    input_shape = get_input_shape(model)
    log(f"Input shape: {input_shape}")
    log(f"Output classes: {len(labels)}")

    section(f"Step 2/5  Quantizing and converting to TFLite ({args.quantize})")
    rep_dir = Path(args.rep_data).resolve() if args.rep_data else None
    if rep_dir:
        log(f"Representative dataset: {rep_dir}")

    # For SavedModel, we need to give TFLiteConverter the on-disk path
    if fmt == "saved_model":
        sm_path = convert_saved_model_path(input_dir, model, fmt)
        # Reload the model via from_saved_model path by passing the path
        # to TFLiteConverter inside convert_to_tflite. We monkey-patch by
        # re-defining the model var as the path string the converter can use.
        # Easiest: just re-create a SavedModel path string the converter accepts.
        # We rely on TFLiteConverter.from_saved_model accepting a path string.
        # Re-do conversion with path-based loader:
        tflite_bytes = _convert_saved_model(
            sm_path,
            quantize=args.quantize,
            rep_data_dir=rep_dir,
            input_shape=input_shape,
            supported_ops=_build_supported_ops(args.target),
        )
    else:
        tflite_bytes = convert_to_tflite(
            model,
            fmt,
            args.quantize,
            rep_dir,
            input_shape,
            supported_ops=_build_supported_ops(args.target),
        )

    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_bytes)
    log(f"Wrote {tflite_path} ({len(tflite_bytes):,} bytes)")

    section("Step 3/5  Inspecting .tflite")
    info = inspect_tflite(tflite_bytes)
    for k, v in info.items():
        log(f"{k:14s} = {v}")

    section("Step 4/5  Generating C/C++ headers")
    if args.c_header:
        h, cc, lbl = write_c_header(
            tflite_bytes,
            labels,
            out_dir,
            header_name=args.header_name,
            input_shape=tuple(info["input_shape"]),
            quantize=args.quantize,
        )
        log(f"Wrote {h}")
        log(f"Wrote {cc}")
        log(f"Wrote {lbl}")
    else:
        log("C header generation skipped (--c-header not set)")
        h = cc = lbl = None
        # Still write labels.txt alongside model for convenience
        (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    section("Step 5/5  Writing model_info.json")
    meta = {
        "source_input": str(input_dir),
        "source_format": fmt,
        "quantization": args.quantize,
        "target": args.target,
        "labels": labels,
        "input_shape": info["input_shape"],
        "input_dtype": info["input_dtype"],
        "output_shape": info["output_shape"],
        "output_dtype": info["output_dtype"],
        "model_size_bytes": info["size_bytes"],
        "model_size_kib": info["size_kib"],
        "model_size_mib": info["size_mib"],
        "tflite_file": str(tflite_path),
        "c_header_h": str(h) if h else None,
        "c_header_cc": str(cc) if cc else None,
        "labels_h": str(lbl) if lbl else None,
        "notes": (
            "For ESP32-S3 deployment with xiaozhi-esp32, copy model.tflite "
            "and labels.txt into your firmware's spiffs/partition, or "
            "embed model_data.cc into the binary if flash is tight."
        ),
    }
    info_path = out_dir / "model_info.json"
    info_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {info_path}")

    section("Summary")
    log(f"Quantization : {args.quantize}")
    log(f"Size on disk  : {info['size_bytes']:,} bytes ({info['size_mib']} MiB)")
    log(f"Input         : {info['input_dtype']} {info['input_shape']}")
    log(f"Output        : {info['output_dtype']} {info['output_shape']}")
    log(f"Classes       : {len(labels)} -> {labels}")
    print("\nDone.\n")
    return meta


def _build_supported_ops(target: str) -> Optional[set]:
    """
    For ESP32-S3 with TFLite-Micro, all-int8 ops are preferred.
    For ESP-DL (Espressif's higher-perf path), the standard TFLite int8
    ops work too — ESP-DL wraps them via pplib. We don't restrict further here.
    """
    if target == "tflite-micro":
        return {tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS}
    if target == "esp-dl":
        return {tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS}
    return None


def _convert_saved_model(
    sm_path: str,
    quantize: str,
    rep_data_dir: Optional[Path],
    input_shape: Tuple[int, ...],
    supported_ops: Optional[set],
) -> bytes:
    """Convert from a SavedModel directory path on disk."""
    log(f"Loading SavedModel from {sm_path} via TFLiteConverter")
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)

    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if supported_ops:
            converter.target_spec.supported_ops = list(supported_ops)
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda: make_representative_data_generator(
            rep_data_dir, input_shape, num_samples=100
        )
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "none":
        converter.optimizations = []
    else:
        raise ValueError(f"Unknown quantize mode: {quantize}")

    return converter.convert()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tm_to_esp32",
        description=(
            "Convert a Google Teachable Machine export into a quantized "
            ".tflite (+ optional C header) package for ESP32-S3 / xiaozhi."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --self-test to verify your environment end-to-end.",
    )
    p.add_argument("--input", "-i", type=str,
                   help="Path to TM export folder (contains keras_model.h5 or saved_model.pb)")
    p.add_argument("--output", "-o", type=str, default="./out",
                   help="Output folder (default: ./out)")
    p.add_argument("--quantize", "-q", choices=["int8", "float16", "none"],
                   default="int8",
                   help="Quantization mode (default: int8, recommended for ESP32-S3)")
    p.add_argument("--rep-data", "-r", type=str, default=None,
                   help="Folder of representative sample images (improves int8 accuracy)")
    p.add_argument("--c-header", action="store_true",
                   help="Also emit model_data.h/.cc + labels.h for static embedding")
    p.add_argument("--header-name", type=str, default="model_data",
                   help="Base name for generated C headers (default: model_data)")
    p.add_argument("--target", choices=["tflite-micro", "esp-dl"], default="tflite-micro",
                   help="Deploy target (default: tflite-micro)")
    p.add_argument("--self-test", action="store_true",
                   help="Build a synthetic TM export, run the whole pipeline, then exit")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    if args.self_test:
        section("SELF-TEST: building synthetic TM export + full pipeline")
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_dir = Path(tmp) / "synthetic_tm"
            build_synthetic_teachable_machine_export(synthetic_dir)
            args.input = str(synthetic_dir)
            args.output = str(Path(tmp) / "out")
            args.quantize = "int8"
            args.c_header = True
            meta = run_conversion(args)
            sz = meta["model_size_bytes"]
            if sz < 100_000:
                raise SystemExit(f"Self-test FAILED: model size {sz} too small")
            if not (Path(args.output) / "model_data.h").exists():
                raise SystemExit("Self-test FAILED: model_data.h not generated")
            print(f"\n[SELF-TEST] OK. Quantized model = {sz:,} bytes. "
                  f"Pipeline verified end-to-end.\n")
        return 0

    if not args.input:
        print("ERROR: --input is required (or use --self-test)", file=sys.stderr)
        return 2
    run_conversion(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
