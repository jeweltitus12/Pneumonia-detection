"""
Export a trained Keras .h5 model to a lightweight .tflite file for production deployment.

Run locally (requires tensorflow from requirements.txt):
    python scripts/export_tflite.py
    python scripts/export_tflite.py --model MobileNetV2

Commit backend/weights/pneumonia_model.tflite to GitHub for Render deploys.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_builder import KERAS_CUSTOM_OBJECTS, normalize_model_name
from services.model_registry import get_registry_entry, get_weights_path


def export_model(model_name: str = "MobileNetV2") -> None:
    name = normalize_model_name(model_name)
    entry = get_registry_entry(name)
    h5_path = get_weights_path(name)
    tflite_filename = entry.get("tflite_filename")

    if not h5_path.exists():
        print(f"Missing Keras model: {h5_path}")
        print("Run `python scripts/bootstrap_model.py` or `python scripts/train_model.py` first.")
        sys.exit(1)

    if not tflite_filename:
        print(f"{name} does not define a TFLite export path. Saved weights: {h5_path}")
        sys.exit(0)

    import tensorflow as tf

    tflite_path = h5_path.parent / tflite_filename
    model = tf.keras.models.load_model(str(h5_path), custom_objects=KERAS_CUSTOM_OBJECTS)

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"Exported {tflite_path} ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained Keras model to TFLite")
    parser.add_argument(
        "--model",
        default="MobileNetV2",
        help="Model architecture to export (only MobileNetV2 has a TFLite target)",
    )
    args = parser.parse_args()
    export_model(args.model)


if __name__ == "__main__":
    main()
