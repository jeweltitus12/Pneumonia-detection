"""
Export the Keras .h5 model to a lightweight .tflite file for production deployment.

Run locally (requires tensorflow from requirements.txt):
    python scripts/export_tflite.py

Commit backend/weights/pneumonia_model.tflite to GitHub for Render deploys.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
H5_PATH = BACKEND_DIR / "weights" / "pneumonia_model.h5"
TFLITE_PATH = BACKEND_DIR / "weights" / "pneumonia_model.tflite"


def main() -> None:
    if not H5_PATH.exists():
        print(f"Missing Keras model: {H5_PATH}")
        print("Run `python scripts/bootstrap_model.py` first.")
        sys.exit(1)

    import tensorflow as tf

    model = tf.keras.models.load_model(str(H5_PATH))

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)

    size_mb = TFLITE_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported {TFLITE_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
