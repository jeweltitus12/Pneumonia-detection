import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "weights" / "pneumonia_model.tflite"
_env_model_path = os.environ.get("MODEL_PATH")
if _env_model_path:
    _path = Path(_env_model_path)
    MODEL_PATH = _path if _path.is_absolute() else BASE_DIR / _path
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

IMAGE_SIZE = (224, 224)

_interpreter = None
_model_load_error = None


def _get_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter  # local dev fallback

    return Interpreter


def _preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a chest X-ray for model inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def load_model(force_reload: bool = False):
    """Load the TFLite model once and cache it in memory."""
    global _interpreter, _model_load_error

    if _interpreter is not None and not force_reload:
        return _interpreter

    if not MODEL_PATH.exists():
        _model_load_error = (
            f"Model file not found at {MODEL_PATH}. "
            "Run `python scripts/export_tflite.py` locally, then commit the .tflite file."
        )
        raise FileNotFoundError(_model_load_error)

    try:
        Interpreter = _get_interpreter()
        _interpreter = Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
        _model_load_error = None
        logger.info("Loaded pneumonia model from %s", MODEL_PATH)
        return _interpreter
    except Exception as exc:
        _model_load_error = f"Failed to load model: {exc}"
        raise RuntimeError(_model_load_error) from exc


def get_model_status() -> dict:
    """Return model availability for health checks."""
    if _interpreter is not None:
        return {"loaded": True, "path": str(MODEL_PATH), "error": None}

    if MODEL_PATH.exists():
        return {"loaded": False, "path": str(MODEL_PATH), "error": _model_load_error}

    return {
        "loaded": False,
        "path": str(MODEL_PATH),
        "error": _model_load_error or "Model file missing",
    }


def predict_image(image_path: str) -> tuple[str, float]:
    """
    Run pneumonia detection on a chest X-ray image.

    Returns:
        Tuple of (prediction label, confidence percentage).
    """
    interpreter = load_model()
    batch = _preprocess_image(image_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    raw_prediction = interpreter.get_tensor(output_details[0]["index"])

    if raw_prediction.ndim == 2 and raw_prediction.shape[1] > 1:
        pneumonia_score = float(raw_prediction[0][1])
    else:
        pneumonia_score = float(np.squeeze(raw_prediction))

    pneumonia_score = float(np.clip(pneumonia_score, 0.0, 1.0))

    if pneumonia_score >= 0.5:
        return "Pneumonia", round(pneumonia_score * 100, 2)

    return "Normal", round((1.0 - pneumonia_score) * 100, 2)
