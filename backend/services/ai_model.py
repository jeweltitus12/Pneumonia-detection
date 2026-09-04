import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from services.model_registry import (
    AVAILABLE_MODELS,
    get_image_size,
    get_weights_path,
    normalize_model_name,
)

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
# Real TFLite exports are much larger than a placeholder/gitkeep stub.
_MIN_TFLITE_BYTES = 100_000


def _tflite_is_usable(path: Path | None = None) -> bool:
    """Return True only when the TFLite file exists and looks like a real model."""
    target = path or MODEL_PATH
    if not target.exists():
        return False
    if target.stat().st_size < _MIN_TFLITE_BYTES:
        return False
    try:
        with target.open("rb") as handle:
            header = handle.read(8)
        # TensorFlow Lite flatbuffer models identify as TFL3 at offset 4.
        return len(header) >= 8 and header[4:8] == b"TFL3"
    except OSError:
        return False


def _keras_weights_exist(model_name: str) -> bool:
    return get_weights_path(model_name).exists()


_interpreter = None
_model_load_error = None
_keras_models: dict[str, object] = {}
_warmup_model_name: str | None = None


def _get_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter  # local dev fallback

    return Interpreter


def _preprocess_image(image_path: str, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess a chest X-ray for model inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def load_model(force_reload: bool = False):
    """Load the TFLite MobileNetV2 model once and cache it in memory."""
    global _interpreter, _model_load_error

    if _interpreter is not None and not force_reload:
        return _interpreter

    if not _tflite_is_usable():
        _model_load_error = (
            f"Valid TFLite model not found at {MODEL_PATH}. "
            "The file may be missing or a placeholder. "
            "Use a trained .h5 model instead, or run `python scripts/export_tflite.py`."
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

    if _keras_models:
        loaded_name, loaded_model = next(iter(_keras_models.items()))
        if not _is_tflite_handle(loaded_model):
            return {
                "loaded": True,
                "path": str(get_weights_path(loaded_name)),
                "error": None,
            }

    if _tflite_is_usable():
        return {"loaded": False, "path": str(MODEL_PATH), "error": _model_load_error}

    available = [name for name in AVAILABLE_MODELS if is_model_available(name)]
    if available:
        return {
            "loaded": False,
            "path": str(get_weights_path(available[0])),
            "error": _model_load_error or "Default TFLite model unavailable; Keras models are ready.",
        }

    return {
        "loaded": False,
        "path": str(MODEL_PATH),
        "error": _model_load_error or "No trained model weights found",
    }


def _load_keras_model(model_name: str):
    import tensorflow as tf
    from services.model_builder import KERAS_CUSTOM_OBJECTS

    path = get_weights_path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained {model_name} weights not found at {path}. "
            f"Train with: python scripts/train_model.py --dataset <chest_xray> --model {model_name}"
        )

    return tf.keras.models.load_model(str(path), custom_objects=KERAS_CUSTOM_OBJECTS)


def get_keras_model(model_name: str):
    """Load and cache the Keras .h5 model used for Grad-CAM explainability."""
    name = normalize_model_name(model_name)
    if not get_weights_path(name).exists():
        raise FileNotFoundError(
            f"Keras weights for {name} were not found at {get_weights_path(name)}. "
            "Grad-CAM requires the trained .h5 model file."
        )

    if name not in _keras_models:
        _keras_models[name] = _load_keras_model(name)
    return _keras_models[name]


def is_gradcam_available(model_name: str) -> bool:
    """Return whether Grad-CAM can be generated for the requested architecture."""
    name = normalize_model_name(model_name)
    if not get_weights_path(name).exists():
        return False

    try:
        _require_tensorflow()
    except Exception:
        return False
    return True


def _require_tensorflow():
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for Grad-CAM explainability. "
            "Install tensorflow locally or use requirements-dev.txt."
        ) from exc


def generate_gradcam_explanation(image_path: str, model_name: str) -> dict:
    """Generate Grad-CAM visualization for the selected Keras model."""
    from services.grad_cam import GradCamError, generate_grad_cam

    name = normalize_model_name(model_name)
    model = get_keras_model(name)
    return generate_grad_cam(image_path, model, name)


def warmup_models() -> str | None:
    """Load the default deployment model, or the first available Keras model."""
    global _warmup_model_name

    if _tflite_is_usable():
        load_model()
        _warmup_model_name = "MobileNetV2"
        return _warmup_model_name

    for name in AVAILABLE_MODELS:
        if _keras_weights_exist(name):
            get_keras_model(name)
            _warmup_model_name = name
            return name

    return None


def get_warmup_model_name() -> str | None:
    return _warmup_model_name


def get_model(model_name: str):
    """
    Return the selected trained model for prediction.

    Supported names:
        MobileNetV2, DenseNet121, ResNet50, EfficientNetB0, VGG16, InceptionV3

    MobileNetV2 uses the existing TFLite file when Keras weights are absent so
    the current production path keeps working.
    """
    name = normalize_model_name(model_name)

    if name == "MobileNetV2":
        if _tflite_is_usable():
            return load_model()
        if _keras_weights_exist(name):
            if name not in _keras_models:
                _keras_models[name] = _load_keras_model(name)
            return _keras_models[name]
        raise FileNotFoundError(
            "No usable MobileNetV2 weights found. Train with "
            "`python scripts/train_model.py --dataset <chest_xray> --model MobileNetV2`."
        )

    if name not in _keras_models:
        _keras_models[name] = _load_keras_model(name)
    return _keras_models[name]


def _score_from_raw(raw_prediction) -> float:
    if getattr(raw_prediction, "ndim", 1) == 2 and raw_prediction.shape[1] > 1:
        pneumonia_score = float(raw_prediction[0][1])
    else:
        pneumonia_score = float(np.squeeze(raw_prediction))
    return float(np.clip(pneumonia_score, 0.0, 1.0))


def _labels_from_score(pneumonia_score: float) -> tuple[str, float]:
    PNEUMONIA_THRESHOLD = float(os.environ.get("PNEUMONIA_THRESHOLD", "0.5"))
    if pneumonia_score >= PNEUMONIA_THRESHOLD:
        return "Pneumonia", round(pneumonia_score * 100, 2)
    return "Normal", round((1.0 - pneumonia_score) * 100, 2)


def _predict_tflite(image_path: str) -> tuple[str, float]:
    interpreter = load_model()
    batch = _preprocess_image(image_path, IMAGE_SIZE)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    raw_prediction = interpreter.get_tensor(output_details[0]["index"])
    return _labels_from_score(_score_from_raw(raw_prediction))


def _is_tflite_handle(model) -> bool:
    return hasattr(model, "get_input_details") and hasattr(model, "invoke")


def is_model_available(model_name: str) -> bool:
    """Return whether trained weights exist for the requested architecture."""
    name = normalize_model_name(model_name)
    if name == "MobileNetV2":
        return _tflite_is_usable() or _keras_weights_exist(name)
    return _keras_weights_exist(name)


def list_models() -> list[dict]:
    """Return supported model names and whether weights are available locally."""
    return [
        {
            "name": name,
            "available": is_model_available(name),
            "gradcam_available": is_gradcam_available(name),
        }
        for name in AVAILABLE_MODELS
    ]


def predict_image(image_path: str, model_name: str | None = None) -> tuple[str, float]:
    """
    Run pneumonia detection on a chest X-ray image.

    Returns:
        Tuple of (prediction label, confidence percentage).

    With no model_name, uses the existing MobileNetV2 TFLite pipeline.
    """
    if model_name is None:
        if _tflite_is_usable():
            return _predict_tflite(image_path)
        if _keras_weights_exist("MobileNetV2"):
            model_name = "MobileNetV2"
        else:
            model_name = next(name for name in AVAILABLE_MODELS if is_model_available(name))

    name = normalize_model_name(model_name)
    if name == "MobileNetV2" and _tflite_is_usable():
        return _predict_tflite(image_path)

    model = get_model(name)
    if _is_tflite_handle(model):
        return _predict_tflite(image_path)

    batch = _preprocess_image(image_path, get_image_size(name))
    raw_prediction = model.predict(batch, verbose=0)
    return _labels_from_score(_score_from_raw(raw_prediction))
