from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.ai_model import (
    predict_image,
    get_model_status,
    is_model_available,
    is_gradcam_available,
    generate_gradcam_explanation,
    list_models,
    get_warmup_model_name,
)
from services.grad_cam import GradCamError
from services.model_registry import AVAILABLE_MODELS, normalize_model_name, get_weights_path
from models.database import save_prediction, get_history, get_stats
from pathlib import Path
import uuid

api_blueprint = Blueprint("api", __name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DEFAULT_MODEL = "MobileNetV2"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _parse_model_name() -> str:
    """Read and validate the requested model from multipart form or JSON body."""
    model_name = None

    if request.is_json and request.json:
        model_name = request.json.get("model")

    if not model_name:
        model_name = request.form.get("model")

    if not model_name:
        model_name = DEFAULT_MODEL

    if not isinstance(model_name, str):
        raise ValueError("Model name must be a string")

    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Model name cannot be empty")

    # Reject path-like values from the client.
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise ValueError("Invalid model name")

    return normalize_model_name(model_name)


@api_blueprint.route("/health", methods=["GET"])
def health():
    status = get_model_status()
    return jsonify(
        {
            "status": "ok",
            "model_loaded": status["loaded"],
            "model_path": status["path"],
            "model_error": status["error"],
        }
    ), 200


def _resolve_default_model() -> str:
    models_list = list_models()
    warmed = get_warmup_model_name()
    if warmed and is_model_available(warmed):
        return warmed
    if is_model_available(DEFAULT_MODEL):
        return DEFAULT_MODEL
    for entry in models_list:
        if entry.get("available"):
            return entry["name"]
    return DEFAULT_MODEL


@api_blueprint.route("/models", methods=["GET"])
def models():
    return jsonify(
        {
            "models": list_models(),
            "default": _resolve_default_model(),
            "supported": list(AVAILABLE_MODELS),
        }
    ), 200


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed file types are png, jpg, jpeg"}), 400

    try:
        model_name = _parse_model_name()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not is_model_available(model_name):
        return jsonify(
            {
                "error": (
                    f"Trained weights for {model_name} are not available. "
                    f"Train with: python scripts/train_model.py --dataset <chest_xray> --model {model_name}"
                )
            }
        ), 503

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = UPLOADS_DIR / unique_name
    file.save(filepath)

    try:
        prediction, confidence = predict_image(str(filepath), model_name=model_name)

        save_prediction(original_name, prediction, confidence, model_name=model_name)

        response = {
            "message": "Prediction successful",
            "prediction": prediction,
            "confidence": confidence,
            "model": model_name,
            "model_name": model_name,
            "filename": original_name,
        }

        if is_gradcam_available(model_name):
            try:
                gradcam = generate_gradcam_explanation(str(filepath), model_name)
                response["gradcam"] = {
                    "heatmap": gradcam["heatmap"],
                    "overlay": gradcam["overlay"],
                    "layer": gradcam["layer"],
                    "backbone_layer": gradcam["backbone_layer"],
                }
            except GradCamError as exc:
                response["gradcam_error"] = str(exc)
            except Exception as exc:
                response["gradcam_error"] = (
                    f"Grad-CAM generation failed for {model_name}: {exc}"
                )
        else:
            if model_name == "MobileNetV2" and not get_weights_path(model_name).exists():
                response["gradcam_error"] = (
                    "Grad-CAM for MobileNetV2 requires the trained Keras weights file "
                    "(pneumonia_model.h5). Only the TFLite deployment artifact is available."
                )
            else:
                response["gradcam_error"] = (
                    f"Grad-CAM is unavailable for {model_name}. "
                    f"Train the model first: python scripts/train_model.py --dataset <chest_xray> --model {model_name}"
                )

        return jsonify(response), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/history", methods=["GET"])
def history():
    try:
        history_data = get_history()
        return jsonify(history_data), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/stats", methods=["GET"])
def stats():
    try:
        stats_data = get_stats()
        return jsonify(stats_data), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
