from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.ai_model import predict_image, get_model_status
from models.database import save_prediction, get_history, get_stats
from pathlib import Path
import os
import uuid

api_blueprint = Blueprint("api", __name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed file types are png, jpg, jpeg"}), 400

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = UPLOADS_DIR / unique_name
    file.save(filepath)

    try:
        prediction, confidence = predict_image(str(filepath))

        save_prediction(original_name, prediction, confidence)

        return jsonify(
            {
                "message": "Prediction successful",
                "prediction": prediction,
                "confidence": confidence,
                "filename": original_name,
            }
        ), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
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
