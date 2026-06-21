from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from models.database import init_db
from routes.api import api_blueprint
from services.ai_model import load_model, get_model_status
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

init_db()
app.register_blueprint(api_blueprint, url_prefix="/api")


@app.route("/")
def index():
    status = get_model_status()
    return jsonify(
        {
            "message": "Pneumonia Detection AI API is running.",
            "model_loaded": status["loaded"],
            "model_path": status["path"],
        }
    )


def _warmup_model() -> None:
    try:
        load_model()
        print(f"Model loaded from {get_model_status()['path']}")
    except Exception as exc:
        print(f"Warning: model not loaded on startup ({exc})")


_warmup_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, port=port, host="0.0.0.0")
