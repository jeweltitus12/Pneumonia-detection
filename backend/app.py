import logging
import os
import warnings
from pathlib import Path

# Quiet TensorFlow / absl noise before any lazy TF import during model warmup.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from models.database import init_db
from routes.api import api_blueprint
from services.ai_model import warmup_models, get_model_status, get_warmup_model_name

warnings.filterwarnings("ignore", message=".*tf.placeholder.*")
warnings.filterwarnings("ignore", message=".*compile_metrics.*")


class _SuppressDevServerWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "This is a development server" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_SuppressDevServerWarning())

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
        loaded = warmup_models()
        if loaded:
            print(f"Model loaded: {loaded} ({get_model_status()['path']})")
        else:
            print("Warning: no trained model weights found on startup")
    except Exception as exc:
        print(f"Warning: model not loaded on startup ({exc})")


_warmup_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
