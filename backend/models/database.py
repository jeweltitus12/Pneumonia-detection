import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database.db"
DEFAULT_MODEL_NAME = "MobileNetV2"


def _table_columns(cursor) -> set[str]:
    cursor.execute("PRAGMA table_info(predictions)")
    return {row[1] for row in cursor.fetchall()}


def _ensure_schema(cursor) -> None:
    """Apply non-destructive schema updates for existing databases."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'MobileNetV2',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    columns = _table_columns(cursor)

    if "model_name" not in columns:
        cursor.execute(
            f"ALTER TABLE predictions ADD COLUMN model_name TEXT NOT NULL DEFAULT '{DEFAULT_MODEL_NAME}'"
        )
        columns = _table_columns(cursor)

    # Backfill from the legacy `model` column when present.
    if "model" in columns:
        cursor.execute(
            """
            UPDATE predictions
            SET model_name = COALESCE(NULLIF(model, ''), model_name, ?)
            WHERE model_name IS NULL OR model_name = '' OR model_name = ?
            """,
            (DEFAULT_MODEL_NAME, DEFAULT_MODEL_NAME),
        )


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _ensure_schema(cursor)
    conn.commit()
    conn.close()


def save_prediction(filename, prediction, confidence, model_name=DEFAULT_MODEL_NAME, model=None):
    """
    Persist a prediction record.

    `model` is accepted only for backward compatibility with older callers.
    """
    if model is not None and model_name == DEFAULT_MODEL_NAME:
        model_name = model

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _ensure_schema(cursor)

    columns = _table_columns(cursor)
    if "model" in columns:
        cursor.execute(
            """
            INSERT INTO predictions (filename, prediction, confidence, model_name, model, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, prediction, confidence, model_name, model_name, datetime.now()),
        )
    else:
        cursor.execute(
            """
            INSERT INTO predictions (filename, prediction, confidence, model_name, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, prediction, confidence, model_name, datetime.now()),
        )

    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id


def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _ensure_schema(cursor)
    conn.commit()

    columns = _table_columns(cursor)
    if "model_name" in columns:
        model_expr = "COALESCE(NULLIF(model_name, ''), ?)"
        model_params = [DEFAULT_MODEL_NAME]
    elif "model" in columns:
        model_expr = "COALESCE(NULLIF(model, ''), ?)"
        model_params = [DEFAULT_MODEL_NAME]
    else:
        model_expr = "?"
        model_params = [DEFAULT_MODEL_NAME]

    cursor.execute(
        f"""
        SELECT id, filename, prediction, confidence, {model_expr} AS model_name, timestamp
        FROM predictions
        ORDER BY timestamp DESC
        """,
        model_params,
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "model_name": row[4] or DEFAULT_MODEL_NAME,
            "timestamp": row[5],
        }
        for row in rows
    ]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Pneumonia'")
    pneumonia = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'")
    normal = cursor.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "pneumonia": pneumonia,
        "normal": normal,
    }
