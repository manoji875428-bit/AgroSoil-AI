from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .data_utils import project_root

DATABASE_PATH = project_root() / "data" / "agrosoil.db"


@contextmanager
def _connect(database_path: str | Path = DATABASE_PATH):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_database(database_path: str | Path = DATABASE_PATH) -> Path:
    path = Path(database_path)
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                region TEXT,
                crop TEXT,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                ph REAL,
                organic_carbon REAL,
                predicted_fertility TEXT,
                model_confidence REAL,
                recommendation_summary TEXT,
                source_file TEXT,
                analysis_key TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at ON analysis_history(created_at DESC)")
    return path


def _analysis_key(record: dict[str, Any]) -> str:
    key_fields = {
        field: record.get(field)
        for field in [
            "input_mode", "region", "crop", "nitrogen", "phosphorus", "potassium", "ph",
            "organic_carbon", "predicted_fertility", "model_confidence", "recommendation_summary", "source_file",
        ]
    }
    return hashlib.sha256(json.dumps(key_fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def save_analysis(
    input_mode: str,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    ph: float | None = None,
    organic_carbon: float | None = None,
    predicted_fertility: str | None = None,
    model_confidence: float | None = None,
    recommendation_summary: str | None = None,
    source_file: str | None = None,
    region: str | None = None,
    crop: str | None = None,
    created_at: str | None = None,
    database_path: str | Path = DATABASE_PATH,
) -> dict[str, Any]:
    if not input_mode or not input_mode.strip():
        raise ValueError("Input mode is required.")
    numeric_values = {"nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium, "ph": ph, "organic_carbon": organic_carbon}
    for field, value in numeric_values.items():
        if value is not None:
            try:
                numeric_values[field] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be numeric or empty.") from error
    if numeric_values["ph"] is not None and not 0 <= numeric_values["ph"] <= 14:
        raise ValueError("pH must be between 0 and 14.")
    if any(value is not None and value < 0 for field, value in numeric_values.items() if field != "ph"):
        raise ValueError("Nutrient values cannot be negative.")
    if model_confidence is not None:
        model_confidence = float(model_confidence)
        if not 0 <= model_confidence <= 1:
            raise ValueError("Model confidence must be between 0 and 1.")
    record = {
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_mode": input_mode.strip(),
        "region": region,
        "crop": crop,
        **numeric_values,
        "predicted_fertility": predicted_fertility,
        "model_confidence": model_confidence,
        "recommendation_summary": recommendation_summary,
        "source_file": source_file,
    }
    record["analysis_key"] = _analysis_key(record)
    init_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO analysis_history
            (created_at, input_mode, region, crop, nitrogen, phosphorus, potassium, ph, organic_carbon,
             predicted_fertility, model_confidence, recommendation_summary, source_file, analysis_key)
            VALUES (:created_at, :input_mode, :region, :crop, :nitrogen, :phosphorus, :potassium, :ph,
                    :organic_carbon, :predicted_fertility, :model_confidence, :recommendation_summary,
                    :source_file, :analysis_key)
            """,
            record,
        )
        inserted = cursor.rowcount == 1
        if inserted:
            record["id"] = cursor.lastrowid
    if not inserted:
        existing = get_analysis_history(database_path=database_path, limit=1, analysis_key=record["analysis_key"])
        record = existing[0] if existing else record
    record["inserted"] = inserted
    return record


def get_analysis_history(
    database_path: str | Path = DATABASE_PATH,
    limit: int | None = 100,
    analysis_key: str | None = None,
) -> list[dict[str, Any]]:
    init_database(database_path)
    query = "SELECT * FROM analysis_history"
    parameters: list[Any] = []
    if analysis_key:
        query += " WHERE analysis_key = ?"
        parameters.append(analysis_key)
    query += " ORDER BY datetime(created_at) DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(max(1, int(limit)))
    with _connect(database_path) as connection:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def get_analysis_by_id(analysis_id: int, database_path: str | Path = DATABASE_PATH) -> dict[str, Any] | None:
    init_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute("SELECT * FROM analysis_history WHERE id = ?", (int(analysis_id),)).fetchone()
        return dict(row) if row else None


def delete_analysis(analysis_id: int, database_path: str | Path = DATABASE_PATH) -> bool:
    init_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM analysis_history WHERE id = ?", (int(analysis_id),))
        return cursor.rowcount == 1


def clear_analysis_history(database_path: str | Path = DATABASE_PATH) -> int:
    init_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM analysis_history")
        return cursor.rowcount
