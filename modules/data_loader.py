from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_utils import NUMERIC_FEATURES, REQUIRED_FEATURES, TARGET_COLUMN


def load_raw_dataset(path: str | Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV and return a user-facing error instead of raising to the app."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return None, f"Dataset file was not found: {dataset_path}"
    try:
        dataframe = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return None, f"The dataset could not be read: {error}"
    if dataframe.empty:
        return None, "The dataset is empty. Add at least one soil observation before continuing."
    return dataframe, None


def validate_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return validation findings without mutating the input dataframe."""
    missing_required = [column for column in REQUIRED_FEATURES if column not in dataframe.columns]
    numeric_validation: dict[str, dict[str, Any]] = {}
    for column in NUMERIC_FEATURES:
        if column not in dataframe.columns:
            numeric_validation[column] = {"present": False, "invalid_values": 0, "status": "missing"}
            continue
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_values = int(dataframe[column].notna().sum() - converted.notna().sum())
        numeric_validation[column] = {
            "present": True,
            "invalid_values": invalid_values,
            "status": "valid" if invalid_values == 0 else "convertible with warnings",
        }
    return {
        "required_columns": REQUIRED_FEATURES,
        "missing_required_columns": missing_required,
        "has_target": TARGET_COLUMN in dataframe.columns,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "missing_values": int(dataframe.isna().sum().sum()),
        "missing_by_column": {column: int(count) for column, count in dataframe.isna().sum().items() if count},
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numeric_validation": numeric_validation,
        "valid": not missing_required,
    }


def get_dataset_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build compact, JSON-friendly overview values for the UI and metadata."""
    summary = validate_dataset(dataframe)
    summary["columns"] = [str(column) for column in dataframe.columns]
    if TARGET_COLUMN in dataframe.columns:
        summary["target_distribution"] = {
            str(label): int(count) for label, count in dataframe[TARGET_COLUMN].value_counts(dropna=False).items()
        }
    else:
        summary["target_distribution"] = {}
    return summary
