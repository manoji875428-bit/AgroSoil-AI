from __future__ import annotations

from pathlib import Path
from typing import Final


REQUIRED_FEATURES: Final[list[str]] = ["N", "P", "K", "pH", "Organic_Carbon"]
TARGET_COLUMN: Final[str] = "Fertility"
NUMERIC_FEATURES: Final[list[str]] = REQUIRED_FEATURES.copy()
PLAUSIBLE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "N": (0.0, float("inf")),
    "P": (0.0, float("inf")),
    "K": (0.0, float("inf")),
    "pH": (0.0, 14.0),
    "Organic_Carbon": (0.0, float("inf")),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_raw_path() -> Path:
    return project_root() / "data" / "raw" / "soil_demo_synthetic.csv"


def default_processed_path() -> Path:
    return project_root() / "data" / "processed" / "soil_processed.csv"


def default_summary_path() -> Path:
    return project_root() / "data" / "processed" / "preprocessing_summary.json"
