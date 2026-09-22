from __future__ import annotations

import math
from typing import Any

import pandas as pd


# Demo/Prototype thresholds only. Production deployment should replace these with
# region- and crop-specific ranges validated by qualified agronomists and soil labs.
NITROGEN_THRESHOLDS = {"low": 35.0, "adequate_min": 35.0, "adequate_max": 65.0, "high": 65.0}
PHOSPHORUS_THRESHOLDS = {"low": 22.0, "adequate_min": 22.0, "adequate_max": 42.0, "high": 42.0}
POTASSIUM_THRESHOLDS = {"low": 30.0, "adequate_min": 30.0, "adequate_max": 55.0, "high": 55.0}
PH_THRESHOLDS = {"acidic_max": 6.0, "suitable_min": 6.0, "suitable_max": 7.5, "alkaline_min": 7.5}
ORGANIC_CARBON_THRESHOLDS = {"low": 1.2, "adequate_min": 1.2, "adequate_max": 2.5, "high": 2.5}

NUTRIENT_INPUTS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "pH": "pH",
    "Organic_Carbon": "Organic Carbon",
}


def validate_nutrient_input(values: dict[str, Any] | pd.DataFrame) -> dict[str, float]:
    """Validate one nutrient profile and return normalized numeric values."""
    if isinstance(values, pd.DataFrame):
        if values.empty:
            raise ValueError("Nutrient input is empty.")
        values = values.iloc[0].to_dict()
    missing = [column for column in NUTRIENT_INPUTS if column not in values]
    if missing:
        raise ValueError(f"Missing required nutrient values: {', '.join(missing)}")

    normalized: dict[str, float] = {}
    for column in NUTRIENT_INPUTS:
        try:
            value = float(values[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{NUTRIENT_INPUTS[column]} must be numeric.") from error
        if not math.isfinite(value):
            raise ValueError(f"{NUTRIENT_INPUTS[column]} must be a finite number.")
        if column != "pH" and value < 0:
            raise ValueError(f"{NUTRIENT_INPUTS[column]} cannot be negative.")
        if column == "pH" and not 0 <= value <= 14:
            raise ValueError("pH must be between 0 and 14.")
        normalized[column] = value
    return normalized


def _status_result(nutrient: str, value: float, status: str, message: str) -> dict[str, Any]:
    return {"nutrient": nutrient, "value": value, "status": status, "message": message}


def _classify_range(value: float, thresholds: dict[str, float], nutrient: str, unit: str = "") -> dict[str, Any]:
    if value < thresholds["adequate_min"]:
        return _status_result(nutrient, value, "Low", f"Below the demo/prototype adequate range{unit}.")
    if value > thresholds["adequate_max"]:
        return _status_result(nutrient, value, "High", f"Above the demo/prototype adequate range{unit}.")
    return _status_result(nutrient, value, "Adequate", f"Within the demo/prototype adequate range{unit}.")


def analyze_nitrogen(value: float) -> dict[str, Any]:
    return _classify_range(float(value), NITROGEN_THRESHOLDS, "Nitrogen", " for nitrogen")


def analyze_phosphorus(value: float) -> dict[str, Any]:
    return _classify_range(float(value), PHOSPHORUS_THRESHOLDS, "Phosphorus", " for phosphorus")


def analyze_potassium(value: float) -> dict[str, Any]:
    return _classify_range(float(value), POTASSIUM_THRESHOLDS, "Potassium", " for potassium")


def analyze_ph(value: float) -> dict[str, Any]:
    numeric_value = float(value)
    thresholds = PH_THRESHOLDS
    if numeric_value < thresholds["acidic_max"]:
        return _status_result("pH", numeric_value, "Acidic", "Below the demo/prototype suitable pH range.")
    if numeric_value > thresholds["suitable_max"]:
        return _status_result("pH", numeric_value, "Alkaline", "Above the demo/prototype suitable pH range.")
    return _status_result("pH", numeric_value, "Suitable", "Within the demo/prototype suitable pH range.")


def analyze_organic_carbon(value: float) -> dict[str, Any]:
    return _classify_range(float(value), ORGANIC_CARBON_THRESHOLDS, "Organic Carbon", " for organic carbon")


def analyze_all_nutrients(values: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    normalized = validate_nutrient_input(values)
    analyses = [
        analyze_nitrogen(normalized["N"]),
        analyze_phosphorus(normalized["P"]),
        analyze_potassium(normalized["K"]),
        analyze_ph(normalized["pH"]),
        analyze_organic_carbon(normalized["Organic_Carbon"]),
    ]
    return {
        "values": normalized,
        "nutrients": analyses,
        "npk_total": normalized["N"] + normalized["P"] + normalized["K"],
        "summary": calculate_nutrient_summary(analyses),
    }


def analyze_npk(values: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    normalized = validate_nutrient_input(values)
    nutrient_results = [
        analyze_nitrogen(normalized["N"]),
        analyze_phosphorus(normalized["P"]),
        analyze_potassium(normalized["K"]),
    ]
    low_nutrients = [result["nutrient"] for result in nutrient_results if result["status"] == "Low"]
    if len(low_nutrients) > 1:
        pattern = "Multiple nutrient deficiencies"
    elif low_nutrients:
        pattern = f"{low_nutrients[0]} relatively low"
    else:
        pattern = "No major deficiency detected"
    return {"npk_total": normalized["N"] + normalized["P"] + normalized["K"], "pattern": pattern, "nutrients": nutrient_results}


def calculate_nutrient_summary(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    nutrient_results = [result for result in analyses if result["nutrient"] != "pH"]
    low_count = sum(result["status"] == "Low" for result in nutrient_results)
    adequate_count = sum(result["status"] in {"Adequate", "Suitable"} for result in analyses)
    high_count = sum(result["status"] in {"High", "Alkaline"} for result in analyses)
    ph_result = next((result for result in analyses if result["nutrient"] == "pH"), None)
    organic_carbon_result = next((result for result in analyses if result["nutrient"] == "Organic Carbon"), None)
    if low_count == 0 and ph_result and ph_result["status"] == "Suitable":
        overall_condition = "No major deficiency detected"
    elif low_count >= 2:
        overall_condition = "Multiple nutrient deficiencies flagged"
    else:
        overall_condition = "Some nutrient attention flagged"
    return {
        "total_nutrients_analyzed": len(analyses),
        "low_nutrients": low_count,
        "adequate_nutrients": adequate_count,
        "high_nutrients": high_count,
        "pH_status": ph_result["status"] if ph_result else "Unavailable",
        "organic_carbon_status": organic_carbon_result["status"] if organic_carbon_result else "Unavailable",
        "overall_condition": overall_condition,
    }
