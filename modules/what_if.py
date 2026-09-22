from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .data_utils import REQUIRED_FEATURES
from .model import predict_fertility


def derive_input_ranges(dataframe: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    """Derive broad UI ranges from processed data, with clearly prototype fallbacks."""
    fallback = {"N": (0.0, 200.0), "P": (0.0, 200.0), "K": (0.0, 250.0), "pH": (0.0, 14.0), "Organic_Carbon": (0.0, 20.0)}
    if dataframe is None or dataframe.empty:
        return fallback
    ranges = {}
    for feature in REQUIRED_FEATURES:
        if feature not in dataframe.columns:
            ranges[feature] = fallback[feature]
            continue
        values = pd.to_numeric(dataframe[feature], errors="coerce").dropna()
        if values.empty:
            ranges[feature] = fallback[feature]
            continue
        minimum = float(values.min())
        maximum = float(values.max())
        padding = max((maximum - minimum) * 0.2, 1.0 if feature != "pH" else 0.2)
        lower = max(0.0, minimum - padding)
        upper = min(14.0, maximum + padding) if feature == "pH" else maximum + padding
        ranges[feature] = (round(lower, 2), round(upper, 2))
    return ranges


def validate_simulation_inputs(profile: dict[str, Any], ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    missing = [feature for feature in REQUIRED_FEATURES if feature not in profile or profile[feature] in (None, "")]
    if missing:
        raise ValueError(f"Missing simulation values: {', '.join(missing)}")
    normalized: dict[str, float] = {}
    supported_ranges = ranges or derive_input_ranges()
    for feature in REQUIRED_FEATURES:
        try:
            value = float(profile[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be numeric.") from error
        if not math.isfinite(value):
            raise ValueError(f"{feature} must be finite.")
        if feature != "pH" and value < 0:
            raise ValueError(f"{feature} cannot be negative.")
        if feature == "pH" and not 0 <= value <= 14:
            raise ValueError("pH must be between 0 and 14.")
        lower, upper = supported_ranges[feature]
        if value < lower or value > upper:
            raise ValueError(f"{feature} is outside the supported prototype input range ({lower} to {upper}).")
        normalized[feature] = value
    return normalized


def run_soil_prediction(model: Any, profile: dict[str, Any], ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    validated_profile = validate_simulation_inputs(profile, ranges)
    prediction = predict_fertility(model, validated_profile)
    return {"profile": validated_profile, **prediction}


def compare_predictions(current: dict[str, Any], simulated: dict[str, Any]) -> dict[str, Any]:
    changes = {
        feature: round(simulated["profile"][feature] - current["profile"][feature], 4)
        for feature in REQUIRED_FEATURES
    }
    return {
        "prediction_changed": current["prediction"] != simulated["prediction"],
        "confidence_change": round(simulated["confidence"] - current["confidence"], 4),
        "probability_changes": {
            label: round(simulated["probabilities"][label] - current["probabilities"][label], 4)
            for label in simulated["probabilities"]
        },
        "parameter_changes": changes,
    }


def run_what_if_simulation(model: Any, current_profile: dict[str, Any], simulated_profile: dict[str, Any], ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    current = run_soil_prediction(model, current_profile, ranges)
    simulated = run_soil_prediction(model, simulated_profile, ranges)
    return {"current": current, "simulated": simulated, "comparison": compare_predictions(current, simulated)}
