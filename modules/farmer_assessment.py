from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = [
    "soil_color",
    "moisture_appearance",
    "previous_crop",
    "previous_growth",
    "previous_yield",
    "visible_problems",
    "water_retention",
    "fertilizer_usage",
    "overall_condition",
]

ALLOWED_VALUES = {
    "soil_color": {"Dark brown", "Brown", "Light brown", "Reddish", "Grayish", "Not sure"},
    "moisture_appearance": {"Dry", "Slightly moist", "Moderately moist", "Very wet", "Not sure"},
    "previous_growth": {"Poor", "Average", "Good", "Excellent", "Not sure"},
    "previous_yield": {"Low", "Average", "Good", "Not sure"},
    "water_retention": {"Drains quickly", "Moderate", "Retains water for long", "Not sure"},
    "fertilizer_usage": {"Mostly organic", "Mostly chemical fertilizer", "Both", "None", "Not sure"},
    "overall_condition": {"Poor", "Fair", "Good", "Very good", "Not sure"},
}

VISIBLE_PROBLEMS = {
    "Yellowing leaves",
    "Poor growth",
    "Leaf spots",
    "Wilting",
    "Stunted growth",
    "No visible problem",
    "Not sure",
}


def validate_farmer_inputs(values: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in values or values[field] in (None, "", [])]
    invalid: list[str] = []
    for field, allowed in ALLOWED_VALUES.items():
        if field in values and values[field] not in allowed:
            invalid.append(field)
    problems = values.get("visible_problems", [])
    if not isinstance(problems, list) or not problems:
        if "visible_problems" not in missing:
            missing.append("visible_problems")
    elif any(problem not in VISIBLE_PROBLEMS for problem in problems):
        invalid.append("visible_problems")
    notes = values.get("farmer_notes", "")
    if not isinstance(notes, str):
        invalid.append("farmer_notes")
    return {"valid": not missing and not invalid, "missing": sorted(set(missing)), "invalid": sorted(set(invalid))}


def analyze_farmer_observations(values: dict[str, Any]) -> dict[str, Any]:
    validation = validate_farmer_inputs(values)
    if not validation["valid"]:
        details = validation["missing"] + validation["invalid"]
        raise ValueError(f"Please complete the farmer observations: {', '.join(details)}")

    problems = values["visible_problems"]
    concerns: list[dict[str, str]] = []
    if values["moisture_appearance"] in {"Dry", "Very wet"}:
        concerns.append({"issue": "Possible moisture concern", "reason": f"Observation indicates soil appears {values['moisture_appearance'].lower()}."})
    if values["water_retention"] in {"Drains quickly", "Retains water for long"}:
        concerns.append({"issue": "Possible drainage concern", "reason": f"Observation indicates water {values['water_retention'].lower()}."})
    if values["previous_growth"] in {"Poor", "Average"} or values["previous_yield"] == "Low":
        concerns.append({"issue": "Poor previous crop performance", "reason": "Previous growth or yield observations indicate that further field investigation may be useful."})
    stress_problems = [problem for problem in problems if problem not in {"No visible problem", "Not sure"}]
    if stress_problems:
        concerns.append({"issue": "Visible plant stress", "reason": f"Observed problems include: {', '.join(stress_problems)}."})
    if values["overall_condition"] in {"Poor", "Fair"}:
        concerns.append({"issue": "General soil condition concern", "reason": f"Overall field observation was marked {values['overall_condition'].lower()}."})

    if not concerns:
        overall_observation = "No major observation-based concern was flagged; routine soil testing can still improve confidence."
    elif len(concerns) > 1:
        overall_observation = "Several indicators suggest that further soil testing may be useful."
    else:
        overall_observation = "One observation-based concern was flagged; further soil testing may be useful."
    return {
        "observations": values.copy(),
        "concerns": concerns,
        "overall_observation": overall_observation,
        "laboratory_values_available": False,
        "scientific_note": "Farmer Experience Mode provides an observation-based preliminary assessment. It does not replace laboratory soil testing and does not directly measure soil nutrients.",
    }


def generate_farmer_summary(assessment: dict[str, Any]) -> dict[str, Any]:
    observations = assessment["observations"]
    problems = observations["visible_problems"]
    return {
        "soil_appearance": observations["soil_color"],
        "moisture": observations["moisture_appearance"],
        "previous_crop": observations["previous_crop"],
        "previous_crop_performance": f"{observations['previous_growth']} growth / {observations['previous_yield']} yield",
        "visible_problems": ", ".join(problems),
        "water_retention": observations["water_retention"],
        "overall_condition": observations["overall_condition"],
        "overall_observation": assessment["overall_observation"],
    }


def generate_follow_up_guidance(assessment: dict[str, Any]) -> list[str]:
    guidance = [
        "Upload a laboratory soil report for exact nutrient analysis.",
        "Perform laboratory soil testing if chemical values are unknown.",
        "Upload a soil image for preliminary visual assessment.",
        "Enter known N/P/K/pH/Organic Carbon values manually if available.",
    ]
    if assessment["concerns"]:
        guidance.append("Discuss the observed field concerns with a qualified agronomist alongside verified soil-test results.")
    return guidance
