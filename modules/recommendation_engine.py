from __future__ import annotations

from typing import Any


PROTOTYPE_GUIDANCE = (
    "Prototype guidance — validate with crop, soil type, region and professional agronomic advice."
)


def _recommendation(
    nutrient: str,
    status: str,
    recommendation: str,
    priority: str,
    reason: str,
) -> dict[str, str]:
    return {
        "nutrient": nutrient,
        "status": status,
        "recommendation": recommendation,
        "priority": priority,
        "reason": reason,
        "guidance_note": PROTOTYPE_GUIDANCE,
    }


def recommend_nitrogen_support(result: dict[str, Any]) -> dict[str, str]:
    status = str(result["status"])
    if status == "Low":
        return _recommendation(
            "Nitrogen",
            status,
            "Consider a nitrogen-support fertilizer category or an appropriate organic nutrient-management option.",
            "High",
            "The analyzed nitrogen value is below the configured demo/prototype threshold.",
        )
    if status == "High":
        return _recommendation(
            "Nitrogen",
            status,
            "Avoid adding a targeted nitrogen-support category until the result is reviewed with crop and soil context.",
            "Medium",
            "The analyzed nitrogen value is above the configured demo/prototype range.",
        )
    return _recommendation(
        "Nitrogen",
        status,
        "No targeted nitrogen category is indicated by this prototype analysis.",
        "Informational",
        "The analyzed nitrogen value is within the configured demo/prototype adequate range.",
    )


def recommend_phosphorus_support(result: dict[str, Any]) -> dict[str, str]:
    status = str(result["status"])
    if status == "Low":
        return _recommendation(
            "Phosphorus",
            status,
            "Consider a phosphorus-support fertilizer category after professional soil-test review.",
            "High",
            "The analyzed phosphorus value is below the configured demo/prototype threshold.",
        )
    if status == "High":
        return _recommendation(
            "Phosphorus",
            status,
            "Avoid adding a targeted phosphorus-support category until accumulation and crop context are reviewed.",
            "Medium",
            "The analyzed phosphorus value is above the configured demo/prototype range.",
        )
    return _recommendation(
        "Phosphorus",
        status,
        "No targeted phosphorus category is indicated by this prototype analysis.",
        "Informational",
        "The analyzed phosphorus value is within the configured demo/prototype adequate range.",
    )


def recommend_potassium_support(result: dict[str, Any]) -> dict[str, str]:
    status = str(result["status"])
    if status == "Low":
        return _recommendation(
            "Potassium",
            status,
            "Consider a potassium-support fertilizer category after professional soil-test review.",
            "High",
            "The analyzed potassium value is below the configured demo/prototype threshold.",
        )
    if status == "High":
        return _recommendation(
            "Potassium",
            status,
            "Avoid adding a targeted potassium-support category until the result is reviewed with crop and soil context.",
            "Medium",
            "The analyzed potassium value is above the configured demo/prototype range.",
        )
    return _recommendation(
        "Potassium",
        status,
        "No targeted potassium category is indicated by this prototype analysis.",
        "Informational",
        "The analyzed potassium value is within the configured demo/prototype adequate range.",
    )


def recommend_ph_management(result: dict[str, Any]) -> dict[str, str]:
    status = str(result["status"])
    if status == "Acidic":
        return _recommendation(
            "pH",
            status,
            "Review general soil pH-management practices with a qualified agronomist; do not apply a corrective amendment from this prototype alone.",
            "Medium",
            "The analyzed pH is below the configured demo/prototype suitable range.",
        )
    if status == "Alkaline":
        return _recommendation(
            "pH",
            status,
            "Review general alkaline-soil management practices with a qualified agronomist; do not apply a corrective amendment from this prototype alone.",
            "Medium",
            "The analyzed pH is above the configured demo/prototype suitable range.",
        )
    return _recommendation(
        "pH",
        status,
        "Continue monitoring pH through validated soil testing; no targeted pH-management category is indicated here.",
        "Informational",
        "The analyzed pH is within the configured demo/prototype suitable range.",
    )


def recommend_organic_matter_management(result: dict[str, Any]) -> dict[str, str]:
    status = str(result["status"])
    if status == "Low":
        return _recommendation(
            "Organic Carbon",
            status,
            "Consider compost, well-decomposed organic matter, crop-residue management or another suitable organic amendment category.",
            "Medium",
            "The analyzed organic carbon value is below the configured demo/prototype threshold.",
        )
    if status == "High":
        return _recommendation(
            "Organic Carbon",
            status,
            "Maintain current organic-matter practices and confirm interpretation with crop and soil context.",
            "Informational",
            "The analyzed organic carbon value is above the configured demo/prototype range.",
        )
    return _recommendation(
        "Organic Carbon",
        status,
        "Maintain suitable organic-matter practices; no targeted organic amendment category is indicated here.",
        "Informational",
        "The analyzed organic carbon value is within the configured demo/prototype adequate range.",
    )


def build_recommendation_summary(recommendations: list[dict[str, str]]) -> dict[str, Any]:
    deficiencies = [item for item in recommendations if item["status"] == "Low"]
    pH_issues = [item for item in recommendations if item["nutrient"] == "pH" and item["status"] in {"Acidic", "Alkaline"}]
    high_nutrients = [item for item in recommendations if item["status"] == "High"]
    summary_lines: list[str] = []
    if len(deficiencies) > 1:
        summary_lines.append(f"{len(deficiencies)} nutrient deficiencies detected")
        summary_lines.append("Multiple nutrient deficiencies detected.")
    elif deficiencies:
        summary_lines.append(f"{deficiencies[0]['nutrient']} requires attention")
    else:
        summary_lines.append("No low nutrient status detected")
    if pH_issues:
        summary_lines.append("pH requires monitoring")
    if high_nutrients:
        summary_lines.append(f"{', '.join(item['nutrient'] for item in high_nutrients)} should be reviewed before adding more")
    if not deficiencies and not pH_issues and not high_nutrients:
        summary_lines.append("All analyzed conditions are within the prototype ranges")

    return {
        "recommendation_count": len(recommendations),
        "detected_issue_count": len(deficiencies) + len(pH_issues) + len(high_nutrients),
        "deficiency_count": len(deficiencies),
        "high_condition_count": len(high_nutrients),
        "pH_issue": bool(pH_issues),
        "summary_lines": summary_lines,
        "guidance_note": PROTOTYPE_GUIDANCE,
    }


def generate_recommendations(analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate category-level prototype guidance from Part 4 analysis output."""
    results = {item["nutrient"]: item for item in analysis.get("nutrients", [])}
    required = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Organic Carbon"]
    missing = [nutrient for nutrient in required if nutrient not in results]
    if missing:
        raise ValueError(f"Nutrient analysis is missing: {', '.join(missing)}")

    recommendations = [
        recommend_nitrogen_support(results["Nitrogen"]),
        recommend_phosphorus_support(results["Phosphorus"]),
        recommend_potassium_support(results["Potassium"]),
        recommend_ph_management(results["pH"]),
        recommend_organic_matter_management(results["Organic Carbon"]),
    ]
    return {
        "recommendations": recommendations,
        "summary": build_recommendation_summary(recommendations),
    }
