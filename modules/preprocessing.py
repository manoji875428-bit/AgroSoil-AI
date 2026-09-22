from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_utils import NUMERIC_FEATURES, PLAUSIBLE_RANGES, TARGET_COLUMN, default_processed_path, default_summary_path


@dataclass(frozen=True)
class PreprocessingConfig:
    remove_duplicates: bool = True
    impute_missing: bool = True
    flag_outliers: bool = True
    add_engineered_features: bool = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _numeric_conversion(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    converted = dataframe.copy()
    invalid_values: dict[str, int] = {}
    for column in NUMERIC_FEATURES:
        if column not in converted.columns:
            continue
        original_non_null = converted[column].notna()
        numeric_values = pd.to_numeric(converted[column], errors="coerce")
        invalid_values[column] = int(original_non_null.sum() - numeric_values.notna().sum())
        converted[column] = numeric_values
    return converted, invalid_values


def _range_warnings(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    checked = dataframe.copy()
    warnings: dict[str, int] = {}
    for column, (lower, upper) in PLAUSIBLE_RANGES.items():
        if column not in checked.columns:
            continue
        invalid_mask = checked[column].notna() & ((checked[column] < lower) | (checked[column] > upper))
        warnings[column] = int(invalid_mask.sum())
        # Values outside the broad physical/semantic range become missing and are imputed.
        checked.loc[invalid_mask, column] = pd.NA
    return checked, warnings


def detect_iqr_outliers(dataframe: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in NUMERIC_FEATURES:
        if column not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            counts[column] = 0
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        counts[column] = int(((values < lower_bound) | (values > upper_bound)).sum())
    return counts


def _impute_missing_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    filled = dataframe.copy()
    numeric_fill_values: dict[str, float] = {}
    categorical_fill_values: dict[str, str] = {}
    for column in filled.columns:
        if pd.api.types.is_numeric_dtype(filled[column]):
            median = filled[column].median()
            if pd.notna(median):
                numeric_fill_values[column] = float(median)
                filled[column] = filled[column].fillna(median)
        else:
            non_null = filled[column].dropna()
            if not non_null.empty:
                mode = str(non_null.mode().iloc[0])
                categorical_fill_values[column] = mode
                filled[column] = filled[column].fillna(mode)
    return filled, {"numeric": numeric_fill_values, "categorical": categorical_fill_values}


def preprocess_dataset(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a soil dataframe while retaining explainable quality metadata."""
    settings = config or PreprocessingConfig()
    rows_before = int(len(dataframe))
    missing_before = int(dataframe.isna().sum().sum())
    duplicates_before = int(dataframe.duplicated().sum())
    working, invalid_values = _numeric_conversion(dataframe)
    working, range_warnings = _range_warnings(working)

    if settings.remove_duplicates:
        working = working.drop_duplicates().reset_index(drop=True)
    duplicates_removed = duplicates_before if settings.remove_duplicates else 0
    outlier_counts = detect_iqr_outliers(working) if settings.flag_outliers else {}
    imputation_values: dict[str, Any] = {}
    if settings.impute_missing:
        working, imputation_values = _impute_missing_values(working)

    if settings.add_engineered_features and all(column in working.columns for column in ["N", "P", "K"]):
        working["NPK_Total"] = working[["N", "P", "K"]].sum(axis=1)

    numeric_features = [column for column in working.columns if pd.api.types.is_numeric_dtype(working[column])]
    categorical_features = [column for column in working.columns if column not in numeric_features and column != TARGET_COLUMN]
    target = TARGET_COLUMN if TARGET_COLUMN in working.columns else None
    report = {
        "rows_before": rows_before,
        "rows_after": int(len(working)),
        "columns_before": int(len(dataframe.columns)),
        "columns_after": int(len(working.columns)),
        "duplicates_before": duplicates_before,
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_before,
        "missing_values_after": int(working.isna().sum().sum()),
        "invalid_numeric_values": invalid_values,
        "range_warnings": range_warnings,
        "outlier_counts": outlier_counts,
        "imputation_values": imputation_values,
        "features": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
        "config": {
            "remove_duplicates": settings.remove_duplicates,
            "impute_missing": settings.impute_missing,
            "flag_outliers": settings.flag_outliers,
            "add_engineered_features": settings.add_engineered_features,
        },
    }
    return working, _json_safe(report)


def run_preprocessing(
    dataframe: pd.DataFrame,
    processed_path: str | Path = default_processed_path(),
    summary_path: str | Path = default_summary_path(),
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed, report = preprocess_dataset(dataframe, config)
    output_path = Path(processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    metadata_path = Path(summary_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return processed, report
