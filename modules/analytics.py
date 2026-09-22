from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data_utils import NUMERIC_FEATURES, default_processed_path
from .nutrient_analysis import analyze_all_nutrients


PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#b7ee65"
SECONDARY = "#4da778"
GOLD = "#e0b45e"
GRID = "rgba(160, 190, 160, 0.12)"


def load_analytics_data(path: str | Path = default_processed_path()) -> tuple[pd.DataFrame | None, str | None]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return None, f"Analytics dataset was not found: {dataset_path}"
    try:
        dataframe = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return None, f"Analytics dataset could not be read: {error}"
    if dataframe.empty:
        return None, "The analytics dataset is empty."
    return dataframe, None


def get_dataset_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "total_samples": int(len(dataframe)),
        "total_features": int(len(dataframe.columns)),
        "columns": [str(column) for column in dataframe.columns],
        "available_dimensions": [column for column in ["Region", "Crop", "Soil_Type", "Fertility"] if column in dataframe.columns],
    }


def calculate_nutrient_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in NUMERIC_FEATURES if column in dataframe.columns]
    if not available:
        return pd.DataFrame()
    statistics = dataframe[available].apply(pd.to_numeric, errors="coerce").agg(["count", "mean", "median", "min", "max", "std"]).T
    statistics.index = ["Organic Carbon" if column == "Organic_Carbon" else column for column in statistics.index]
    return statistics.round(2)


def calculate_deficiency_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in NUMERIC_FEATURES if column in dataframe.columns]
    records: list[dict[str, str]] = []
    for _, row in dataframe.iterrows():
        values = {column: row[column] for column in available}
        try:
            analysis = analyze_all_nutrients({**values, "pH": row.get("pH"), "Organic_Carbon": row.get("Organic_Carbon")})
        except (TypeError, ValueError):
            continue
        statuses = {item["nutrient"]: item["status"] for item in analysis["nutrients"]}
        records.append({"Pattern": " + ".join(f"{key}: {value}" for key, value in statuses.items())})
    if not records:
        return pd.DataFrame(columns=["Pattern", "Samples", "Percentage"])
    distribution = pd.DataFrame(records)["Pattern"].value_counts().rename_axis("Pattern").reset_index(name="Samples")
    distribution["Percentage"] = (distribution["Samples"] / len(records) * 100).round(1)
    return distribution


def calculate_region_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Region" not in dataframe.columns:
        return pd.DataFrame()
    available = [column for column in NUMERIC_FEATURES if column in dataframe.columns]
    return dataframe.groupby("Region", dropna=False)[available].mean(numeric_only=True).round(2).reset_index()


def calculate_crop_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Crop" not in dataframe.columns:
        return pd.DataFrame()
    available = [column for column in NUMERIC_FEATURES if column in dataframe.columns]
    return dataframe.groupby("Crop", dropna=False)[available].mean(numeric_only=True).round(2).reset_index()


def calculate_fertility_distribution(dataframe: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    if "Fertility" not in dataframe.columns:
        return pd.DataFrame(columns=["Fertility", "Samples", "Percentage"])
    if group_column and group_column in dataframe.columns:
        counts = dataframe.groupby([group_column, "Fertility"], dropna=False).size().reset_index(name="Samples")
        counts["Percentage"] = counts.groupby(group_column)["Samples"].transform(lambda values: (values / values.sum() * 100).round(1))
        return counts
    counts = dataframe["Fertility"].value_counts(dropna=False).rename_axis("Fertility").reset_index(name="Samples")
    counts["Percentage"] = (counts["Samples"] / counts["Samples"].sum() * 100).round(1)
    return counts


def calculate_correlation_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    return dataframe[numeric_columns].corr(numeric_only=True).round(2)


def _figure_layout(figure: go.Figure, height: int = 300) -> go.Figure:
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin={"l": 12, "r": 12, "t": 42, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#b9c6b9"},
        title_font={"family": "Manrope, sans-serif", "color": "#eaf2ea", "size": 15},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return figure


def nutrient_distribution_figure(dataframe: pd.DataFrame, column: str) -> go.Figure:
    figure = px.histogram(
        dataframe,
        x=column,
        nbins=14,
        title=f"{column} distribution",
        color_discrete_sequence=[ACCENT],
    )
    figure.update_traces(marker_line_color="#102017", marker_line_width=1, opacity=.88)
    return _figure_layout(figure)


def correlation_figure(dataframe: pd.DataFrame, columns: list[str]) -> go.Figure:
    correlation = dataframe[columns].corr(numeric_only=True)
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale=[[0, "#173b2b"], [.5, "#39734b"], [1, "#b7ee65"]],
        zmin=-1,
        zmax=1,
        title="Feature correlation matrix",
    )
    return _figure_layout(figure, height=390)


def class_distribution_figure(dataframe: pd.DataFrame, target_column: str = "Fertility") -> go.Figure:
    counts = dataframe[target_column].value_counts().rename_axis(target_column).reset_index(name="Samples")
    order = [label for label in ["Low", "Medium", "High"] if label in counts[target_column].tolist()]
    figure = px.bar(
        counts,
        x=target_column,
        y="Samples",
        title="Fertility label distribution",
        category_orders={target_column: order},
        color=target_column,
        color_discrete_map={"Low": "#d88962", "Medium": GOLD, "High": ACCENT},
    )
    return _figure_layout(figure)


def regional_summary_figure(dataframe: pd.DataFrame, region_column: str = "Region") -> go.Figure:
    counts = dataframe[region_column].value_counts().rename_axis(region_column).reset_index(name="Samples")
    figure = px.bar(
        counts,
        x=region_column,
        y="Samples",
        title="Samples by region",
        color_discrete_sequence=[SECONDARY],
    )
    return _figure_layout(figure)


def feature_importance_figure(importances: dict[str, float]) -> go.Figure:
    importance_frame = pd.DataFrame({"Feature": list(importances), "Importance": list(importances.values())})
    importance_frame = importance_frame.sort_values("Importance", ascending=True)
    figure = px.bar(
        importance_frame,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest feature importance",
        color="Importance",
        color_continuous_scale=[[0, "#39734b"], [1, "#b7ee65"]],
    )
    figure.update_coloraxes(showscale=False)
    return _figure_layout(figure)


def confusion_matrix_figure(matrix: list[list[int]], class_order: list[str]) -> go.Figure:
    figure = px.imshow(
        matrix,
        x=class_order,
        y=class_order,
        text_auto=True,
        color_continuous_scale=[[0, "#102319"], [.5, "#39734b"], [1, "#b7ee65"]],
        title="Confusion matrix · actual vs predicted",
        labels={"x": "Predicted class", "y": "Actual class", "color": "Samples"},
    )
    return _figure_layout(figure, height=360)


def prediction_probability_figure(probabilities: dict[str, float]) -> go.Figure:
    probability_frame = pd.DataFrame({"Class": list(probabilities), "Probability": list(probabilities.values())})
    figure = px.bar(
        probability_frame,
        x="Class",
        y="Probability",
        title="Prediction probabilities",
        color="Class",
        color_discrete_map={"Low": "#d88962", "Medium": GOLD, "High": ACCENT},
    )
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    figure.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    return _figure_layout(figure)


def probability_comparison_figure(current: dict[str, float], simulated: dict[str, float]) -> go.Figure:
    probability_frame = pd.DataFrame(
        [
            {"Class": label, "Probability": value, "Profile": "Current soil"}
            for label, value in current.items()
        ]
        + [
            {"Class": label, "Probability": value, "Profile": "Simulated soil"}
            for label, value in simulated.items()
        ]
    )
    figure = px.bar(
        probability_frame,
        x="Class",
        y="Probability",
        color="Profile",
        barmode="group",
        title="Current vs simulated class probabilities",
        color_discrete_map={"Current soil": "#6f8f73", "Simulated soil": "#b7ee65"},
    )
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    figure.update_traces(hovertemplate="%{x}: %{y:.1%}<extra>%{fullData.name}</extra>")
    return _figure_layout(figure)


def npk_comparison_figure(values: dict[str, float]) -> go.Figure:
    nutrient_frame = pd.DataFrame({"Nutrient": ["N", "P", "K"], "Value": [values["N"], values["P"], values["K"]]})
    figure = px.bar(
        nutrient_frame,
        x="Nutrient",
        y="Value",
        title="NPK comparison",
        color="Nutrient",
        color_discrete_map={"N": "#b7ee65", "P": "#e0b45e", "K": "#4da778"},
    )
    return _figure_layout(figure)


def nutrient_status_figure(analyses: list[dict[str, object]]) -> go.Figure:
    status_frame = pd.DataFrame(
        {"Nutrient": [str(item["nutrient"]) for item in analyses], "Status": [str(item["status"]) for item in analyses]}
    )
    figure = px.bar(
        status_frame,
        x="Nutrient",
        y=[1] * len(status_frame),
        title="Nutrient status overview",
        color="Status",
        color_discrete_map={"Low": "#d88962", "Adequate": "#b7ee65", "Suitable": "#b7ee65", "High": "#e0b45e", "Acidic": "#d88962", "Alkaline": "#e0b45e"},
    )
    figure.update_yaxes(visible=False, range=[0, 1.25])
    figure.update_traces(hovertemplate="%{x}: %{marker.color}<extra></extra>")
    return _figure_layout(figure)


def normalized_npk_figure(dataframe: pd.DataFrame) -> go.Figure:
    available = [column for column in ["N", "P", "K"] if column in dataframe.columns]
    normalized = dataframe[available].apply(pd.to_numeric, errors="coerce")
    normalized = (normalized - normalized.mean()) / normalized.std(ddof=0).replace(0, 1)
    long_frame = normalized.reset_index(drop=True).melt(var_name="Nutrient", value_name="Standardized value")
    figure = px.box(long_frame, x="Nutrient", y="Standardized value", color="Nutrient", title="NPK comparison on standardized scale", color_discrete_map={"N": ACCENT, "P": GOLD, "K": SECONDARY})
    figure.update_layout(showlegend=False)
    return _figure_layout(figure)


def grouped_region_nutrient_figure(dataframe: pd.DataFrame, nutrient: str) -> go.Figure:
    figure = px.bar(dataframe, x="Region", y=nutrient, title=f"Average {nutrient} by region", color="Region", color_discrete_sequence=[SECONDARY])
    figure.update_layout(showlegend=False)
    return _figure_layout(figure)


def grouped_crop_nutrient_figure(dataframe: pd.DataFrame, nutrient: str) -> go.Figure:
    figure = px.bar(dataframe, x="Crop", y=nutrient, title=f"Average {nutrient} by crop", color="Crop", color_discrete_sequence=[ACCENT])
    figure.update_layout(showlegend=False)
    return _figure_layout(figure)


def deficiency_pattern_figure(dataframe: pd.DataFrame) -> go.Figure:
    figure = px.bar(dataframe, x="Samples", y="Pattern", orientation="h", title="Nutrient status patterns", color="Samples", color_continuous_scale=[[0, "#39734b"], [1, "#b7ee65"]])
    figure.update_coloraxes(showscale=False)
    return _figure_layout(figure, height=max(300, 42 * len(dataframe)))


def fertility_by_group_figure(dataframe: pd.DataFrame, group_column: str) -> go.Figure:
    figure = px.bar(dataframe, x=group_column, y="Samples", color="Fertility", barmode="stack", title=f"Fertility distribution by {group_column.lower()}", color_discrete_map={"Low": "#d88962", "Medium": GOLD, "High": ACCENT})
    return _figure_layout(figure)
