from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#b7ee65"
SECONDARY = "#4da778"
GOLD = "#e0b45e"
GRID = "rgba(160, 190, 160, 0.12)"


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
