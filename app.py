from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modules.analytics import (
    class_distribution_figure,
    correlation_figure,
    nutrient_distribution_figure,
    regional_summary_figure,
)
from modules.data_loader import get_dataset_summary, load_raw_dataset, validate_dataset
from modules.data_utils import NUMERIC_FEATURES, TARGET_COLUMN, default_processed_path, default_raw_path
from modules.preprocessing import run_preprocessing


st.set_page_config(
    page_title="AGROSOIL AI | DataXcelerate 2026",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


NAVIGATION = {
    "HOME": [("⌂", "Overview", "home")],
    "ANALYZE": [
        ("▣", "Soil Report", "soil-report"),
        ("◉", "Soil Image", "soil-image"),
        ("♧", "Farmer Experience", "farmer-experience"),
    ],
    "INTELLIGENCE": [
        ("◫", "Data Intelligence", "data-intelligence"),
        ("◈", "Soil Health", "soil-health"),
        ("▦", "Analytics", "analytics"),
        ("⌖", "Regional Insights", "regional-insights"),
    ],
    "SIMULATION": [("◇", "What-If Simulator", "simulator")],
    "SYSTEM": [
        ("▤", "Analysis History", "history"),
        ("✦", "Model Insights", "model-insights"),
        ("ⓘ", "About", "about"),
    ],
}


@st.cache_data

def get_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #eaf2ea;
        --muted: #9aa99d;
        --muted-strong: #b9c6b9;
        --panel: rgba(19, 31, 25, 0.78);
        --panel-soft: rgba(26, 42, 31, 0.58);
        --line: rgba(161, 190, 157, 0.14);
        --lime: #b7ee65;
        --lime-soft: #8ac84d;
        --gold: #e0b45e;
        --green: #1d8b5a;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 80% 4%, rgba(73, 126, 61, .18), transparent 25rem),
            radial-gradient(circle at 9% 45%, rgba(25, 92, 61, .16), transparent 28rem),
            linear-gradient(135deg, #07110d 0%, #0b1812 48%, #101b15 100%);
        font-family: 'DM Sans', sans-serif;
    }

    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .22;
        background-image: linear-gradient(rgba(163, 197, 156, .035) 1px, transparent 1px), linear-gradient(90deg, rgba(163, 197, 156, .035) 1px, transparent 1px);
        background-size: 54px 54px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { max-width: 1440px; padding: 2.2rem 3rem 4rem; }
    h1, h2, h3, h4, p { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.02em; }
    .eyebrow { color: var(--lime); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .section-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .55rem; }
    .section-title { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: clamp(1.5rem, 2.4vw, 2.15rem); font-weight: 700; line-height: 1.12; margin: 0; }
    .section-copy { color: var(--muted); font-size: .93rem; margin: .5rem 0 0; }

    [data-testid="stSidebar"] { background: rgba(7, 18, 12, .88); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.25rem; }
    [data-testid="stSidebar"] .stButton > button { justify-content: flex-start; border: 0; background: transparent; color: #9eafa1; padding: .62rem .8rem; font-size: .84rem; transition: all .2s ease; }
    [data-testid="stSidebar"] .stButton > button:hover { color: var(--ink); background: rgba(183, 238, 101, .08); border-color: transparent; transform: translateX(3px); }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] { color: var(--lime); background: rgba(183, 238, 101, .1); border-left: 2px solid var(--lime); }
    .brand { display: flex; align-items: center; gap: .75rem; padding: .2rem .55rem 2.4rem; }
    .brand-mark { width: 2.45rem; height: 2.45rem; display: grid; place-items: center; border-radius: 12px; color: #112015; background: linear-gradient(145deg, var(--lime), #79bd4a); box-shadow: 0 0 24px rgba(183, 238, 101, .2); font-size: 1.35rem; }
    .brand-name { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 800; letter-spacing: .08em; line-height: 1; }
    .brand-sub { color: var(--lime); font-size: .63rem; font-weight: 700; letter-spacing: .2em; margin-top: .3rem; }
    .nav-label { color: #64766a; font-size: .62rem; font-weight: 700; letter-spacing: .16em; margin: 1.05rem .8rem .28rem; }
    .sidebar-foot { border-top: 1px solid var(--line); color: #6e8173; font-size: .69rem; line-height: 1.5; margin: 2.4rem .55rem 0; padding-top: 1.1rem; }

    .hero { animation: rise .65s ease both; padding: 2rem 0 2.65rem; }
    .hero-grid { align-items: end; display: grid; gap: 2rem; grid-template-columns: minmax(0, 1.25fr) minmax(250px, .75fr); }
    .hero-title { font-family: 'Manrope', sans-serif; font-size: clamp(2.8rem, 6vw, 5.5rem); font-weight: 800; letter-spacing: -.065em; line-height: .98; margin: .65rem 0 1.1rem; max-width: 760px; }
    .hero-title span { color: var(--lime); }
    .hero-copy { color: var(--muted-strong); font-size: 1rem; line-height: 1.65; max-width: 560px; }
    .hero-meta { border-left: 1px solid var(--line); padding: .5rem 0 .5rem 1.5rem; }
    .hero-meta-title { color: var(--muted); font-size: .71rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-meta-value { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; margin-top: .45rem; }
    .hero-meta-copy { color: var(--muted); font-size: .83rem; line-height: 1.55; margin-top: .35rem; }

    .stButton > button { border: 1px solid rgba(183, 238, 101, .28); border-radius: 10px; color: #112015; background: var(--lime); font-weight: 700; padding: .72rem 1.1rem; transition: all .2s ease; }
    .stButton > button:hover { background: #d0f991; border-color: #d0f991; box-shadow: 0 8px 24px rgba(183, 238, 101, .16); transform: translateY(-2px); }
    .secondary-btn .stButton > button { color: var(--muted-strong); background: transparent; border-color: var(--line); }
    .secondary-btn .stButton > button:hover { color: var(--ink); background: rgba(255,255,255,.04); border-color: rgba(183, 238, 101, .35); }

    .kpi-card, .action-card, .feature-card, .workflow-card, .about-panel { background: linear-gradient(145deg, rgba(24, 40, 29, .8), rgba(14, 26, 19, .72)); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 16px 45px rgba(0,0,0,.12); }
    .kpi-card { min-height: 128px; padding: 1.1rem 1.2rem; }
    .kpi-label { color: var(--muted); font-size: .68rem; font-weight: 700; letter-spacing: .11em; }
    .kpi-value { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 2.05rem; font-weight: 700; line-height: 1; margin: 1rem 0 .45rem; }
    .kpi-note { color: #708273; font-size: .72rem; }
    .kpi-accent { color: var(--lime); }
    .section { animation: rise .7s ease both; margin-top: 3.2rem; }
    .section-head { align-items: end; display: flex; justify-content: space-between; margin-bottom: 1.25rem; }
    .action-card { min-height: 238px; padding: 1.35rem; transition: all .25s ease; }
    .action-card:hover, .feature-card:hover { border-color: rgba(183, 238, 101, .32); box-shadow: 0 18px 40px rgba(0,0,0,.2); transform: translateY(-5px); }
    .action-icon, .feature-icon { align-items: center; background: rgba(183, 238, 101, .1); border: 1px solid rgba(183, 238, 101, .13); border-radius: 11px; color: var(--lime); display: flex; font-size: 1.35rem; height: 2.7rem; justify-content: center; width: 2.7rem; }
    .action-title { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 1.08rem; font-weight: 700; margin: 1.15rem 0 .6rem; }
    .action-copy, .feature-copy { color: var(--muted); font-size: .86rem; line-height: 1.55; min-height: 3.8rem; }
    .action-link { color: var(--lime); font-size: .79rem; font-weight: 700; margin-top: 1.15rem; }
    .feature-card { min-height: 174px; padding: 1.25rem; transition: all .25s ease; }
    .feature-title { color: var(--ink); font-size: .98rem; font-weight: 700; margin: .95rem 0 .45rem; }
    .workflow { align-items: stretch; display: flex; gap: .6rem; }
    .workflow-card { flex: 1; padding: 1.15rem; position: relative; }
    .workflow-card:not(:last-child)::after { color: var(--lime); content: '→'; font-size: 1.25rem; position: absolute; right: -.88rem; top: 1.15rem; z-index: 2; }
    .workflow-no { color: var(--lime); font-family: 'Manrope', sans-serif; font-size: .72rem; font-weight: 800; }
    .workflow-title { color: var(--ink); font-size: .82rem; font-weight: 700; letter-spacing: .08em; margin-top: .9rem; }
    .workflow-copy { color: var(--muted); font-size: .75rem; margin-top: .3rem; }
    .about-panel { padding: 1.6rem; }
    .placeholder-icon { color: var(--lime); font-size: 2.2rem; }
    .placeholder-title { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 2rem; font-weight: 700; margin: .7rem 0 .6rem; }
    .placeholder-copy { color: var(--muted-strong); font-size: .95rem; line-height: 1.65; max-width: 680px; }
    .status-pill { color: var(--gold); display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .11em; margin-top: 1.2rem; padding: .45rem .7rem; border: 1px solid rgba(224,180,94,.24); border-radius: 99px; text-transform: uppercase; }
    @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 900px) { .block-container { padding: 1.5rem 1.25rem 3rem; } .hero-grid { grid-template-columns: 1fr; } .hero-meta { border-left: 0; border-top: 1px solid var(--line); padding: 1.2rem 0 0; } .workflow { flex-wrap: wrap; } .workflow-card { flex: 1 1 30%; min-width: 130px; } .workflow-card::after { display: none; } }
    @media (max-width: 600px) { .hero-title { font-size: 3.1rem; } .section-head { align-items: start; flex-direction: column; gap: .4rem; } .workflow-card { flex-basis: 45%; } }
    </style>
    """


def go_to(page: str) -> None:
    st.session_state["page"] = page


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">🌱</div><div><div class="brand-name">AGROSOIL</div><div class="brand-sub">AI PORTAL</div></div></div>', unsafe_allow_html=True)
        current_page = st.session_state.get("page", "home")
        for group, items in NAVIGATION.items():
            st.markdown(f'<div class="nav-label">{group}</div>', unsafe_allow_html=True)
            for icon, label, page in items:
                if st.button(f"{icon}   {label}", key=f"nav-{page}", use_container_width=True, type="primary" if page == current_page else "secondary"):
                    go_to(page)
                    st.rerun()
        st.markdown('<div class="sidebar-foot">DataXcelerate 2026 • PS22<br>Soil intelligence for a more resilient tomorrow.</div>', unsafe_allow_html=True)


def render_section_header(label: str, title: str, copy: str = "") -> None:
    copy_markup = f'<p class="section-copy">{copy}</p>' if copy else ""
    st.markdown(f'<div class="section-head"><div><div class="section-label">{label}</div><h2 class="section-title">{title}</h2>{copy_markup}</div></div>', unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, note: str, accent: bool = False) -> None:
    value_class = "kpi-value kpi-accent" if accent else "kpi-value"
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="{value_class}">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def render_action_card(icon: str, title: str, copy: str, button_label: str, page: str, key: str) -> None:
    st.markdown(f'<div class="action-card"><div class="action-icon">{icon}</div><div class="action-title">{title}</div><div class="action-copy">{copy}</div><div class="action-link">{button_label}</div></div>', unsafe_allow_html=True)
    if st.button(button_label, key=key, use_container_width=True):
        go_to(page)
        st.rerun()


def render_feature_card(icon: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>', unsafe_allow_html=True)


def render_workflow() -> None:
    stages = [("01", "INPUT", "Collect soil signals"), ("02", "PREPROCESS", "Prepare clean data"), ("03", "ANALYZE", "Read nutrient context"), ("04", "PREDICT", "Model fertility state"), ("05", "RECOMMEND", "Guide the next step")]
    cards = "".join(f'<div class="workflow-card"><div class="workflow-no">{number}</div><div class="workflow-title">{title}</div><div class="workflow-copy">{copy}</div></div>' for number, title, copy in stages)
    st.markdown(f'<div class="workflow">{cards}</div>', unsafe_allow_html=True)


def render_data_intelligence() -> None:
    st.markdown('<div class="hero"><div class="hero-grid"><div><div class="eyebrow">◫ DATA INTELLIGENCE · PART 2</div><h1 class="hero-title">See the signal<br><span>before the model.</span></h1><p class="hero-copy">A transparent preprocessing workspace for validating soil data, understanding quality and preparing a clean foundation for future ML training.</p></div><div class="hero-meta"><div class="hero-meta-title">Demo / Synthetic Dataset</div><div class="hero-meta-value">Ready for inspection.</div><div class="hero-meta-copy">This development dataset is synthetic and exists to exercise the pipeline. It does not represent real farm soil.</div></div></div></div>', unsafe_allow_html=True)

    raw_dataframe, load_error = load_raw_dataset(default_raw_path())
    if load_error or raw_dataframe is None:
        st.error(load_error or "The current dataset could not be loaded.")
        return

    validation = validate_dataset(raw_dataframe)
    if not validation["valid"]:
        missing_columns = ", ".join(validation["missing_required_columns"])
        st.error(f"The dataset is missing required soil columns: {missing_columns}")
        return

    try:
        processed_dataframe, processing_report = run_preprocessing(raw_dataframe)
    except (OSError, ValueError, TypeError) as error:
        st.error(f"The dataset could not be preprocessed safely: {error}")
        return

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Dataset overview", "A clean view of what arrived", "Source: soil_demo_synthetic.csv · No ML predictions or recommendations are generated here.")
    overview_columns = st.columns(4)
    overview_kpis = [
        ("TOTAL SAMPLES", str(validation["row_count"]), "Rows in raw source", False),
        ("TOTAL FEATURES", str(validation["column_count"]), "Columns in raw source", True),
        ("MISSING VALUES", str(validation["missing_values"]), "Before preprocessing", False),
        ("DUPLICATE ROWS", str(validation["duplicate_rows"]), "Detected in raw source", False),
    ]
    for column, (label, value, note, accent) in zip(overview_columns, overview_kpis):
        with column:
            render_kpi_card(label, value, note, accent)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Data quality", "What changed during preparation", "Suspicious values are reported transparently; potential IQR outliers are flagged, not deleted.")
    quality_columns = st.columns(2)
    with quality_columns[0]:
        missing_rows = pd.DataFrame(
            [(column, count) for column, count in validation["missing_by_column"].items()],
            columns=["Column", "Missing values"],
        )
        if missing_rows.empty:
            st.success("No missing values were detected in the raw dataset.")
        else:
            st.dataframe(missing_rows, use_container_width=True, hide_index=True)
        st.markdown(f'<div class="kpi-note">Missing values after imputation: <strong>{processing_report["missing_values_after"]}</strong></div>', unsafe_allow_html=True)
    with quality_columns[1]:
        quality_rows = []
        for column, details in validation["numeric_validation"].items():
            quality_rows.append({"Column": column, "Status": details["status"], "Invalid values": details["invalid_values"]})
        quality_rows.extend([
            {"Column": "Duplicates", "Status": "Removed" if processing_report["duplicates_removed"] else "Retained", "Invalid values": processing_report["duplicates_removed"]},
            {"Column": "Range warnings", "Status": "Flagged for imputation", "Invalid values": sum(processing_report["range_warnings"].values())},
        ])
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Soil feature statistics", "Know the shape of each signal", "Statistics below use the processed dataset after validation and imputation.")
    statistics = processed_dataframe[[column for column in NUMERIC_FEATURES if column in processed_dataframe.columns]].describe().T
    statistics["median"] = processed_dataframe[[column for column in NUMERIC_FEATURES if column in processed_dataframe.columns]].median()
    statistics = statistics.rename(columns={"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "std": "Standard deviation"})
    statistics = statistics[["Mean", "Median", "Minimum", "Maximum", "Standard deviation"]].round(2)
    statistics.index = ["Organic Carbon" if column == "Organic_Carbon" else column for column in statistics.index]
    st.dataframe(statistics, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Nutrient distribution", "Read every soil signal", "Interactive distributions reveal spread without implying a prediction.")
    distribution_columns = st.columns(3)
    for column, feature in zip(distribution_columns, NUMERIC_FEATURES[:3]):
        with column:
            st.plotly_chart(nutrient_distribution_figure(processed_dataframe, feature), use_container_width=True, config={"displayModeBar": False})
    distribution_columns = st.columns(2)
    for column, feature in zip(distribution_columns, NUMERIC_FEATURES[3:]):
        with column:
            st.plotly_chart(nutrient_distribution_figure(processed_dataframe, feature), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Correlation analysis", "See relationships, not conclusions", "Correlation is descriptive only in Part 2 and does not establish causation.")
    st.plotly_chart(correlation_figure(processed_dataframe, [column for column in NUMERIC_FEATURES if column in processed_dataframe.columns]), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Fertility distribution", "Inspect the available target", "Labels are shown only when they exist in the source dataset.")
    if TARGET_COLUMN in processed_dataframe.columns:
        st.plotly_chart(class_distribution_figure(processed_dataframe), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Fertility labels are not available in the current dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Regional analytics", "Understand sample coverage", "Counts reflect the source file only; no regional statistics are fabricated.")
    if "Region" in processed_dataframe.columns:
        st.plotly_chart(regional_summary_figure(processed_dataframe), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Regional data is not available in the current dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Processing output", "A reusable ML-ready handoff", "The cleaned file and metadata are generated for Part 3 without training a model.")
    output_columns = st.columns(2)
    with output_columns[0]:
        st.markdown(f'<div class="about-panel"><div class="section-label">Processed dataset</div><div class="placeholder-copy">{Path(default_processed_path()).as_posix()}</div><div class="status-pill">{processing_report["rows_after"]} ROWS · {processing_report["columns_after"]} COLUMNS</div></div>', unsafe_allow_html=True)
    with output_columns[1]:
        st.markdown('<div class="about-panel"><div class="section-label">Feature selection</div><div class="placeholder-copy"><strong>Numerical:</strong> ' + ", ".join(processing_report["numeric_features"]) + '<br><strong>Categorical:</strong> ' + ", ".join(processing_report["categorical_features"]) + f'<br><strong>Target:</strong> {processing_report["target"] or "Not available"}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_home() -> None:
    st.markdown('<div class="hero"><div class="hero-grid"><div><div class="eyebrow">🌱 DataXcelerate 2026 · Problem Statement 22</div><h1 class="hero-title">Understand Your Soil.<br><span>Grow Smarter.</span></h1><p class="hero-copy">AI-powered soil intelligence for nutrient analysis, fertility prediction and data-driven agricultural decisions.</p></div><div class="hero-meta"><div class="hero-meta-title">The intelligence layer for modern farms</div><div class="hero-meta-value">From soil signals to smarter action.</div><div class="hero-meta-copy">A single, focused workspace for turning the hidden story beneath every field into an advantage.</div></div></div></div>', unsafe_allow_html=True)
    primary, secondary, _ = st.columns([1.15, 1.25, 5])
    with primary:
        if st.button("Analyze Soil  →", key="hero-analyze", use_container_width=True):
            go_to("soil-report")
            st.rerun()
    with secondary:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("Explore Insights", key="hero-insights", use_container_width=True):
            go_to("soil-health")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Your command centre", "A clear view of soil health")
    kpi_columns = st.columns(5)
    kpis = [("TOTAL SAMPLES", "0", "Awaiting first analysis", False), ("SOIL HEALTH", "-- / 100", "No score calculated", True), ("LOW FERTILITY", "--", "No classifications yet", False), ("MEDIUM FERTILITY", "--", "No classifications yet", False), ("HIGH FERTILITY", "--", "No classifications yet", False)]
    for column, (label, value, note, accent) in zip(kpi_columns, kpis):
        with column:
            render_kpi_card(label, value, note, accent)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Start here", "How would you like to analyze your soil?", "Choose the signal that is closest to the field today.")
    action_columns = st.columns(3)
    actions = [("▣", "SOIL REPORT", "Upload laboratory soil reports and analyze N, P, K, pH and Organic Carbon.", "Analyze Report  →", "soil-report", "action-report"), ("◉", "SOIL IMAGE", "Upload a soil image for preliminary visual assessment.", "Analyze Image  →", "soil-image", "action-image"), ("♧", "FARMER EXPERIENCE", "Describe field observations and receive a preliminary assessment.", "Start Assessment  →", "farmer-experience", "action-farmer")]
    for column, action in zip(action_columns, actions):
        with column:
            render_action_card(*action)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("Intelligence layer", "One platform. Complete soil intelligence.", "Designed to make complex agricultural signals easier to see, understand and act on.")
    feature_columns = st.columns(4)
    features = [("🧬", "Nutrient Intelligence", "Analyze N, P, K, pH and Organic Carbon."), ("🧠", "ML Fertility Prediction", "Predict soil fertility using supervised ML."), ("📊", "Regional Analytics", "Discover nutrient patterns across regions."), ("🔬", "What-If Simulation", "Explore how soil parameter changes affect model predictions.")]
    for column, feature in zip(feature_columns, features):
        with column:
            render_feature_card(*feature)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_section_header("The operating model", "How AgroSoil AI works", "A deliberate path from raw field context to confident agricultural decisions.")
    render_workflow()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="about-panel"><div class="eyebrow">BUILT FOR SMARTER AGRICULTURE</div><h2 class="section-title" style="margin-top:.65rem;">Make every field decision more informed.</h2><p class="placeholder-copy">AgroSoil AI brings soil data, machine intelligence and practical field context into one calm, decision-ready experience. Built for the next generation of resilient, measurable agriculture.</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_placeholder(page: str) -> None:
    page_content = {
        "soil-report": ("▣", "Soil Report Analysis", "Upload-ready workspace for laboratory soil reports and nutrient analysis.", "REPORT ANALYSIS · COMING IN PART 2"),
        "soil-image": ("◉", "Soil Image Assessment", "A future computer vision workspace for preliminary visual soil assessment.", "IMAGE INTELLIGENCE · COMING IN PART 2"),
        "farmer-experience": ("♧", "Farmer Experience", "A guided field observation flow for capturing the human context behind the soil data.", "FIELD EXPERIENCE · COMING IN PART 2"),
        "soil-health": ("◈", "Soil Health", "A focused health overview will surface nutrient signals, score context and the story behind each sample.", "SOIL HEALTH · MODULE PREVIEW"),
        "analytics": ("▦", "Analytics", "A visual analytics workspace for exploring soil patterns and model-ready insights.", "ANALYTICS · MODULE PREVIEW"),
        "regional-insights": ("⌖", "Regional Insights", "Compare regional soil patterns and discover where nutrient context changes the decision.", "REGIONAL INTELLIGENCE · MODULE PREVIEW"),
        "simulator": ("◇", "What-If Simulator", "Explore how changing soil parameters could influence future model predictions.", "SIMULATION · COMING IN PART 3"),
        "history": ("▤", "Analysis History", "A chronological home for completed soil analyses, observations and recommendations.", "HISTORY · COMING IN PART 2"),
        "model-insights": ("✦", "Model Insights", "Understand the signals, confidence and reasoning behind future AI predictions.", "MODEL TRANSPARENCY · COMING IN PART 2"),
        "about": ("ⓘ", "About AgroSoil AI", "AgroSoil AI is the DataXcelerate 2026 submission for PS22: Soil Health Analysis & Fertilizer Recommendation System.", "DATAXCELERATE 2026 · PS22"),
    }
    icon, title, copy, status = page_content.get(page, page_content["about"])
    st.markdown(f'<div class="hero"><div class="eyebrow">AGROSOIL AI · WORKSPACE</div><div class="placeholder-icon">{icon}</div><h1 class="placeholder-title">{title}</h1><p class="placeholder-copy">{copy}</p><div class="status-pill">{status}</div></div>', unsafe_allow_html=True)
    if page != "about":
        st.markdown('<div class="about-panel"><div class="section-label">Part 1 foundation</div><h2 class="section-title">The experience is ready for the intelligence layer.</h2><p class="placeholder-copy">This page is intentionally a polished placeholder. No simulated data, ML predictions or recommendations are shown before the underlying module is implemented.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section"><div class="about-panel"><div class="section-label">Mission</div><h2 class="section-title">Make soil intelligence understandable, useful and actionable.</h2><p class="placeholder-copy">The portal is designed as a premium foundation for nutrient analysis, fertility prediction, regional analytics and fertilizer recommendations. Part 1 focuses on a clear, credible product experience without inventing model results.</p></div></div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(get_css(), unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    render_sidebar()
    page = st.session_state["page"]
    if page == "home":
        render_home()
    elif page == "data-intelligence":
        render_data_intelligence()
    else:
        render_placeholder(page)


if __name__ == "__main__":
    main()
