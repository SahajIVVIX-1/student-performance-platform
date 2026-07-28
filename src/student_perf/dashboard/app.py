"""
Streamlit interactive dashboard for Student Performance Prediction & XAI.

Tabs:
1. Overview - Dataset statistics and target distribution
2. Model Leaderboard - Comparison table and performance chart
3. Single Prediction - Form for student features, API prediction, SHAP & LIME
4. Explainability Explorer - Deep dive into SHAP vs LIME for test samples
5. Experiment Tracking - MLflow runs history
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Student Performance Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_METRICS_PATH = _PROCESSED_DIR / "metrics_summary.csv"
API_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Caching data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_metrics():
    if _METRICS_PATH.exists():
        return pd.read_csv(_METRICS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_dataset():
    raw_path = _PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path, sep=";")
    return None


@st.cache_data
def fetch_api_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎓 Student Performance Prediction & Explainability Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Production-grade Machine Learning pipeline with SHAP & LIME XAI interpretations</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏆 Model Leaderboard",
    "🔮 Single Prediction",
    "🔍 Explainability Explorer",
    "📈 Experiment Tracking",
])


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab1:
    st.header("Dataset Overview")
    df_raw = load_dataset()
    if df_raw is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df_raw))
        col2.metric("Total Features", len(df_raw.columns) - 1)
        pass_count = (df_raw["G3"] >= 10).sum()
        fail_count = (df_raw["G3"] < 10).sum()
        col3.metric("Pass Rate (G3 >= 10)", f"{(pass_count / len(df_raw))*100:.1f}%")
        col4.metric("Average Final Grade", f"{df_raw['G3'].mean():.2f} / 20")

        st.subheader("Target Variable Distribution (G3 Final Grade)")
        fig = px.histogram(
            df_raw, x="G3", nbins=20,
            title="Distribution of G3 Final Grades",
            color_discrete_sequence=["#3B82F6"],
            labels={"G3": "Final Grade (0-20)"}
        )
        fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Pass Threshold (10)")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Study Time vs Final Grade")
            fig_study = px.box(df_raw, x="studytime", y="G3", color="studytime",
                               labels={"studytime": "Weekly Study Time Band", "G3": "Final Grade"})
            st.plotly_chart(fig_study, use_container_width=True)
        with col_b:
            st.subheader("Past Failures vs Final Grade")
            fig_fail = px.box(df_raw, x="failures", y="G3", color="failures",
                              labels={"failures": "Past Failures", "G3": "Final Grade"})
            st.plotly_chart(fig_fail, use_container_width=True)

        with st.expander("View Raw Sample Data"):
            st.dataframe(df_raw.head(20))
    else:
        st.warning("Raw dataset file not found in data/raw/student-mat.csv. Please run ingestion first.")


# ---------------------------------------------------------------------------
# Tab 2: Model Leaderboard
# ---------------------------------------------------------------------------
with tab2:
    st.header("Model Performance Leaderboard")
    df_metrics = load_metrics()

    if not df_metrics.empty:
        st.markdown("Comparison across all 8 benchmarked models + AutoML baseline.")
        
        # Format metrics table
        metrics_display = df_metrics.copy()
        if "error" in metrics_display.columns:
            metrics_display = metrics_display[metrics_display["error"].isna()]
        
        sort_col = "f1" if "f1" in metrics_display.columns else metrics_display.columns[1]
        metrics_display = metrics_display.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        metrics_display.index += 1

        st.dataframe(metrics_display, use_container_width=True)

        st.subheader("Model Benchmark Comparison (F1 vs ROC-AUC)")
        if "f1" in metrics_display.columns and "roc_auc" in metrics_display.columns:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=metrics_display["model"], y=metrics_display["f1"], name="F1 Score", marker_color="#2563EB"))
            fig_bar.add_trace(go.Bar(x=metrics_display["model"], y=metrics_display["roc_auc"], name="ROC-AUC", marker_color="#10B981"))
            fig_bar.update_layout(barmode="group", xaxis_title="Model", yaxis_title="Score", height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Training Time vs Inference Latency")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if "train_time_s" in metrics_display.columns:
                fig_tr = px.bar(metrics_display, x="model", y="train_time_s", title="Training Time (seconds)", color="train_time_s")
                st.plotly_chart(fig_tr, use_container_width=True)
        with col_t2:
            if "inference_latency_ms" in metrics_display.columns:
                fig_inf = px.bar(metrics_display, x="model", y="inference_latency_ms", title="Inference Latency per sample (ms)", color="inference_latency_ms")
                st.plotly_chart(fig_inf, use_container_width=True)
    else:
        st.info("No trained metrics summary found yet. Run model training pipeline to populate this leaderboard.")


# ---------------------------------------------------------------------------
# Tab 3: Single Prediction & Instant XAI
# ---------------------------------------------------------------------------
with tab3:
    st.header("Single Student Outcome Prediction & Explanation")
    st.markdown("Enter student attributes below to get real-time prediction and feature attribution via SHAP & LIME.")

    api_models = fetch_api_models()
    available_models = [m["name"] for m in api_models.get("models", [])] if api_models else ["random_forest", "xgboost", "logistic_regression", "mlp_torch"]
    selected_model = st.selectbox("Select Model for Inference", available_models)

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            G1 = st.slider("First Period Grade (G1)", 0, 20, 12)
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 13)
            studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x], index=1)
            failures = st.selectbox("Past Failures", options=[0, 1, 2, 3], index=0)
            absences = st.number_input("Absences", 0, 93, 2)
        with col2:
            age = st.slider("Age", 15, 22, 17)
            sex = st.selectbox("Sex", ["F", "M"])
            Medu = st.selectbox("Mother Education", [0, 1, 2, 3, 4], index=3)
            Fedu = st.selectbox("Father Education", [0, 1, 2, 3, 4], index=2)
            higher = st.selectbox("Wants Higher Education", ["yes", "no"], index=0)
        with col3:
            internet = st.selectbox("Internet Access", ["yes", "no"], index=0)
            schoolsup = st.selectbox("School Support", ["yes", "no"], index=1)
            famsup = st.selectbox("Family Support", ["yes", "no"], index=0)
            paid = st.selectbox("Paid Extra Classes", ["yes", "no"], index=1)
            romantic = st.selectbox("Romantic Relationship", ["yes", "no"], index=1)

        submit = st.form_submit_button("Predict & Explain")

    if submit:
        payload = {
            "features": {
                "school": "GP", "sex": sex, "age": age, "address": "U", "famsize": "GT3", "Pstatus": "T",
                "Medu": Medu, "Fedu": Fedu, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
                "traveltime": 1, "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
                "famsup": famsup, "paid": paid, "activities": "yes", "nursery": "yes", "higher": higher,
                "internet": internet, "romantic": romantic, "famrel": 4, "freetime": 3, "goout": 3,
                "Dalc": 1, "Walc": 1, "health": 5, "absences": absences, "G1": G1, "G2": G2
            },
            "model_name": selected_model
        }

        with st.spinner("Connecting to REST API for prediction and explanations..."):
            try:
                res_exp = requests.post(f"{API_URL}/explain", json=payload, timeout=10)
                if res_exp.status_code == 200:
                    exp_data = res_exp.json()
                    pred_label = "PASS (G3 >= 10)" if exp_data["prediction"] == 1 else "FAIL (G3 < 10)"
                    color = "#10B981" if exp_data["prediction"] == 1 else "#EF4444"
                    
                    st.markdown(f"### Prediction Result: <span style='color:{color}; font-weight:bold;'>{pred_label}</span>", unsafe_allow_html=True)
                    if exp_data.get("shap_lime_rank_correlation") is not None:
                        st.caption(f"SHAP vs LIME Spearman Rank Correlation: **{exp_data['shap_lime_rank_correlation']:.3f}**")

                    col_s, col_l = st.columns(2)
                    with col_s:
                        st.subheader("SHAP Feature Contributions")
                        shap_df = pd.DataFrame(exp_data["shap_contributions"])
                        if not shap_df.empty:
                            fig_shap = px.bar(
                                shap_df, x="weight", y="feature", orientation="h",
                                color="weight", color_continuous_scale="RdYlGn",
                                title="SHAP Values (Signed Impact)"
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                    with col_l:
                        st.subheader("LIME Feature Contributions")
                        lime_df = pd.DataFrame(exp_data["lime_contributions"])
                        if not lime_df.empty:
                            fig_lime = px.bar(
                                lime_df, x="weight", y="feature", orientation="h",
                                color="weight", color_continuous_scale="RdYlGn",
                                title="LIME Local Feature Weights"
                            )
                            st.plotly_chart(fig_lime, use_container_width=True)
                else:
                    st.error(f"API Error ({res_exp.status_code}): {res_exp.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server at {API_URL}. Make sure `uvicorn` API server is running! Error: {e}")


# ---------------------------------------------------------------------------
# Tab 4: Explainability Explorer
# ---------------------------------------------------------------------------
with tab4:
    st.header("Explainability Explorer (SHAP & LIME Static Artifacts)")
    st.markdown("Explore pre-generated global and local explanation plots across trained models.")

    shap_plots_dir = _PROJECT_ROOT / "data" / "processed" / "shap_plots"
    lime_dir = _PROJECT_ROOT / "data" / "processed" / "lime_outputs"

    if shap_plots_dir.exists():
        model_options = [f.name.replace("_shap_summary.png", "") for f in shap_plots_dir.glob("*_shap_summary.png")]
        if model_options:
            exp_model = st.selectbox("Select Model to Inspect", model_options)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                summary_img = shap_plots_dir / f"{exp_model}_shap_summary.png"
                if summary_img.exists():
                    st.subheader("Global SHAP Summary (Beeswarm)")
                    st.image(str(summary_img))
            with col_m2:
                imp_img = shap_plots_dir / f"{exp_model}_shap_importance.png"
                if imp_img.exists():
                    st.subheader("Global SHAP Feature Importance")
                    st.image(str(imp_img))

            st.subheader("Local Individual Student Explanations (Waterfall)")
            w_cols = st.columns(3)
            for i, archetype in enumerate(["high_performer", "borderline", "low_performer"]):
                waterfall_img = shap_plots_dir / f"{exp_model}_waterfall_{archetype}.png"
                with w_cols[i]:
                    st.caption(archetype.replace("_", " ").title())
                    if waterfall_img.exists():
                        st.image(str(waterfall_img))
                    else:
                        st.info(f"Waterfall plot for {archetype} not generated yet.")
        else:
            st.info("No SHAP plot artifacts found. Run model training pipeline first.")
    else:
        st.info("SHAP plots directory not found. Run model training pipeline first.")


# ---------------------------------------------------------------------------
# Tab 5: Experiment Tracking (MLflow)
# ---------------------------------------------------------------------------
with tab5:
    st.header("MLflow Experiment Runs History")
    try:
        from student_perf.tracking.mlflow_utils import get_all_runs
        runs_df = get_all_runs()
        if not runs_df.empty:
            st.dataframe(runs_df, use_container_width=True)
        else:
            st.info("No MLflow runs recorded in ./mlruns directory yet.")
    except Exception as e:
        st.warning(f"Could not load MLflow runs: {e}")
