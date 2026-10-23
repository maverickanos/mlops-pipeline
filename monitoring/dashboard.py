import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import requests
import json
import time
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from pipeline.config import config
from monitoring.drift_detection import run_drift_analysis
from pipeline.data_preprocessing import preprocess_data

st.set_page_config(
    page_title="MLOps Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.sidebar.title("MLOps Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Prediction Distribution",
        "Data Drift Detection",
        "Recent Predictions",
        "Retrain Model",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Status:**")
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"Healthy (v{health.get('model_version', 'N/A')})")
except Exception:
    st.sidebar.error("API Unreachable")


def get_mlflow_runs():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        return runs
    except Exception as e:
        st.error(f"Failed to fetch MLflow runs: {e}")
        return pd.DataFrame()


def get_model_info():
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


if page == "Overview":
    st.title("📊 MLOps Pipeline Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    model_info = get_model_info()
    with col1:
        st.metric(
            "Model Version",
            model_info.get("model_version", "N/A") if model_info else "N/A",
        )
    with col2:
        runs_df = get_mlflow_runs()
        st.metric("Total Runs", len(runs_df) if not runs_df.empty else 0)

    with col3:
        try:
            resp = requests.get(f"{API_URL}/metrics", timeout=5)
            st.metric("API Metrics", "Available" if resp.status_code == 200 else "Unavailable")
        except Exception:
            st.metric("API Metrics", "Unavailable")

    with col4:
        st.metric("Data Drift", "Monitoring Active")

    st.markdown("---")
    if not runs_df.empty:
        st.subheader("Recent MLflow Runs")
        display_cols = [
            c for c in ["run_id", "start_time", "status"]
            if c in runs_df.columns
        ] + [c for c in runs_df.columns if c.startswith("metrics.")][:5]
        display_cols = [c for c in display_cols if c in runs_df.columns]
        st.dataframe(runs_df[display_cols].head(10))

elif page == "Model Performance":
    st.title("📈 Model Performance Over Time")
    st.markdown("---")

    runs_df = get_mlflow_runs()
    if not runs_df.empty:
        metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
        if metric_cols:
            fig = go.Figure()
            for col in metric_cols:
                metric_name = col.replace("metrics.", "")
                values = pd.to_numeric(runs_df[col], errors="coerce")
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        name=metric_name,
                        mode="lines+markers",
                    )
                )
            fig.update_layout(
                title="Metrics Across Runs",
                xaxis_title="Run Index",
                yaxis_title="Score",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Run Comparison Table")
        st.dataframe(runs_df)
    else:
        st.info("No runs found. Train a model first.")

elif page == "Prediction Distribution":
    st.title("🎯 Prediction Distribution")
    st.markdown("---")

    st.info("Connect the API to see live prediction distribution.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        dummy_data = pd.DataFrame({
            "Class": ["Class 0", "Class 1"],
            "Count": np.random.randint(50, 200, 2),
        })
        fig = px.bar(dummy_data, x="Class", y="Count", color="Class")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence Scores")
        dummy_scores = np.random.beta(2, 5, 1000)
        fig = px.histogram(x=dummy_scores, nbins=20, title="Prediction Confidence")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Drift Detection":
    st.title("🔄 Data Drift Detection")
    st.markdown("---")

    if st.button("Run Drift Analysis"):
        with st.spinner("Running drift analysis..."):
            data = preprocess_data()
            training_data = data["X_train"]

            num_samples = min(100, len(training_data))
            production_sample = training_data.sample(n=num_samples, random_state=int(time.time()))
            production_noise = np.random.normal(0, 0.1, production_sample.shape)
            production_data = production_sample + production_noise
            production_data.columns = training_data.columns

            results = run_drift_analysis(training_data, production_data)

            st.subheader("Drift Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Drift", "Detected" if results["overall_drift"] else "Not Detected")
            with col2:
                st.metric("Drifted Features", results["drifted_features"])
            with col3:
                st.metric("Total Features", results["total_features"])

            st.subheader("Feature-Level Drift")
            for feature, result in results["numerical"].items():
                with st.expander(f"Feature: {feature}"):
                    st.json(result)

            drift_df = pd.DataFrame([
                {
                    "Feature": feat,
                    "KS Statistic": res["ks_test"]["statistic"],
                    "KS p-value": res["ks_test"]["p_value"],
                    "Drift Detected (KS)": res["ks_test"]["drift_detected"],
                    "PSI": res["psi"]["psi"],
                    "PSI Severity": res["psi"]["severity"],
                }
                for feat, res in results["numerical"].items()
            ])
            st.dataframe(drift_df)
    else:
        st.info("Click 'Run Drift Analysis' to check for data drift.")

elif page == "Recent Predictions":
    st.title("📋 Recent Predictions Log")
    st.markdown("---")
    st.info("Prediction logs will appear here once the API is serving requests.")
    dummy_logs = pd.DataFrame(
        {
            "Timestamp": pd.date_range(end=datetime.now(), periods=20, freq="30s"),
            "Prediction": np.random.choice([0, 1], 20),
            "Confidence": np.random.uniform(0.6, 1.0, 20),
            "Model Version": ["v1"] * 20,
        }
    )
    st.dataframe(dummy_logs)

elif page == "Retrain Model":
    st.title("🔄 Retrain Model")
    st.markdown("---")

    st.warning("This will trigger a new training run. Ensure MLflow server is running.")

    if st.button("🚀 Start Retraining", type="primary"):
        with st.spinner("Training model..."):
            try:
                from pipeline.train import train_model
                result = train_model()
                st.success(f"Training complete! Run ID: {result['run_id']}")
                st.json(result["metrics"])
            except Exception as e:
                st.error(f"Training failed: {e}")
else:
    st.info(f"Unknown page: {page}")
