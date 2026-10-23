import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from typing import Dict, Any, List, Optional
import json
import os

from pipeline.config import config
from pipeline.data_preprocessing import preprocess_data


def load_production_model():
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        latest_versions = client.get_latest_versions(config.model_name, stages=["Production"])
        if latest_versions:
            model_uri = f"models:/{config.model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            version = latest_versions[0].version
            run_id = latest_versions[0].run_id
            return model, version, run_id
    except Exception:
        pass

    model = mlflow.sklearn.load_model(f"models:/{config.model_name}/latest")
    return model, "latest", None


def compare_runs(
    experiment_name: str = None,
    metric_names: List[str] = None,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment_name = experiment_name or config.experiment_name

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if metric_names is None:
        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    param_cols = [c for c in runs.columns if c.startswith("params.")]
    metric_cols = [f"metrics.{m}" for m in metric_names if f"metrics.{m}" in runs.columns]

    cols = ["run_id", "start_time", "status"] + param_cols + metric_cols
    cols = [c for c in cols if c in runs.columns]

    return runs[cols]


def evaluate_model(
    model=None,
    model_uri: Optional[str] = None,
    output_dir: str = "evaluation_output",
) -> Dict[str, Any]:
    data = preprocess_data()
    X_test, y_test = data["X_test"], data["y_test"]

    if model is None and model_uri:
        model = mlflow.sklearn.load_model(model_uri)
    elif model is None:
        model, _, _ = load_production_model()

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=100)
    plt.close()

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"metrics": metrics, "output_dir": output_dir}


if __name__ == "__main__":
    print("=== Comparing runs ===")
    runs_df = compare_runs()
    print(runs_df.to_string())

    print("\n=== Evaluating production model ===")
    result = evaluate_model()
    print(f"Metrics: {result['metrics']}")
    print(f"Plots saved to: {result['output_dir']}")
