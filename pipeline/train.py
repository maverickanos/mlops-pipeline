import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib
import json
import os
from typing import Dict, Any

from pipeline.config import config
from pipeline.data_preprocessing import preprocess_data


def train_model() -> Dict[str, Any]:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    data = preprocess_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", config.dataset)

        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("cv_folds", config.cv_folds)
        mlflow.log_param("scoring", config.scoring)

        rf = RandomForestClassifier(random_state=config.random_state)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=config.model_params,
            cv=config.cv_folds,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_dict(
            {"confusion_matrix": cm.tolist(), "classification_report": classification_report(y_test, y_pred, output_dict=True)},
            "evaluation_results.json",
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=config.model_name,
        )

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        with open("cv_results.json", "w") as f:
            json.dump(
                {
                    "mean_test_score": cv_results["mean_test_score"].tolist(),
                    "std_test_score": cv_results["std_test_score"].tolist(),
                    "params": [
                        str(p) for p in cv_results["params"]
                    ],
                },
                f,
                indent=2,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/model.pkl")

    return {
        "run_id": run_id,
        "best_params": best_params,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        },
        "model_path": "models/model.pkl",
        "artifact_uri": artifact_uri,
    }


if __name__ == "__main__":
    result = train_model()
    print(f"Training complete!")
    print(f"Run ID: {result['run_id']}")
    print(f"Best params: {result['best_params']}")
    print(f"Metrics: {result['metrics']}")
