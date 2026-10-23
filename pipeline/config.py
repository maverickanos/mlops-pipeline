import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingConfig:
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    })
    test_size: float = 0.2
    random_state: int = 42
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = "mlops-production-pipeline"
    model_name: str = "breast_cancer_classifier"
    cv_folds: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1
    dataset: str = "breast_cancer"
    target_column: str = "target"


config = TrainingConfig()
