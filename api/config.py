import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    model_registry_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name: str = "breast_cancer_classifier"
    model_stage: str = "Production"
    model_version: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    app_title: str = "MLOps Production Pipeline API"
    app_description: str = "Production-grade ML model serving API with monitoring"
    app_version: str = "1.0.0"


api_config = APIConfig()
