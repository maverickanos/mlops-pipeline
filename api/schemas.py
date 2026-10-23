from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of feature values for prediction")


class BatchPredictionRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="List of feature vectors for batch prediction")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    class_name: str
    model_version: str
    run_id: str


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    class_names: List[str]
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    run_id: Optional[str] = None
    input_features: List[str]
    model_metrics: Optional[Dict[str, float]] = None
    num_classes: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    active_connections: int
