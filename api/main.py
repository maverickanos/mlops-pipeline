import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from typing import List, Optional
import time

from api.config import api_config
from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
)
from monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_ERRORS,
    MODEL_VERSION_GAUGE,
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(
    title=api_config.app_title,
    description=api_config.app_description,
    version=api_config.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_version = "unknown"
model_run_id = None
model_metadata = {}


@app.on_event("startup")
async def load_model():
    global model, model_version, model_run_id, model_metadata

    mlflow.set_tracking_uri(api_config.model_registry_uri)

    try:
        if api_config.model_version:
            model_uri = f"models:/{api_config.model_name}/{api_config.model_version}"
        else:
            model_uri = f"models:/{api_config.model_name}/{api_config.model_stage}"

        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            api_config.model_name, stages=[api_config.model_stage]
        )
        if latest_versions:
            model_version = latest_versions[0].version
            model_run_id = latest_versions[0].run_id
        else:
            model_version = "latest"
            model_run_id = "unknown"

        MODEL_VERSION_GAUGE.set(float(model_version) if model_version.isdigit() else 0)

        scaler_path = "preprocessor.pkl"
        if os.path.exists(scaler_path):
            model_metadata["scaler_loaded"] = True
        else:
            model_metadata["scaler_loaded"] = False

        model_metadata["n_features"] = (
            model.n_features_in_ if hasattr(model, "n_features_in_") else 0
        )
        model_metadata["classes"] = (
            model.classes_.tolist() if hasattr(model, "classes_") else []
        )

        print(f"Model loaded: {api_config.model_name} v{model_version}")

    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        active_connections=0,
    )


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name=api_config.model_name,
        model_version=model_version,
        model_stage=api_config.model_stage,
        run_id=model_run_id,
        input_features=[f"feature_{i}" for i in range(model_metadata.get("n_features", 0))],
        num_classes=len(model_metadata.get("classes", [])),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0, 1])

        latency = time.time() - start_time

        PREDICTION_COUNT.labels(
            model_version=model_version, prediction_class=str(prediction)
        ).inc()
        PREDICTION_LATENCY.observe(latency)

        classes = model_metadata.get("classes", [0, 1])
        class_name = str(classes[prediction]) if prediction < len(classes) else str(prediction)

        return PredictionResponse(
            prediction=prediction,
            probability=proba,
            class_name=class_name,
            model_version=model_version,
            run_id=model_run_id or "unknown",
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features = np.array(request.instances)
        predictions = model.predict(features).tolist()
        probas = model.predict_proba(features)[:, 1].tolist()

        latency = time.time() - start_time

        PREDICTION_COUNT.labels(
            model_version=model_version, prediction_class="batch"
        ).inc(len(predictions))
        PREDICTION_LATENCY.observe(latency)

        classes = model_metadata.get("classes", [0, 1])
        class_names = [
            str(classes[p]) if p < len(classes) else str(p) for p in predictions
        ]

        return BatchPredictionResponse(
            predictions=[int(p) for p in predictions],
            probabilities=probas,
            class_names=class_names,
            model_version=model_version,
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=api_config.api_host,
        port=api_config.api_port,
        reload=True,
    )
