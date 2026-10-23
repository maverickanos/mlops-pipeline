import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app
from api.schemas import PredictionRequest, BatchPredictionRequest


client = TestClient(app)


class TestHealth:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_info_endpoint(self):
        response = client.get("/model-info")
        assert response.status_code in [200, 503]


class TestSchemas:
    def test_prediction_request_valid(self):
        req = PredictionRequest(features=[0.1] * 30)
        assert len(req.features) == 30

    def test_prediction_request_invalid(self):
        with pytest.raises(Exception):
            PredictionRequest()

    def test_batch_prediction_request(self):
        req = BatchPredictionRequest(instances=[[0.1] * 30, [0.2] * 30])
        assert len(req.instances) == 2


class TestPrediction:
    def test_predict_missing_model(self):
        response = client.post(
            "/predict",
            json={"features": [0.1] * 30},
        )
        assert response.status_code in [200, 503]

    def test_predict_invalid_features(self):
        response = client.post(
            "/predict",
            json={"features": [0.1] * 5},
        )
        assert response.status_code in [400, 503]

    def test_batch_predict(self):
        response = client.post(
            "/predict-batch",
            json={"instances": [[0.1] * 30, [0.2] * 30]},
        )
        assert response.status_code in [200, 503]


class TestMetrics:
    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
