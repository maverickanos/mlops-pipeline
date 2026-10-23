import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.data_preprocessing import load_dataset, preprocess_data
from pipeline.config import config
from pipeline.train import train_model
from pipeline.evaluate import evaluate_model, compare_runs


class TestDataPreprocessing:
    def test_load_dataset_breast_cancer(self):
        X, y = load_dataset("breast_cancer")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[1] == 30
        assert len(y) == 569

    def test_load_dataset_iris(self):
        X, y = load_dataset("iris")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[1] == 4
        assert len(y) == 150

    def test_preprocess_data(self):
        result = preprocess_data(save_preprocessor=False)
        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert "scaler" in result
        assert len(result["X_train"]) > len(result["X_test"])
        assert result["X_train"].shape[1] == 30

    def test_train_test_split_size(self):
        result = preprocess_data(test_size=0.3, random_state=42, save_preprocessor=False)
        total = len(result["X_train"]) + len(result["X_test"])
        assert abs(len(result["X_test"]) / total - 0.3) < 0.05


class TestTraining:
    def test_train_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = train_model()
                assert "run_id" in result
                assert "best_params" in result
                assert "metrics" in result
                assert "model_path" in result
                assert os.path.exists(result["model_path"])
                assert all(
                    m in result["metrics"]
                    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                )
            finally:
                os.chdir(orig_dir)


class TestEvaluation:
    def test_evaluate_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                train_result = train_model()
                eval_result = evaluate_model(
                    model_uri=f"runs:/{train_result['run_id']}/model",
                    output_dir=tmpdir,
                )
                assert "metrics" in eval_result
                assert all(
                    m in eval_result["metrics"]
                    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                )
                assert os.path.exists(os.path.join(tmpdir, "classification_report.json"))
                assert os.path.exists(os.path.join(tmpdir, "metrics.json"))
            finally:
                os.chdir(orig_dir)

    def test_compare_runs(self):
        df = compare_runs()
        assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
