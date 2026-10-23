import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Dict, Any

from pipeline.config import config


def load_dataset(dataset_name: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    dataset_name = dataset_name or config.dataset
    if dataset_name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
    elif dataset_name == "iris":
        data = load_iris(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = data.data
    y = data.target
    return X, y


def create_preprocessor(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, k: int = 10
) -> Tuple[pd.DataFrame, SelectKBest]:
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    return X_train_selected, selector, selected_features


def preprocess_data(
    dataset_name: str = None,
    test_size: float = None,
    random_state: int = None,
    save_preprocessor: bool = True,
) -> Dict[str, Any]:
    X, y = load_dataset(dataset_name)
    test_size = test_size or config.test_size
    random_state = random_state or config.random_state

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = create_preprocessor(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    _, selector, selected_features = select_features(X_train_scaled, y_train, k=X_train.shape[1])

    if save_preprocessor:
        joblib.dump(scaler, "preprocessor.pkl")
        joblib.dump(selector, "selector.pkl")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_features,
        "feature_names": X.columns.tolist(),
    }
