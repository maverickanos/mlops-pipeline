import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected_percentiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_percentiles[0] = -np.inf
    expected_percentiles[-1] = np.inf

    expected_counts = np.histogram(expected, bins=expected_percentiles)[0]
    actual_counts = np.histogram(actual, bins=expected_percentiles)[0]

    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)

    expected_prop = np.clip(expected_prop, 1e-6, 1)
    actual_prop = np.clip(actual_prop, 1e-6, 1)

    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return float(psi)


def detect_drift_ks(
    reference: np.ndarray, production: np.ndarray, alpha: float = 0.05
) -> Dict:
    statistic, p_value = ks_2samp(reference, production)
    return {
        "test": "KS Test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
        "alpha": alpha,
    }


def detect_drift_psi(
    reference: np.ndarray,
    production: np.ndarray,
    psi_threshold: float = 0.2,
) -> Dict:
    psi_value = compute_psi(reference, production)
    return {
        "test": "Population Stability Index (PSI)",
        "psi": psi_value,
        "drift_detected": bool(psi_value > psi_threshold),
        "threshold": psi_threshold,
        "severity": "high" if psi_value > 0.5 else ("medium" if psi_value > 0.25 else "low"),
    }


def detect_drift_categorical(
    reference: np.ndarray,
    production: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    ref_counts = pd.Series(reference).value_counts()
    prod_counts = pd.Series(production).value_counts()

    all_categories = list(set(ref_counts.index) | set(prod_counts.index))

    ref_table = [ref_counts.get(cat, 0) for cat in all_categories]
    prod_table = [prod_counts.get(cat, 0) for cat in all_categories]

    table = np.array([ref_table, prod_table])

    if table.min() == 0:
        table = table + 1

    chi2, p_value, dof, expected = chi2_contingency(table)

    return {
        "test": "Chi-Square Test",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
        "alpha": alpha,
    }


def run_drift_analysis(
    training_data: pd.DataFrame,
    production_data: pd.DataFrame,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
) -> Dict:
    if numerical_features is None:
        numerical_features = training_data.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_features is None:
        categorical_features = training_data.select_dtypes(include=["object", "category"]).columns.tolist()

    results = {"numerical": {}, "categorical": {}, "overall_drift": False}

    for feature in numerical_features:
        if feature not in production_data.columns:
            continue

        ref_values = training_data[feature].dropna().values
        prod_values = production_data[feature].dropna().values

        if len(ref_values) == 0 or len(prod_values) == 0:
            continue

        ks_result = detect_drift_ks(ref_values, prod_values)
        psi_result = detect_drift_psi(ref_values, prod_values)

        results["numerical"][feature] = {
            "ks_test": ks_result,
            "psi": psi_result,
        }

    for feature in categorical_features:
        if feature not in production_data.columns:
            continue

        ref_values = training_data[feature].dropna().values
        prod_values = production_data[feature].dropna().values

        if len(ref_values) == 0 or len(prod_values) == 0:
            continue

        chi2_result = detect_drift_categorical(ref_values, prod_values)
        results["categorical"][feature] = chi2_result

    all_feature_results = list(results["numerical"].values()) + list(results["categorical"].values())
    drift_flags = []

    for r in all_feature_results:
        if isinstance(r, dict):
            if "ks_test" in r:
                drift_flags.append(r["ks_test"]["drift_detected"])
            elif "drift_detected" in r:
                drift_flags.append(r["drift_detected"])

    results["overall_drift"] = any(drift_flags) if drift_flags else False
    results["drifted_features"] = sum(drift_flags) if drift_flags else 0
    results["total_features"] = len(drift_flags) if drift_flags else 0

    return results


if __name__ == "__main__":
    np.random.seed(42)
    ref_data = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, 1000),
        "feature_2": np.random.normal(5, 2, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })

    prod_data = pd.DataFrame({
        "feature_1": np.random.normal(0.5, 1.5, 500),
        "feature_2": np.random.normal(5, 2, 500),
        "category": np.random.choice(["A", "B", "C", "D"], 500),
    })

    results = run_drift_analysis(ref_data, prod_data)
    print(f"Overall drift detected: {results['overall_drift']}")
    print(f"Drifted features: {results['drifted_features']}/{results['total_features']}")
