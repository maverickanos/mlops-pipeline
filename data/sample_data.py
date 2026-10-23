import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
import os


def generate_sample_data(dataset_name: str = "breast_cancer", output_dir: str = "."):
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["target_name"] = [target_names[t] for t in y]

    os.makedirs(output_dir, exist_ok=True)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_path = os.path.join(output_dir, f"{dataset_name}_train.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Generated {dataset_name} dataset:")
    print(f"  Training: {train_path} ({len(train_df)} samples)")
    print(f"  Test: {test_path} ({len(test_df)} samples)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {len(target_names)}")

    return train_path, test_path


if __name__ == "__main__":
    generate_sample_data("breast_cancer", ".")
    generate_sample_data("iris", ".")
