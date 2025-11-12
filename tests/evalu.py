# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def evaluate_model(test_data_path: str) -> dict:

    REGISTERED_MODEL_NAME = "Iris_RF_V3"
    mlflow.set_tracking_uri("http://34.41.163.255:8100/")
    client = MlflowClient()
    
    latest_version_info = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[])  # all stages
    latest_version = sorted(latest_version_info, key=lambda x: int(x.version))[-1]
    print(f"Using latest model version: {latest_version.version}")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version.version}"
    model = mlflow.sklearn.load_model(model_uri)

    data = pd.read_csv(test_data_path)

    X_test = data.drop(columns=["species"])
    y_test = data["species"]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the Iris model")
    parser.add_argument("test.csv", required=True, help="Path to test CSV")
    args = parser.parse_args()

    results = evaluate_model(args.test_data)
    print("Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
