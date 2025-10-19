# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(test_data_path: str, model_path: str) -> dict:
    model = joblib.load(model_path)
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
    parser.add_argument("model.pkl", required=True, help="Path to trained model")
    args = parser.parse_args()

    results = evaluate_model(args.test_data, args.model)
    print("Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
