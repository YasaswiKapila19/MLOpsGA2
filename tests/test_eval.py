import pytest
from evalu import evaluate_model

def test_evaluate_model_output_type():
    metrics = evaluate_model("data/test.csv", "model.pkl")
    assert isinstance(metrics, dict), "Evaluation metrics should be a dictionary"

def test_accuracy_threshold():
    metrics = evaluate_model("data/test.csv", "model.pkl")
    assert metrics["accuracy"] > 0.7, "Accuracy should be above 70%"

