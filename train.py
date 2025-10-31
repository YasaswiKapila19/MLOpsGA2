import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
import os
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
import joblib
import os

mlflow.set_tracking_uri("http://127.0.0.1:8100") 
EXPERIMENT_NAME = "iris_rf_tuning_V2"
REGISTERED_MODEL_NAME = "Iris_RF_V3"


def main():
    file_path = os.path.join(os.path.dirname(__file__), 'iris.csv')
    df = pd.read_csv(file_path)
    print("Dataset preview:")
    print(df.head())

    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print(client.search_experiments())
    
    param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 4, None],
    "max_features": ["sqrt", "log2", None]
    }

    best_val = -1.0
    best_run_id = None
    best_params = None
    
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        with mlflow.start_run():
            mlflow.log_params(params)
            clf = RandomForestClassifier(random_state=42, **params)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            mlflow.log_metric("val_accuracy", acc)
            prec = precision_score(y_eval,preds, average = "weighted")
            f1 = f1_score(y_eval,preds, average = "weighted")
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(clf, name="model")
            run_id = mlflow.active_run().info.run_id
            if acc > best_val:
                best_val = acc
                best_run_id = run_id
                best_params = params
                result = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
                print("Registered model version:", result.version)

    print("Best validation accuracy:", best_val)
    print("Best params:", best_params)
    
    
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)

#     # Run inference on the eval set
#     y_pred = clf.predict(X_eval)

#     # Print evaluation results
#     print("\nEvaluation Results:")
#     print("Accuracy:", accuracy_score(y_eval, y_pred))
#     print("Classification Report:")
#     print(classification_report(y_eval, y_pred))

#     # Save the trained model
#     model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
#     joblib.dump(clf, model_path)
#     print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
