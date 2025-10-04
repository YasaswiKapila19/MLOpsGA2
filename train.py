import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

def main():
    # Load CSV from same folder
    file_path = os.path.join(os.path.dirname(__file__), 'iris.csv')
    df = pd.read_csv(file_path)

    # Show the first few rows (optional)
    print("Dataset preview:")
    print(df.head())

    # Separate features and labels
    X = df.drop('species', axis=1)  # Assuming 'species' is the target column
    y = df['species']

    # Split into train and eval (test) sets
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Run inference on the eval set
    y_pred = clf.predict(X_eval)

    # Print evaluation results
    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_eval, y_pred))
    print("Classification Report:")
    print(classification_report(y_eval, y_pred))

if __name__ == '__main__':
    main()
