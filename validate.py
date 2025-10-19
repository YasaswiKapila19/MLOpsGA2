rc/validate.py

import pandas as pd

def validate_data(csv_path: str) -> bool:
    Returns:
        bool: True if validation passes, otherwise raises ValueError.

    df = pd.read_csv(csv_path)

    # Check for missing values
    if df.isnull().sum().any():
        raise ValueError("Data contains missing values.")

    # Check for 'target' column existence
    if "species" not in df.columns:
        raise ValueError("Missing 'species' column in dataset.")

    # Check for sufficient samples
    if len(df) < 10:
        raise ValueError("Dataset is too small to proceed.")

    print("Data validation passed.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate input data for Iris model")
    parser.add_argument("test.csv", required=True, help="Path to input CSV")
    args = parser.parse_args()

    validate_data(args.data)
