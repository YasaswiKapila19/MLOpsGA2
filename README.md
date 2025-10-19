evalu.py - evaluate_model function that returns various metrics such as precision, accuracy, recall, f1-score.
validate.py - validate_data function which runs a few tests such as checking for null values, species column, less than 10 rows etc. Raises a ValueError if it fails any of the test cases and returns True otherwise.
tests/test_eval.py - unit tests that check metrics such as accuracy, precision of the model retrieved from DVC.
tests/test_val.py - unit test that runs the validate_data function and fails if any of the cases written in the function have failed.
data/test.csv.dvc - data created for evaluation of the model. Stored using dvc in the gcp storage bucket.
iris.csv.dvc - data created for training the model. Stored using dvc in the gcp storage bucket.
.github/workflows/main.yml - Config file for the github action triggered during push/PR to auto run the unit tests.
inference.txt - Evaluation reports generated when training the model.
train.py - Pipeline for training the model on iris.csv. Uses a Random Forest Classifier.
