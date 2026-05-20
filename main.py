import pandas as pd

from cleaning_utils import (  # remove_error_rows,
    build_error_report_json,
    delete_column,
    get_valid_distribution,
    impute_proportional,
)
from evaluation_utils import evaluate_model
from format_data_utils import binarize_target, encode_categorical, normalize_features
from training_supervisor import cross_validate_model, train_validate_simple


def read_data(file_path):
    """
    Reads data from a CSV file and returns a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def write_data(data, file_path):
    """
    Writes a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame to write to the CSV file.
    file_path (str): The path to the CSV file where the data will be written.

    Returns:
    None
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


def clean_data(data):
    errors = build_error_report_json(data)
    distribution = get_valid_distribution(data, errors)
    # data_cleaned = remove_error_rows(data, errors)
    data_cleaned = impute_proportional(data, errors, distribution)
    data_cleaned = delete_column(data_cleaned, "age", inplace=True)
    # data_cleaned = delete_column(data_cleaned, "speeding_violations", inplace=True)
    # data_cleaned = delete_column(data_cleaned, "past_accidents", inplace=True)
    # data_cleaned = delete_column(data_cleaned, "driving_experience", inplace=False)
    return data_cleaned


def main():
    data = read_data(file_path)
    if data is None:
        return

    data = clean_data(data)

    if data is None:
        return

    encode_categorical(data, inplace=True)
    normalize_features(data, inplace=True)
    binarize_target(data, target="outcome", inplace=True)
    write_data(data, "car_insurance_formatted.csv")

    print(
        "Data preprocessing completed. Formatted data saved to 'car_insurance_formatted.csv'."
    )

    print("\nStarting logistic regression training (simple split)...")
    model, X_test, y_test = train_validate_simple(
        data,
        target="outcome",
        test_size=0.2,
        random_state=42,
        model_out="models/logistic_regression_model.joblib",
    )
    print("Logistic regression training completed.\n")

    print("Running cross-validation (5-fold) to improve evaluation...")
    cv_scores = cross_validate_model(data, target="outcome", cv=5)
    print(f"Cross-val mean: {cv_scores.mean():.4f}, std: {cv_scores.std():.4f}\n")

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, show_samples=False, sample_limit=20)
    print("Model evaluation completed.")


if __name__ == "__main__":
    file_path = "car_insurance.csv"
    main()
