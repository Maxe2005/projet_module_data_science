import pandas as pd

from cleaning_utils import (  # remove_error_rows,
    build_error_report_json,
    delete_column,
    get_valid_distribution,
    impute_proportional,
    remove_error_rows,
)
from compare_classifiers import ClassifierComparison
from evaluation_utils import evaluate_model
from format_data_utils import binarize_target, encode_categorical, normalize_features
from model_persistence_utils import load_model
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
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


def clean_data(
    data: pd.DataFrame, cleaning_type: str = "remove", columns_to_remove: list = []
) -> pd.DataFrame:
    """Fonction de nettoyage des données. Supprime ou impute les lignes avec des erreurs selon le type spécifié.
    Args:
        data (pd.DataFrame): Le DataFrame à nettoyer.
        type (str, optional): "remove" supprimer les lignes avec erreurs, "distribution" pour imputer proportionnellement.
        columns_to_remove (list, optional): Liste des colonnes à supprimer après le nettoyage.
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    errors = build_error_report_json(data)
    data_cleaned: pd.DataFrame = pd.DataFrame()
    if cleaning_type == "remove":
        data_cleaned = remove_error_rows(data, errors)
    elif cleaning_type == "distribution":
        distribution = get_valid_distribution(data, errors)
        data_cleaned = impute_proportional(data, errors, distribution)
    else:
        raise ValueError(
            "Le paramètre 'type' doit être soit 'remove' soit 'distribution'."
        )
    for col in columns_to_remove:
        data_cleaned = delete_column(data_cleaned, col)
    return data_cleaned


def preprocess_data(data, target="outcome", path_out="car_insurance_formatted.csv"):
    data = clean_data(
        data,
        cleaning_type="remove",
        columns_to_remove=["id", "children", "education", "married"],
    )

    encode_categorical(data, inplace=True)
    normalize_features(data, inplace=True)
    binarize_target(data, target="outcome", inplace=True)
    write_data(data, path_out)
    return data


def simple_train_validate(data):
    print("\nStarting logistic regression training (simple split)...")
    model, X_test, y_test = train_validate_simple(
        data,
        target="outcome",
        test_size=0.2,
        random_state=42,
        model_out="models/logistic_regression_model.pkl",
    )
    print("Logistic regression training completed.\n")

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, show_samples=False, sample_limit=20)
    print("Model evaluation completed.")


def cross_validate(data):
    cross_validate_model(
        data,
        target="outcome",
        cv=5,
        model_out="models/logistic_regression_cv_best.pkl",
    )


def compare_classifiers(data_path):
    print("Comparing classifiers...")
    comparison = ClassifierComparison(
        data_path=data_path,
        target="outcome",
        test_size=0.2,
        random_state=42,
        cv_folds=5,
    )

    comparison.compare_all_classifiers()


def evaluate_saved_model(model_path, data, target="outcome"):
    print(f"Loading saved model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded.")

    X_test = data.drop(columns=[target])
    y_test = data[target]

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, show_samples=False, sample_limit=20)
    print("Model evaluation completed.")


def main():
    data = read_data(file_path)
    if data is None:
        return

    data = preprocess_data(data, path_out=file_path_out)

    # simple_train_validate(data)

    cross_validate(data)

    # compare_classifiers(file_path_out)


if __name__ == "__main__":
    file_path = "car_insurance.csv"
    file_path_out = "car_insurance_formatted.csv"
    main()
