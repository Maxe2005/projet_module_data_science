import pandas as pd

from format_data_utils import encode_categorical, normalize_features


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


def main():
    data = read_data(file_path)
    if data is None:
        return

    encode_categorical(data, inplace=True)
    normalize_features(data, inplace=True)
    write_data(data, "car_insurance_formatted.csv")


if __name__ == "__main__":
    file_path = "car_insurance.csv"
    main()
