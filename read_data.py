import pandas as pd


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


def main():
    data = read_data("car_insurance.csv")

    if data is None:
        print("Failed to read the data.")
        return

    t = data.isna()
    print(t)


if __name__ == "__main__":
    main()
