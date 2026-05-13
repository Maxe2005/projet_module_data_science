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


def overview_missing_data(data, print_each_line=False):
    if data is not None:
        missing_values = data.isna()
        nb_missing_values = missing_values.sum().sum()
        print(f"Number of missing values: {nb_missing_values}")
        nb_lines_with_missing_values = missing_values.any(axis=1).sum()
        print(f"Number of lines with missing values: {nb_lines_with_missing_values}")
        nb_missing_values_per_column = missing_values.sum()
        print("\nNumber of missing values per column:")
        print(nb_missing_values_per_column)

        lines_with_more_than_one_missing_value = missing_values.sum(axis=1) > 1
        print("\n")
        if print_each_line:
            print("Lines with more than one missing value:")
        count = 0
        for index, row in missing_values.iterrows():
            if lines_with_more_than_one_missing_value.at[index]:
                if print_each_line:
                    missing_columns = row[row].index.tolist()
                    print(
                        f"Line {index} has missing values in columns: {missing_columns}"
                    )
                count += 1
        print(f"Number of lines with more than one missing value: {count}")
    else:
        print("Data is None, cannot overview missing data.")


def delete_column(data, column_name):
    if data is not None:
        data = data.drop(columns=[column_name])
        return data
    else:
        print("Data is None, cannot delete column.")
        return None


def delete_lines_with_ids(data, ids):
    """
    Deletes lines from the DataFrame based on their IDs.

    Parameters:
    data (pd.DataFrame): The DataFrame from which to delete lines.
    ids (list): A list of IDs of the lines to delete.

    Returns:
    pd.DataFrame: The DataFrame with the specified lines deleted.
    """
    if data is not None:
        data = data[~data.index.isin(ids)]
        return data
    else:
        print("Data is None, cannot delete lines with ID.")
        return None


def main():
    data = read_data("car_insurance.csv")

    if data is None:
        print("Failed to read the data.")
        return

    overview_missing_data(data)


if __name__ == "__main__":
    main()
