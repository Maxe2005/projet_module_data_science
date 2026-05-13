import pandas as pd


def detect_outliers(
    data: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, float, float]:
    """
    Detecte les outliers dans une colonne spécifique du
    DataFrame en utilisant la méthode IQR (Interquartile Range).

    Args:
        data (str): DataFrame pandas contenant les données.
        column (pd.DataFrame): Nom de la colonne à analyser pour les outliers.

    Returns:
        tuple[pd.DataFrame, str, float, float] :
        Un DataFrame contenant les lignes avec des outliers,
        le nom de la colonne analysée,
        et les bornes inférieure et supérieure pour les outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Outliers detected in {column}:")
    for i, row in outliers.iterrows():
        print(f"Ligne: {i}, id : {row["id"]}, Value: {row[column]}")
    return outliers, lower_bound, upper_bound


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
