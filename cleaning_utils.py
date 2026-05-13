import pandas as pd

# On a tester plusieurs encadrements pour l'IQR, 0.95 semble être un bon compromis pour notre dataset

encadrement = 0.95


def detect_outliers_by_column(
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
    Q1 = data[column].quantile(0.5 - encadrement / 2)
    Q3 = data[column].quantile(0.5 + encadrement / 2)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Outliers detected in {column}:")
    for i, row in outliers.iterrows():
        print(f"Ligne: {i}, id : {row['id']}, Value: {row[column]}")
    return outliers, lower_bound, upper_bound


def detect_outliers(data: pd.DataFrame) -> dict:
    """Detecte les outliers dans le DataFrame et retourne les IDs concernés.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.

    Returns:
        dict: Un dictionnaire mappant chaque colonne numérique à une liste d'IDs
              des lignes avec des outliers.
              Format: {colonne: [id1, id2, ...]}
    """
    if data is None:
        return {}

    id_column = "id" if "id" in data.columns else None
    outliers_by_column = {}

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            Q1 = data[column].quantile(0.5 - encadrement / 2)
            Q3 = data[column].quantile(0.5 + encadrement / 2)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_outliers = data[
                (data[column] < lower_bound) | (data[column] > upper_bound)
            ]
            if not column_outliers.empty:
                outliers_by_column[column] = (
                    column_outliers[id_column].tolist()
                    if id_column is not None
                    else column_outliers.index.tolist()
                )

    return outliers_by_column


def detect_missing_values(data: pd.DataFrame) -> dict:
    """Detecte les valeurs manquantes par colonne et retourne les IDs concernés.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.

    Returns:
        dict: Dictionnaire avec pour chaque colonne une liste d'IDs des lignes
              où la valeur est manquante.
              Format: {colonne: [id1, id2, ...]}
    """
    if data is None:
        return {}

    id_column = "id" if "id" in data.columns else None
    missing_by_column = {}

    for column in data.columns:
        missing_rows = data[data[column].isna()]
        if not missing_rows.empty:
            ids = (
                missing_rows[id_column].tolist()
                if id_column is not None
                else missing_rows.index.tolist()
            )
            missing_by_column[column] = ids

    return missing_by_column


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
