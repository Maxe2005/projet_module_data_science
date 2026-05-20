import numpy as np
import pandas as pd

encadrement = 0.95


def detect_outliers_by_column(
    data: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, float, float]:
    """
    Detecte les outliers dans une colonne spécifique du
    DataFrame en utilisant la méthode IQR (Interquartile Range).

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.
        column (str): Nom de la colonne à analyser pour les outliers.

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
    """Detecte les outliers dans le DataFrame et retourne les indices concernés.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.

    Returns:
        dict: Un dictionnaire mappant chaque colonne numérique à une liste d'indices
              des lignes avec des outliers.
              Format: {colonne: [index1, index2, ...]}
    """
    if data is None:
        return {}

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
                outliers_by_column[column] = column_outliers.index.tolist()

    return outliers_by_column


def detect_missing_values(data: pd.DataFrame) -> dict:
    """Detecte les valeurs manquantes par colonne et retourne les indices concernés.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.

    Returns:
        dict: Dictionnaire avec pour chaque colonne une liste d'indices des lignes
              où la valeur est manquante.
              Format: {colonne: [index1, index2, ...]}
    """
    if data is None:
        return {}

    missing_by_column = {}

    for column in data.columns:
        missing_rows = data[data[column].isna()]
        if not missing_rows.empty:
            missing_by_column[column] = missing_rows.index.tolist()

    return missing_by_column


def build_error_report_json(data: pd.DataFrame) -> dict:
    """Construit un rapport des valeurs erronées dans le DataFrame.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.

    Returns:
        dict: Dictionnaire contenant les indices erronés par colonne.
              Format: {colonne: [index1, index2, ...]}
    """
    outliers = detect_outliers(data)
    missing = detect_missing_values(data)
    merged = {}

    for column, ids in outliers.items():
        merged[column] = list(ids)
    for column, ids in missing.items():
        if column not in merged:
            merged[column] = []
        merged[column].extend(ids)

    # Conserver des indices uniques en préservant l'ordre d'apparition
    for column in merged:
        merged[column] = list(dict.fromkeys(merged[column]))
    return merged


def delete_column(data: pd.DataFrame, column_name: str, inplace: bool = False):
    if not inplace:
        data = data.copy()
    data = data.drop(columns=[column_name])
    return data


def delete_lines_with_ids(data: pd.DataFrame, ids: list):
    """
    Deletes lines from the DataFrame based on their indices.

    Args:
        data (pd.DataFrame): The DataFrame from which to delete lines.
        ids (list): A list of indices of the lines to delete.

    Returns:
        pd.DataFrame: The DataFrame with the specified lines deleted.
    """
    if data is not None:
        data = data[~data.index.isin(ids)]
        return data
    else:
        print("Data is None, cannot delete lines with ID.")
        return None


def remove_error_rows(data: pd.DataFrame, errors: dict) -> pd.DataFrame:
    """Supprime les lignes marquées comme erronées dans le DataFrame.

    Args:
        data (pd.DataFrame): DataFrame pandas contenant les données.
        errors (dict): Dictionnaire au format {colonne: [index1, index2, ...]} indiquant
                       les indices de lignes à supprimer.

    Returns:
        pd.DataFrame: Une copie du DataFrame sans les lignes erronées.
    """
    if data is None:
        print("Data is None, cannot remove error rows.")
        return None

    # Collecte unique des indices d'erreurs pour supprimer toutes les lignes concernées.
    error_indices = set()
    for indices in errors.values():
        if indices is None:
            continue
        error_indices.update(indices)

    if not error_indices:
        return data.copy()

    cleaned_data = data[~data.index.isin(error_indices)].copy()
    return cleaned_data


def get_valid_distribution(data: pd.DataFrame, errors: dict) -> dict:
    """
    Récupère la liste des valeurs valides pour chaque colonne contenant des erreurs.

    La distribution est encodée implicitement dans la liste retournée :
    si une valeur apparaît N fois sur 100 valeurs valides, elle aura
    naturellement N% de chances d'être tirée lors de l'imputation.

    Args:
        data (pd.DataFrame):  DataFrame contenant toutes les données.
        errors (dict) : dict au format {colonne: [index1, index2, ...]} listant les indices des valeurs erronées (manquantes ou outliers) par colonne.

    Returns:
        dict au format {colonne: [val1, val2, ...]} contenant uniquement
        les colonnes présentes dans errors.
    """
    distribution = {}

    for col, error_indices in errors.items():
        # Vérification que la colonne existe bien dans le DataFrame
        if col not in data.columns:
            print(
                f"[AVERTISSEMENT] La colonne '{col}' est absente du DataFrame — ignorée."
            )
            continue

        # Exclusion des indices erronés pour ne garder que les valeurs valides
        valid_mask = ~data.index.isin(error_indices)
        valid_values = data.loc[valid_mask, col]

        # Exclusion supplémentaire des NaN résiduels non déclarés dans errors
        valid_values = valid_values.dropna()

        if valid_values.empty:
            print(
                f"[AVERTISSEMENT] Aucune valeur valide trouvée pour la colonne '{col}' — ignorée."
            )
            continue

        # Stockage de la liste brute : la proportion est implicite dans la fréquence d'apparition
        distribution[col] = valid_values.tolist()
    return distribution


def impute_proportional(
    data: pd.DataFrame, errors: dict, repartition: dict
) -> pd.DataFrame:
    """
    Remplace les valeurs erronées par des valeurs tirées aléatoirement
    depuis la distribution empirique des valeurs valides.

    Le tirage est uniforme dans la liste de référence, ce qui garantit
    un respect automatique des proportions observées dans les données valides.

    Args:
        data (pd.DataFrame): DataFrame original contenant les données à corriger.
        errors (dict): dict au format {colonne: [index1, index2, ...]} listant les indices des valeurs à remplacer.
        repartition (dict): dict retourné par get_valid_distribution(), au format {colonne: [val1, val2, ...]}.

    Returns:
        DataFrame corrigé (copie de data, l'original n'est pas modifié).
    """
    corrected_data = data.copy()

    for col, error_indices in errors.items():
        # Vérification que la colonne existe dans le DataFrame
        if col not in corrected_data.columns:
            print(
                f"[AVERTISSEMENT] La colonne '{col}' est absente du DataFrame — ignorée."
            )
            continue
        # Vérification que la distribution de référence est disponible pour cette colonne
        if col not in repartition or len(repartition[col]) == 0:
            print(
                f"[AVERTISSEMENT] Aucune distribution disponible pour '{col}' — colonne ignorée."
            )
            continue

        # Filtrage des indices qui existent réellement dans le DataFrame
        valid_indices = [idx for idx in error_indices if idx in corrected_data.index]

        if not valid_indices:
            print(f"[INFO] Aucun indice valide à corriger pour la colonne '{col}'.")
            continue

        # Tirage aléatoire uniforme dans la liste de référence
        # La proportionnalité est garantie par la fréquence d'apparition dans la liste
        nb_to_replace = len(valid_indices)
        sampled_values = np.random.choice(
            repartition[col], size=nb_to_replace, replace=True
        )

        # Remplacement des valeurs erronées dans le DataFrame corrigé
        corrected_data.loc[valid_indices, col] = sampled_values

    return corrected_data
