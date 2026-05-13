import pandas as pd
from matplotlib import pyplot as plt

from cleaning_utils import detect_missing_values, detect_outliers


def histogram(data, column):
    """Génère un histogramme pour une colonne spécifique du DataFrame."""
    plt.hist(data[column], bins=20, color="blue", edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def overview_outliers_and_missing_data(data):
    if data is not None:
        outliers_by_column = detect_outliers(data)
        missing_values = detect_missing_values(data)

        # Calcul des statistiques globales
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)
        total_outliers = sum(len(df) for df in outliers_by_column.values())
        total_missing = missing_values

        # Affichage du header
        print("\n" + "=" * 90)
        print("📊 RAPPORT D'ANALYSE - OUTLIERS ET VALEURS MANQUANTES")
        print("=" * 90)

        # Statistiques globales
        print("\n📈 STATISTIQUES GLOBALES:")
        print(f"  • Nombre de lignes: {total_rows}")
        print(f"  • Nombre de colonnes: {len(data.columns)}")
        print(f"  • Total de cellules: {total_cells}")
        print(
            f"  • Outliers détectés: {total_outliers} ({100*total_outliers/total_cells:.2f}%)"
        )
        print(
            f"  • Valeurs manquantes: {total_missing} ({100*total_missing/total_cells:.2f}%)"
        )

        # Synthèse par colonne
        print("\n" + "-" * 90)
        print("📋 SYNTHÈSE PAR COLONNE:")
        print("-" * 90)
        print(
            f"{'Colonne':<20} {'Type':<12} {'Outliers':<12} {'Manquantes':<12} {'% Anomalies':<15}"
        )
        print("-" * 90)

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                nb_outliers = len(outliers_by_column.get(column, pd.DataFrame()))
                nb_missing = data[column].isna().sum()
                total_anomalies = nb_outliers + nb_missing
                pct_anomalies = (total_anomalies / total_rows) * 100
                col_type = "Numérique"
            else:
                nb_outliers = 0
                nb_missing = data[column].isna().sum()
                total_anomalies = nb_missing
                pct_anomalies = (total_anomalies / total_rows) * 100
                col_type = "string"

            print(
                f"{column:<20} {col_type:<12} {nb_outliers:<12} {nb_missing:<12} {pct_anomalies:>13.2f}%"
            )

        print("-" * 90)
        print()
    else:
        print("❌ Erreur: Données manquantes, impossible de générer le rapport.")


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
