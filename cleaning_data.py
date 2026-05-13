import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path: str) -> pd.DataFrame:
    """Lis le fichier CSV et retourne un DataFrame pandas.
    
    Args:
        file_path (str): Chemin vers le fichier CSV à lire.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données du fichier CSV.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data read successfully.")
        return data
    except Exception as e:
        print(f"Error reading data: {e}")
        return None
    

def histogram(data, column):
    """Génère un histogramme pour une colonne spécifique du DataFrame."""
    plt.hist(data[column], bins=20, color='blue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()



def detect_outliers(data: pd.DataFrame , column: str) -> tuple[pd.DataFrame, float, float]:
    """
    Detecte les outliers dans une colonne spécifique du DataFrame en utilisant la méthode IQR (Interquartile Range).

    Args:
        data (str): DataFrame pandas contenant les données.
        column (pd.DataFrame): Nom de la colonne à analyser pour les outliers.

    Returns:
        tuple[pd.DataFrame, str, float, float] : Un DataFrame contenant les lignes avec des outliers, le nom de la colonne analysée, et les bornes inférieure et supérieure pour les outliers.
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


def main():
    file_path = "car_insurance.csv"
    data = read_data(file_path)
    
    if data is not None:
        detect_outliers(data, 'driving_experience')

if __name__ == "__main__":    main()


