import sys
from pathlib import Path

import pandas as pd
from correlation_finder_utils import (
    correlations_matrix,
    plot_correlation_heatmap,
    plot_scatter_matrix,
    top_pairwise_correlations,
    top_target_correlations,
)


def explain_correlation():
    print("Signification: Un coefficient de corrélation (Pearson) varie de -1 à 1.")
    print("- 1 : corrélation positive parfaite, -1 : corrélation négative parfaite.")
    print("Valeurs proches de 0 indiquent peu ou pas de relation linéaire.")
    print()


def main(csv_path: str = "car_insurance_formatted.csv"):
    p = Path(csv_path)
    if not p.exists():
        print(f"Fichier {csv_path} introuvable.")
        sys.exit(1)

    df = pd.read_csv(p)

    explain_correlation()

    print("Calcul de la matrice de corrélation (numériques)...\n")
    corr = correlations_matrix(df)
    print(corr.round(3))

    print("\nTop paires de variables (corrélation absolue la plus élevée):")
    for a, b, val in top_pairwise_correlations(df, n=10):
        print(f"- {a} ↔ {b} : {val:.3f}")

    target = "outcome"
    if target in df.columns:
        print(f"\nCorrélations avec la variable de sortie '{target}':")
        t = top_target_correlations(df, target=target, n=20)
        print(t.round(3))
    else:
        print(
            "Colonne cible 'outcome' introuvable; impossible d'afficher corrélations cible."
        )

    # Sélectionner les variables les plus corrélées avec la cible pour visualiser
    if target in df.columns:
        top_features = list(top_target_correlations(df, target=target).head(6).index)
        print(f"\nVariables sélectionnées pour visualisation: {top_features}")
        plot_scatter_matrix(
            df,
            top_features + [target],
            figsize=(12, 12),
            save_path="correlations/scatter_matrix_selected.png",
        )
        print("Scatter matrix saved to scatter_matrix_selected.png")

    # heatmap
    plot_correlation_heatmap(df, save_path="correlations/correlation_heatmap.png")
    print("Heatmap saved to correlation_heatmap.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "car_insurance_formatted.csv"
    main(path)
