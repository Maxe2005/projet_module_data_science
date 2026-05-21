"""Utilitaires pour l'évaluation d'un modèle scikit-learn.

Fournit une fonction `evaluate_model(model, X_test, y_test, ...)`
qui effectue :
- prédictions sur `X_test`
- affichage par échantillon (optionnel)
- calcul et affichage des métriques : accuracy, confusion matrix,
  precision, recall, f1, classification report

Retourne un dictionnaire contenant les métriques calculées.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    show_samples: bool = True,
    sample_limit: int = 20,
    confusion_matrix_image_path: str | None = "confusion_matrix.png",
) -> Dict[str, Any]:
    """Évalue un modèle scikit-learn sur un jeu de test.

    Args:
        model: modèle entraîné possédant la méthode `predict`.
        X_test: features du jeu de test.
        y_test: labels réels du jeu de test.
        show_samples: si True, affiche pour chaque échantillon la prédiction
            et la classe réelle (limité à `sample_limit`).
        sample_limit: nombre maximal d'échantillons à afficher.
        confusion_matrix_image_path: chemin de sortie pour l'image de la
            matrice de confusion. Si `None`, aucun fichier n'est sauvegardé.

    Returns:
        Un dictionnaire contenant les métriques calculées.
    """
    # Prédictions
    y_pred = model.predict(X_test)

    # Affichage par échantillon (limité)
    if show_samples:
        n_print = min(len(y_pred), sample_limit)
        print(f"Affichage des {n_print} premiers échantillons (prédit / réel):")
        for i in range(n_print):
            print(f"  idx={i}: prédit={y_pred[i]}  réel={y_test[i]}")

    # Détection du type de problème (binaire vs multi-classe)
    unique_labels = np.unique(y_test)
    if unique_labels.shape[0] <= 2:
        average = "binary"
    else:
        average = "weighted"

    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Affichage résumé
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"Precision ({average}): {precision:.4f}")
    print(f"Recall ({average}): {recall:.4f}")
    print(f"F1 ({average}): {f1:.4f}")
    print("Classification report:")
    print(report)

    if confusion_matrix_image_path is not None:
        output_path = Path(confusion_matrix_image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            ax=ax, cmap="Blues", colorbar=True, values_format="d"
        )
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix image saved to: {output_path}")

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
    }

    return metrics
