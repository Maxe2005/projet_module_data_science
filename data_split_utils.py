from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["split_train_test"]


def split_train_test(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sépare un DataFrame en jeux d'entraînement et de test.

    Paramètres
    - df: DataFrame Pandas complet contenant les colonnes features et la colonne cible.
    - target: nom de la colonne cible dans `df`.
    - test_size: proportion du jeu de test (ex: 0.2 pour 20%).
    - random_state: graine pour la reproductibilité.
    - stratify: si True, on stratifie la séparation sur la cible lorsque possible.

    Retour
    Renvoie un tuple `(X_train, X_test, y_train, y_test)` où chaque élément
    est un ndarray NumPy prêt à être utilisé avec Scikit-Learn.
    """
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable dans le DataFrame.")

    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()

    stratify_param = df[target] if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test
