"""Utilitaires de persistance pour les modèles entraînés.

Les objets scikit-learn sont sérialisables avec `pickle`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_model(
    model: Any,
    file_path: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """Sauvegarde un modèle entraîné dans un fichier `pickle`.

    Parameters
    ----------
    model : Any
        Modèle entraîné à sérialiser, par exemple `LogisticRegression`,
        `Perceptron` ou `KNeighborsClassifier`.
    file_path : str | Path
        Chemin du fichier de sortie.
    protocol : int, default=pickle.HIGHEST_PROTOCOL
        Version du protocole `pickle` à utiliser.

    Returns
    -------
    Path
        Chemin du fichier créé.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(model, file_handle, protocol=protocol)
    return path


def load_model(
    file_path: str | Path,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers=None,
) -> Any:
    """Charge un modèle entraîné depuis un fichier `pickle`.

    Parameters
    ----------
    file_path : str | Path
        Chemin du fichier `pickle` à charger.
    fix_imports : bool, default=True
        Conserve la compatibilité avec les anciens noms de modules Python 2.
    encoding : str, default='ASCII'
        Encodage utilisé pour les objets sérialisés en Python 2.
    errors : str, default='strict'
        Politique de gestion des erreurs d'encodage.
    buffers : sequence, optional
        Tampons externes transmis à `pickle.load`.

    Returns
    -------
    Any
        Objet désérialisé.
    """
    path = Path(file_path)
    with path.open("rb") as file_handle:
        return pickle.load(
            file_handle,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        )
