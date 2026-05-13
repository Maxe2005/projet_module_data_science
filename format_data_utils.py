from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    bool_map: Optional[Dict] = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Transforme les variables qualitatives en variables numériques.

    - Les colonnes de type bool sont converties en 0/1 (ou via `bool_map`).
    - Les autres colonnes catégorielles (`object`/`category`) sont encodées
      avec `LabelEncoder`.

    Paramètres:
    - df: DataFrame à transformer.
    - cols: liste optionnelle de colonnes à encoder. Si `None`, toutes les
      colonnes `object` et `category` seront traitées (les `bool` sont gérées
      séparément).
    - bool_map: mapping optionnel pour remplacer les valeurs booléennes
      (par ex. {True:1, False:0} ou {'yes':1,'no':0}). Si `None`, on utilise
      la conversion `astype(int)` pour les colonnes bool.
    - inplace: si True, modifie `df` en place et le retourne également.

    Retourne:
    - df_transformed: DataFrame transformé
    - encoders: dictionnaire {col: LabelEncoder} pour les colonnes encodées
    """
    if not inplace:
        df = df.copy()

    encoders: Dict[str, LabelEncoder] = {}

    # Colonnes de type bool
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        if bool_map is not None:
            df[c] = df[c].replace(bool_map)
        else:
            df[c] = df[c].astype(int)

    # Déterminer les colonnes catégorielles à encoder
    if cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols_to_encode = [c for c in cat_cols if c not in bool_cols]
    else:
        cols_to_encode = [c for c in cols if c in df.columns and c not in bool_cols]

    # Encoder avec LabelEncoder
    for c in cols_to_encode:
        le = LabelEncoder()
        # Convertir en string pour éviter les problèmes avec NaN
        series = df[c].astype(str).fillna("__missing__")
        df[c] = pd.Series(le.fit_transform(series), index=df.index)
        encoders[c] = le

    return df, encoders


__all__ = ["encode_categorical"]


def normalize_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalise les variables numériques avec `StandardScaler`.

    - Convertit les colonnes numériques sélectionnées en tableau NumPy,
      applique `StandardScaler` puis remet les valeurs dans le DataFrame.

    Paramètres:
    - df: DataFrame d'entrée.
    - cols: liste optionnelle de colonnes numériques à normaliser. Si `None`,
      toutes les colonnes numériques seront normalisées.
    - scaler: instance de `StandardScaler` à utiliser pour la transformation.
      Si `None`, une nouvelle instance sera créée et ajustée sur les données.
    - inplace: si True, modifie `df` en place.

    Retourne:
    - df_transformed: DataFrame avec colonnes normalisées
    - scaler: l'instance `StandardScaler` (fittée)
    """
    if not inplace:
        df = df.copy()

    # Colonnes numériques candidates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is None:
        cols_to_scale = numeric_cols
    else:
        cols_to_scale = [c for c in cols if c in df.columns and c in numeric_cols]

    if len(cols_to_scale) == 0:
        return df, scaler if scaler is not None else StandardScaler()

    X = df[cols_to_scale].to_numpy(dtype=float)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    df[cols_to_scale] = pd.DataFrame(X_scaled, columns=cols_to_scale, index=df.index)

    return df, scaler


__all__ = ["encode_categorical", "normalize_features"]
