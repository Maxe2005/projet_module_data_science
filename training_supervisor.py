"""Supervision de l'entraînement et évaluation (validation simple + validation croisée).

Fournit:
- `train_validate_simple(data, target, ...)` : entraînement simple + évaluation sur un split test.
- `cross_validate_model(data, target, cv, ...)` : validation croisée avec `KFold` et `cross_val_score`.

Conçu pour être utilisé depuis `main.py` ou en tant que script indépendant.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, get_scorer
from sklearn.model_selection import KFold

from model_persistence_utils import save_model


def _prepare_target(y):
    """Convertit la cible en labels discrets si nécessaire."""
    y = pd.Series(y)
    if pd.api.types.is_float_dtype(y):
        if y.min() >= 0.0 and y.max() <= 1.0:
            return (y >= 0.5).astype(int)
        elif (y.dropna() % 1 == 0).all():
            return y.astype(int)
        else:
            raise ValueError(
                "La cible semble continue; prétraitez la cible avant entraînement."
            )
    return y


def train_validate_simple(
    df: pd.DataFrame,
    target: str = "outcome",
    test_size: float = 0.2,
    random_state: int | None = 0,
    model_out: str | None = None,
    solver: str = "lbfgs",
    max_iter: int = 100,
):
    """Entraîne une LogisticRegression sur un split train/test et retourne le modèle.

    Retourne: (model, X_test, y_test)
    """
    from data_split_utils import split_train_test

    X_train, X_test, y_train, y_test = split_train_test(
        df, target=target, test_size=test_size, random_state=random_state
    )

    y_train = _prepare_target(y_train)
    y_test = _prepare_target(y_test)

    solver_cast = cast(
        Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        solver,
    )
    model = LogisticRegression(solver=solver_cast, max_iter=max_iter)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy (test split): {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    if model_out:
        saved_path = save_model(model, model_out)
        print(f"Modèle sauvegardé dans: {saved_path}")

    return model, X_test, y_test


def cross_validate_model(
    df: pd.DataFrame,
    target: str = "outcome",
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int | None = 0,
    estimator: LogisticRegression | None = None,
    model_out: str | None = None,
):
    """Effectue une validation croisée KFold et retourne les scores.

    Affiche la moyenne et l'écart-type des scores, et sauvegarde le modèle du fold
    qui obtient le meilleur score si `model_out` est fourni.

    Retourne:
        tuple[np.ndarray, float]: les scores par fold et le meilleur score obtenu.
    """
    X = df.drop(columns=[target])
    y = _prepare_target(df[target])

    if estimator is None:
        estimator = LogisticRegression(max_iter=200)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer = get_scorer(scoring)
    scores = []
    best_score = float("-inf")
    best_model = None

    for train_index, test_index in kf.split(X, y):
        fold_estimator = clone(estimator)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        fold_estimator.fit(X_train, y_train)
        score = scorer(fold_estimator, X_test, y_test)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_model = fold_estimator

    print(f"Cross-validation ({cv}-fold) {scoring} scores:")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    scores = np.asarray(scores)
    print(f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")

    if model_out and best_model is not None:
        saved_path = save_model(best_model, model_out)
        print(
            f"Meilleur modèle CV sauvegardé dans: {saved_path} (score={best_score:.4f})"
        )

    return scores, best_score


if __name__ == "__main__":
    # Petit exemple exécuté en ligne de commande
    import argparse

    p = argparse.ArgumentParser(
        description="Supervision: entraînement simple + validation croisée"
    )
    p.add_argument("--data", type=str, default="car_insurance_formatted.csv")
    p.add_argument("--target", type=str, default="outcome")
    p.add_argument("--cv", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.data)

    print("=== Entraînement simple (split) ===")
    train_validate_simple(df, target=args.target, model_out=None)

    print("\n=== Validation croisée ===")
    cross_validate_model(df, target=args.target, cv=args.cv)
