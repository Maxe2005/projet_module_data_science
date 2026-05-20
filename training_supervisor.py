"""Supervision de l'entraînement et évaluation (validation simple + validation croisée).

Fournit:
- `train_validate_simple(data, target, ...)` : entraînement simple + évaluation sur un split test.
- `cross_validate_model(data, target, cv, ...)` : validation croisée avec `KFold` et `cross_val_score`.

Conçu pour être utilisé depuis `main.py` ou en tant que script indépendant.
"""

from __future__ import annotations

from typing import Literal, cast

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score


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
        joblib.dump(model, model_out)
        print(f"Modèle sauvegardé dans: {model_out}")

    return model, X_test, y_test


def cross_validate_model(
    df: pd.DataFrame,
    target: str = "outcome",
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int | None = 0,
    estimator: LogisticRegression | None = None,
):
    """Effectue une validation croisée KFold et retourne les scores.

    Affiche la moyenne et l'écart-type des scores, et renvoie l'objet scores.
    """
    X = df.drop(columns=[target])
    y = _prepare_target(df[target])

    if estimator is None:
        estimator = LogisticRegression(max_iter=200)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=kf, scoring=scoring)

    print(f"Cross-validation ({cv}-fold) {scoring} scores:")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")

    return scores


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
