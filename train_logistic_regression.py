"""Entraînement d'un modèle de régression logistique

Usage minimal:
python train_logistic_regression.py --data-path car_insurance_formatted.csv
"""

from __future__ import annotations

import argparse
from typing import Literal, cast

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_split_utils import split_train_test
from model_persistence_utils import save_model


def train_logistic_regression(
    data_path: str,
    target: str = "outcome",
    test_size: float = 0.2,
    random_state: int | None = 0,
    model_out: str | None = None,
    solver: str = "lbfgs",
    max_iter: int = 100,
):
    """Charge les données, entraîne une LogisticRegression et retourne le modèle entraîné.

    Retourne: (model, X_test, y_test)
    """
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = split_train_test(
        df, target=target, test_size=test_size, random_state=random_state
    )

    solver_cast = cast(
        Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        solver,
    )
    model = LogisticRegression(solver=solver_cast, max_iter=max_iter)
    # Vérification / conversion de la cible pour s'assurer qu'on a bien des labels discrets
    # `split_train_test` retourne des ndarray; on les convertit en Series pour manipulation
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Si la cible est en float mais représente des classes (0.0/1.0), on convertit en int
    if pd.api.types.is_float_dtype(y_train):
        # Cas fréquent: probabilités ou 0.0/1.0 encodées en float -> binariser si dans [0,1]
        if y_train.min() >= 0.0 and y_train.max() <= 1.0:
            y_train = (y_train >= 0.5).astype(int)
            y_test = (y_test >= 0.5).astype(int)
            print(
                "[INFO] Target values appear in [0,1]; binarized at 0.5 for classification."
            )
        # Si toutes les valeurs sont des entiers représentés en float (ex: 0.0,1.0,2.0)
        elif (y_train.dropna() % 1 == 0).all():
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            print(
                "[INFO] Target float values are whole numbers; converted to int labels."
            )
        else:
            raise ValueError(
                f"Target '{target}' appears continuous (float) with {y_train.nunique()} unique values. "
                "A classifier requires discrete labels — please preprocess the target (binarize or categorize) before training."
            )

    model.fit(X_train, y_train)

    # Évaluation simple
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy (test): {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Sauvegarde
    if model_out:
        out_path = save_model(model, model_out)
        print(f"Modèle sauvegardé dans: {out_path}")

    return model, X_test, y_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne une régression logistique")
    p.add_argument("--data-path", type=str, default="car_insurance_formatted.csv")
    p.add_argument("--target", type=str, default="outcome")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument(
        "--model-out", type=str, default="models/logistic_regression_model.pkl"
    )
    p.add_argument("--solver", type=str, default="lbfgs")
    p.add_argument("--max-iter", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_logistic_regression(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        model_out=args.model_out,
        solver=args.solver,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
