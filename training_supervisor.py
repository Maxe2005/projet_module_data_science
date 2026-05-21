"""Supervision de l'entraînement et évaluation (validation simple + validation croisée).

Fournit:
- `train_validate_simple(data, target, ...)` : entraînement simple + évaluation sur un split test.
- `cross_validate_model(data, target, cv, ...)` : validation croisée avec `KFold` et `cross_val_score`.

Conçu pour être utilisé depuis `main.py` ou en tant que script indépendant.
"""

from __future__ import annotations

import sys
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    get_scorer,
)
from sklearn.model_selection import KFold

from model_persistence_utils import save_model

try:
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    _COLOR_ENABLED = True
except ImportError:
    _COLOR_ENABLED = sys.stdout.isatty()


class Colors:
    RESET = "\033[0m" if _COLOR_ENABLED else ""
    BOLD = "\033[1m" if _COLOR_ENABLED else ""
    CYAN = "\033[96m" if _COLOR_ENABLED else ""
    GREEN = "\033[92m" if _COLOR_ENABLED else ""
    YELLOW = "\033[93m" if _COLOR_ENABLED else ""
    MAGENTA = "\033[95m" if _COLOR_ENABLED else ""
    BLUE = "\033[94m" if _COLOR_ENABLED else ""
    WHITE = "\033[97m" if _COLOR_ENABLED else ""
    RED = "\033[91m" if _COLOR_ENABLED else ""


def _style(text: str, color: str = "", bold: bool = False) -> str:
    style = ""
    if bold:
        style += Colors.BOLD
    if color:
        style += color
    return f"{style}{text}{Colors.RESET}" if style else text


def _format_confusion_matrix(cm, labels=None):
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    label_width = max(len(str(label)) for label in labels + ["Actual"])
    cell_width = max(5, max(len(str(int(value))) for value in cm.flatten()))
    header_cells = "".join(f" {str(label).rjust(cell_width)}" for label in labels)
    header_content = f"{'Actual'.rjust(label_width)} │{header_cells}"
    content_width = len(header_content)

    title_text = " CONFUSION MATRIX ".center(content_width)
    lines = [
        _style(f"╔{'═' * content_width}╗", Colors.BLUE),
        _style(f"║{_style(title_text, Colors.CYAN, bold=True)}║", Colors.BLUE),
        _style(f"╠{'═' * content_width}╣", Colors.BLUE),
        _style(f"║{_style(header_content, Colors.YELLOW, bold=True)}║", Colors.BLUE),
        _style(f"╠{'═' * content_width}╣", Colors.BLUE),
    ]

    for row_index, (label, row) in enumerate(zip(labels, cm)):
        content = f"{str(label).rjust(label_width)} │"
        for col_index, value in enumerate(row):
            value_str = f"{int(value):>{cell_width}}"
            if row_index == col_index:
                content += _style(f" {value_str}", Colors.GREEN, bold=True)
            else:
                content += _style(f" {value_str}", Colors.WHITE)
        lines.append(_style("║", Colors.BLUE) + content + _style("║", Colors.BLUE))

    lines.append(_style(f"╚{'═' * content_width}╝", Colors.BLUE))
    return "\n".join(lines)


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
    best_cm = None
    best_fold = 0

    for fold_index, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        fold_estimator = clone(estimator)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        fold_estimator.fit(X_train, y_train)
        score = scorer(fold_estimator, X_test, y_test)
        y_pred = fold_estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_model = fold_estimator
            best_cm = cm
            best_fold = fold_index

    scores = np.asarray(scores)

    frame_width = 50
    title = f" CROSS-VALIDATION {cv}-FOLD REPORT "
    print(_style("╔" + "═" * frame_width + "╗", Colors.CYAN))
    print(_style(f"║{title.center(frame_width)}║", Colors.CYAN, bold=True))
    print(_style("╠" + "═" * frame_width + "╣", Colors.CYAN))

    for i, s in enumerate(scores, 1):
        line = f"Fold {i:>2}: accuracy = {s:.4f}"
        color = Colors.GREEN if s >= scores.mean() else Colors.WHITE
        print(
            _style("║", Colors.CYAN)
            + _style(line.center(frame_width), color)
            + _style("║", Colors.CYAN)
        )

    print(_style("╠" + "═" * frame_width + "╣", Colors.CYAN))
    print(
        _style("║", Colors.CYAN)
        + _style(
            f"Mean accuracy : {scores.mean():.4f}".center(frame_width),
            Colors.YELLOW,
            bold=True,
        )
        + _style("║", Colors.CYAN)
    )
    print(
        _style("║", Colors.CYAN)
        + _style(
            f"Std deviation : {scores.std():.4f}".center(frame_width),
            Colors.YELLOW,
            bold=True,
        )
        + _style("║", Colors.CYAN)
    )
    print(_style("╠" + "═" * frame_width + "╣", Colors.CYAN))
    print(
        _style("║", Colors.CYAN)
        + _style(
            f"Best fold : {best_fold} (accuracy = {best_score:.4f})".center(
                frame_width
            ),
            Colors.MAGENTA,
            bold=True,
        )
        + _style("║", Colors.CYAN)
    )
    print(_style("╚" + "═" * frame_width + "╝", Colors.CYAN))

    if best_cm is not None:
        labels = [str(label) for label in sorted(np.unique(y))]
        print("\n" + _style("Confusion matrix (best fold)", Colors.BLUE, bold=True))
        print(f"Fold {best_fold} (accuracy = {best_score:.4f})")
        print(_format_confusion_matrix(best_cm, labels=labels))

    if model_out and best_model is not None:
        save_model(best_model, model_out)

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
