"""Comparaison de 3 algorithmes de classification sur le jeu de données d'assurance automobile.

Ce script compare les performances des 3 classifieurs suivants :
- Régression logistique (LogisticRegression)
- Perceptron
- K plus proches voisins (KNeighborsClassifier)

Chaque algorithme est évalué par :
1. Entraînement sur le jeu d'entraînement
2. Évaluation par validation croisée
3. Affichage des scores

Usage:
python compare_classifiers.py --data-path car_insurance_formatted.csv
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data_split_utils import split_train_test


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


class ClassifierComparison:
    """Classe pour comparer plusieurs classifieurs."""

    def __init__(
        self,
        data_path: str,
        target: str = "outcome",
        test_size: float = 0.2,
        random_state: int | None = 42,
        cv_folds: int = 5,
    ):
        """
        Initialise la comparaison.

        Paramètres:
        - data_path: chemin du fichier CSV contenant les données formatées
        - target: nom de la colonne cible
        - test_size: proportion du jeu de test
        - random_state: graine pour la reproductibilité
        - cv_folds: nombre de folds pour la validation croisée
        """
        self.data_path = data_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.results = {}

    def load_and_prepare_data(self):
        """Charge les données et les divise en ensembles train/test."""
        print(f"Chargement des données depuis {self.data_path}...")
        df = pd.read_csv(self.data_path)

        print("Division des données en ensembles d'entraînement et de test...")
        X_train, X_test, y_train, y_test = split_train_test(
            df,
            target=self.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Préparation de la cible
        y_train = _prepare_target(y_train)
        y_test = _prepare_target(y_test)

        print(
            f"Données chargées: {X_train.shape[0]} exemples d'entraînement, "
            f"{X_test.shape[0]} exemples de test, "
            f"{X_train.shape[1]} features\n"
        )

        return X_train, X_test, y_train, y_test, df

    def train_and_evaluate(
        self,
        classifier: Any,
        classifier_name: str,
        X_train,
        X_test,
        y_train,
        y_test,
        df: pd.DataFrame,
    ):
        """
        Entraîne un classifieur, l'évalue et calcule les scores de validation croisée.

        Paramètres:
        - classifier: instance du classifieur à entraîner
        - classifier_name: nom du classifieur pour l'affichage
        - X_train, X_test, y_train, y_test: ensembles d'entraînement/test
        - df: DataFrame complet pour la validation croisée
        """
        print(f"{'=' * 60}")
        print(f"Entraînement et évaluation: {classifier_name}")
        print(f"{'=' * 60}")

        # Étape 1: Entraînement du modèle
        print("\n1. Entraînement du modèle sur le jeu d'entraînement...")
        classifier.fit(X_train, y_train)
        print("   ✓ Entraînement complété")

        # Étape 2: Évaluation sur le jeu de test
        print("\n2. Évaluation sur le jeu de test...")
        y_pred = classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred)
        print(f"   Précision (test): {accuracy_test:.4f}")

        # Étape 3: Validation croisée
        print(f"\n3. Validation croisée ({self.cv_folds}-fold)...")
        kfold = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        # Préparer les données pour la validation croisée
        X = df.drop(columns=[self.target]).to_numpy()
        y = df[self.target].to_numpy()
        y = _prepare_target(y)

        cv_scores = cross_val_score(classifier, X, y, cv=kfold, scoring="accuracy")

        print(f"   Scores CV: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Moyenne CV: {cv_scores.mean():.4f}")
        print(f"   Écart-type CV: {cv_scores.std():.4f}")

        # Étape 4: Rapport de classification
        print("\n4. Rapport de classification détaillé:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Stockage des résultats
        self.results[classifier_name] = {
            "test_accuracy": accuracy_test,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
        }

        return classifier

    def compare_all_classifiers(self):
        """Compare les 3 classifieurs spécifiés dans l'énoncé."""
        # Chargement et préparation des données
        X_train, X_test, y_train, y_test, df = self.load_and_prepare_data()

        # Définition des classifieurs avec leurs hyperparamètres
        classifiers = {
            "Régression Logistique": LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=self.random_state
            ),
            "Perceptron": Perceptron(
                max_iter=1000, random_state=self.random_state, tol=1e-3
            ),
            "K Plus Proches Voisins (k=5)": KNeighborsClassifier(n_neighbors=5),
        }

        # Entraînement et évaluation de chaque classifieur
        for classifier_name, classifier in classifiers.items():
            self.train_and_evaluate(
                classifier, classifier_name, X_train, X_test, y_train, y_test, df
            )

        # Affichage du résumé comparatif
        self.display_summary()

    def display_summary(self):
        """Affiche un résumé comparatif de tous les classifieurs."""
        print(f"\n{'=' * 60}")
        print("RÉSUMÉ COMPARATIF")
        print(f"{'=' * 60}\n")

        # Créer un tableau de résumé
        summary_data = []
        for classifier_name, results in self.results.items():
            summary_data.append(
                {
                    "Classifieur": classifier_name,
                    "Précision Test": f"{results['test_accuracy']:.4f}",
                    "Précision CV (mean)": f"{results['cv_mean']:.4f}",
                    "Écart-type CV": f"{results['cv_std']:.4f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Déterminer le meilleur classifieur
        best_classifier = max(self.results.items(), key=lambda x: x[1]["cv_mean"])
        print(
            f"\n✓ Meilleur classifieur (selon validation croisée): {best_classifier[0]}"
        )
        print(f"  Précision CV moyenne: {best_classifier[1]['cv_mean']:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Compare 3 algorithmes de classification"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="car_insurance_formatted.csv",
        help="Chemin du fichier CSV contenant les données formatées",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="outcome",
        help="Nom de la colonne cible",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine pour la reproductibilité",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Nombre de folds pour la validation croisée",
    )
    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_args()

    comparison = ClassifierComparison(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )

    comparison.compare_all_classifiers()


if __name__ == "__main__":
    main()
