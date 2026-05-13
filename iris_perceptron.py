import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from perceptron import Perceptron


def load_data():
    df = pd.read_csv("iris.csv")
    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == "setosa", -1, 1)
    return X, y


def plot_data_with_boundary(X, y, model: Perceptron):
    colors = {-1: "red", 1: "blue"}
    y_colors = [colors[val] for val in y]
    plt.scatter(X[:, 0], X[:, 1], c=y_colors, s=60)

    # decision boundary: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1)/w2
    w = model.weights
    b = model.bias
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xs = np.linspace(x_min, x_max, 200)
    if abs(w[1]) > 1e-8:
        ys = -(b + w[0] * xs) / w[1]
        plt.plot(xs, ys, "k--")

    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.title("Perceptron decision boundary")
    out_path = "decision_boundary.png"
    plt.savefig(out_path)
    print(f"Figure saved to {out_path}")
    plt.close()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = Perceptron(dimension=2, max_iter=1000, learning_rate=0.1)
    model.fit(X_train.tolist(), y_train)

    print("Coefficients (weights):", model.weights)
    print("Bias (w0):", model.bias)

    y_pred = model.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 score:", metrics.f1_score(y_test, y_pred))

    # sauvegarde du modèle entraîné
    model_path = "perceptron_model.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # chargement et vérification des prédictions
    loaded = Perceptron.load(model_path)
    y_pred_loaded = loaded.predict(X_test)
    print("Loaded model accuracy:", metrics.accuracy_score(y_test, y_pred_loaded))
    same = np.array_equal(y_pred, y_pred_loaded)
    print("Loaded predictions equal to original:", same)

    plot_data_with_boundary(X_test, y_test, model)

    # validation croisée avec Perceptron de sklearn
    sk_perc = SKPerceptron(max_iter=1000, tol=1e-3)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(sk_perc, X, y, cv=kf)
    print("Cross-val scores (sklearn Perceptron):", scores)
    print("Cross-val mean:", scores.mean())


if __name__ == "__main__":
    main()
