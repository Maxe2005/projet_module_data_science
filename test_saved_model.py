import argparse

import numpy as np
import pandas as pd
from sklearn import metrics

from perceptron import Perceptron


def load_data(csv_path="iris.csv"):
    df = pd.read_csv(csv_path)
    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == "setosa", -1, 1)
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Test a saved Perceptron model on Iris data"
    )
    parser.add_argument(
        "--model", default="perceptron_model.pkl", help="Path to the saved model file"
    )
    parser.add_argument("--csv", default="iris.csv", help="Path to iris CSV file")
    args = parser.parse_args()

    X, y = load_data(args.csv)

    model = Perceptron.load(args.model)
    y_pred = model.predict(X)

    print("Loaded model:", args.model)
    print("Accuracy:", metrics.accuracy_score(y, y_pred))
    print("Confusion matrix:\n", metrics.confusion_matrix(y, y_pred))
    print("Precision:", metrics.precision_score(y, y_pred))
    print("Recall:", metrics.recall_score(y, y_pred))
    print("F1 score:", metrics.f1_score(y, y_pred))

    # show a few example predictions
    for i in range(5):
        print(f"X[{i}] = {X[i]} -> pred {y_pred[i]} (true {y[i]})")


if __name__ == "__main__":
    main()
