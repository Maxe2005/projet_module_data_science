import numpy as np
import pickle
from typing import List, Union


class Perceptron:
    def __init__(
        self, dimension: int, max_iter: int = 1000, learning_rate: float = 0.01
    ):
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = np.random.randn(dimension)
        self.bias = float(np.random.randn())

    def fit(self, X: List[np.ndarray], y: np.ndarray) -> None:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = X_arr.shape[0]

        for _ in range(self.max_iter):
            errors = 0
            # shuffle indices each epoch
            indices = np.random.permutation(n_samples)
            for i in indices:
                xi = X_arr[i]
                yi = y_arr[i]
                activation = self.bias + np.dot(self.weights, xi)
                y_pred = 1 if activation >= 0 else -1
                if y_pred != yi:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
                    errors += 1
            if errors == 0:
                break

    def predict(self, x: Union[np.ndarray, List[np.ndarray]]) -> Union[int, np.ndarray]:
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            activation = self.bias + np.dot(self.weights, x_arr)
            return 1 if activation >= 0 else -1
        else:
            activations = self.bias + np.dot(x_arr, self.weights)
            return np.where(activations >= 0, 1, -1)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj
