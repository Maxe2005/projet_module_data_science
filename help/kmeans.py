import numpy as np


class KMeans:
    def __init__(self, dimension, max_iter=100, n_clusters=3):
        self.dimension = dimension
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.centers = np.ndarray([])
        self.results = np.ndarray([])

    def fit(self, X: np.ndarray):
        # Randomly initialize cluster centers
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign clusters
            self.results = self._assign_clusters(X)
            # Update cluster centers
            new_centers = self._update_centers(X)
            # Check for convergence
            if np.all(self.centers == new_centers):
                print("Convergence reached in {} iterations.".format(_))
                break
            self.centers = new_centers

    def _assign_clusters(self, X: np.ndarray):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X: np.ndarray):
        new_centers = []
        for i in range(self.n_clusters):
            cluster_points = X[self.results == i]
            if len(cluster_points) > 0:
                new_centers.append(np.mean(cluster_points, axis=0))
            else:
                new_centers.append(self.centers[i])
        return np.array(new_centers)

    def predict(self, x: np.ndarray):
        distances = np.linalg.norm(x - self.centers, axis=1)
        return np.argmin(distances)

    def get_data_cluster(self):
        return self.results
