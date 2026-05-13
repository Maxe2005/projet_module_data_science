import pandas as pd
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception as _:
    pass
import matplotlib.pyplot as plt

from kmeans import KMeans
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score

def load_data():
    df = pd.read_csv("iris.csv")
    X = df.iloc[:, [0, 2]].values
    return X

def affiche(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("Longueur des sépales")
    plt.ylabel("Largeur des pétales")
    plt.title("Données d'iris")
    plt.show()

def affiche_colored_clusters(X, k_means: KMeans):
    plt.scatter(X[:, 0], X[:, 1], c=k_means.results, cmap='viridis')
    plt.scatter(k_means.centers[:, 0], k_means.centers[:, 1], c='black', marker='x', s=200, linewidths=3)
    plt.xlabel("Longueur des sépales")
    plt.ylabel("Largeur des pétales")
    plt.title("Clusters d'iris")
    plt.show()

def main_1(n_clusters=8):
    X = load_data()
    # affiche(X)

    k_means = KMeans(dimension=2, max_iter=100, n_clusters=n_clusters)
    k_means.fit(X)

    affiche_colored_clusters(X, k_means)

def main_2(n_clusters=8):
    X = load_data()
    k_means = SklearnKMeans(n_clusters=n_clusters, random_state=0)
    k_means.fit(X)

    silhouette = silhouette_score(X, k_means.labels_)
    print(f"Silhouette Score: {silhouette}")

    plt.scatter(X[:, 0], X[:, 1], c=k_means.labels_, cmap='viridis')
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c='black', marker='x', s=200, linewidths=3)
    plt.xlabel("Longueur des sépales")
    plt.ylabel("Largeur des pétales")
    plt.title("Clusters d'iris (sklearn)")
    plt.show()

def find_optimal_k(X, max_k=10, nb_runs=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        run_scores = []
        for _ in range(nb_runs):
            k_means = SklearnKMeans(n_clusters=k, random_state=0)
            k_means.fit(X)
            score = silhouette_score(X, k_means.labels_)
            run_scores.append(score)
        avg_score = sum(run_scores) / len(run_scores)
        silhouette_scores.append(avg_score)
        print(f"K: {k}, Silhouette Score: {avg_score}")

    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score pour différents K")
    plt.show()


if __name__ == "__main__":
    main_1(2)
    main_2(2)
    find_optimal_k(load_data())
