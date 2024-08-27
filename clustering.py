import numpy as np
import pandas as pd

from kmeans import KMeans
from logger import Logger
from metrics import euclidean_distance_generalized
from plotting import plot_clusters


def run(data: pd.DataFrame, L: int = 5, metric: callable = euclidean_distance_generalized) -> None:
    """
    Run the clustering process on the given data using the K-means algorithm and display the results.

    :param data: Data to cluster
    :param L: Number of clusters
    :param metric: Distance metric to use
    """
    print(f"\n--------- K-means clustering: L={L}, metric={metric.__name__} ---------")

    # Integer seed to make the randomness deterministic
    rnd = np.random.randint(0, 50)
    seed = 1

    # Initialize k-means and fit the data
    km = KMeans(metric=metric, k=L, random_state=seed, log_level=Logger.WARNING)
    km.fit(data)

    # Print clustering results
    # print(f"\nK-MEANS CLUSTERING")
    # print(f"Metric used: {km.metric.__name__}")
    # print(f"Number of clusters: {km.n_clusters}")
    # print(f"Initialization method used: {km.init}")
    # print(f"Number of initializations: {km.n_init}, seed: {seed}")
    # print(f"Max number of iterations per run: {km.max_iter}")
    # print(f"Tolerance: {km.tol}")

    # print(f"\nDATA:")
    # print(f"Number of samples: {km.n_samples}")
    # print(f"Number of features: {km.n_features}")

    # print(f"\nRESULTS")
    print(f"Clusters: {km.cluster_sizes('desc')}")
    # print(f"Inertia: {km.inertia}")
    # print(f"Iterations run: {km.n_iter}")
    # print("Centroids:")
    # for centroid in km.cluster_centers:
    #     n_values_printed = 10
    #     if len(centroid) <= n_values_printed:
    #         formatted_centroid = "[" + " ".join("{:.2f}".format(x) for x in centroid) + "]"
    #     else:
    #         first_values = " ".join("{:.2f}".format(x) for x in centroid[:n_values_printed // 2])
    #         last_values = " ".join("{:.2f}".format(x) for x in centroid[-n_values_printed // 2:])
    #         formatted_centroid = "[" + first_values + " ... " + last_values + "]"
    #     print(formatted_centroid)

    # Plot clusters
    title = f"Clusters for L={L} using {metric.__name__}"
    plot_clusters(km.X, km.cluster_centers, km.labels, title)
