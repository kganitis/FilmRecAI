import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from kmeans import KMeansClustering, display_kmeans_results
from plotting import plot_clusters


def kmeans_clustering(df, L, metric, averaging='mean', verbose=True, plots=True):
    if verbose:
        print(f"\nPerforming K-means clustering for L={L} using metric={metric}")

    # Perform K-means clustering
    kmeans = KMeansClustering(
        metric=metric,
        k=L,
        # random_state=1,
        init='k-means++',  # for faster results, submitted results where run with init='random'
        # log_level=1,
        max_iter=30,  # results don't seem to improve beyond 30 iterations
        averaging=averaging
    )
    clusters = kmeans.fit_predict(df)

    # Check the sizes of the clusters
    unique, cluster_sizes = np.unique(clusters, return_counts=True)
    print("K-means clusters:", dict(zip(unique, cluster_sizes.tolist())))

    # Display clustering results
    if verbose:
        display_kmeans_results(kmeans)

    # Plot the clusters
    if plots:
        title = f"K-means clusters for L={L} using {metric}"
        plot_clusters(kmeans.X, clusters, title, kmeans.cluster_centers)

    return clusters


def agglomerative_clustering(dist_matrix, L, min_cluster_size=10, verbose=False, plots=False):
    if verbose:
        print("\nPerforming Agglomerative Clustering...")

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=L, metric='precomputed', linkage='average')
    clusters = clustering.fit_predict(dist_matrix)

    # Merge small clusters
    if min_cluster_size > 0:
        clusters = _merge_small_clusters(clusters, min_cluster_size, dist_matrix=dist_matrix, verbose=False)

    # Evaluate the clustering
    silhouette_score_, unique, counts = _evaluate_clustering(dist_matrix, clusters, verbose)

    # Plot results
    if plots:
        _plot_cluster_sizes(unique, counts, 'Cluster Size Distribution')

    return clusters, silhouette_score_


def spectral_clustering(dist_matrix, L, delta, min_cluster_size=0, verbose=False, plots=False):
    if verbose:
        print("\nPerforming Spectral Clustering...")

    # Convert to an affinity matrix using the RBF kernel
    affinity_matrix = np.exp(-dist_matrix ** 2 / (2. * delta ** 2))

    # Perform spectral clustering
    spectral = SpectralClustering(
        n_clusters=L,
        affinity='precomputed',
        assign_labels='kmeans',  # 'kmeans' commonly used, can also try 'discretize'
        random_state=42
    )
    clusters = spectral.fit_predict(affinity_matrix)

    # Merge small clusters
    if min_cluster_size > 0:
        clusters = _merge_small_clusters(clusters, min_cluster_size, dist_matrix=dist_matrix, verbose=verbose)

    # Evaluate the clustering
    silhouette_score_, unique, counts = _evaluate_clustering(dist_matrix, clusters, verbose)

    # Plot results
    if plots:
        _plot_cluster_sizes(unique, counts, 'Cluster Size Distribution')

    return clusters, silhouette_score_


def _evaluate_clustering(dist_matrix, clusters, verbose=False):
    """Evaluate the clustering using silhouette score and print cluster distribution."""

    # Cluster Size Distribution
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))

    silhouette_score_ = silhouette_score(dist_matrix, clusters, metric='precomputed')

    if verbose:
        # print(f"\nSilhouette Score: {silhouette_score_}")
        print("\nCluster Size Distribution:")
        for cluster, size in cluster_distribution.items():
            print(f"Cluster {cluster}: {size} users")

    return silhouette_score_, unique, counts


def _plot_cluster_sizes(unique, counts, title):
    """Plot the cluster size distribution."""
    plt.bar(unique, counts)
    plt.xlabel('Cluster Label')
    plt.ylabel('Cluster Size')
    plt.title(title)
    plt.show()


def _merge_small_clusters(clusters, min_cluster_size, dist_matrix=None, data=None, metric=None, verbose=False):
    """
    Merges small clusters into the nearest larger cluster based on a distance metric or precomputed distance matrix.

    Parameters:
    -----------
    clusters : array-like of shape (n_samples,)
        Array of cluster labels for each data point.

    min_cluster_size : int
        Minimum cluster size. Clusters with fewer elements than this value will be merged into a larger cluster.

    dist_matrix : array-like of shape (n_samples, n_samples), optional
        Precomputed distance matrix. If provided, this matrix is used to calculate the distances between clusters.

    data : array-like of shape (n_samples, n_features), optional
        The data points. Required if `dist_matrix` is not provided. Used in conjunction with `metric` to calculate distances.

    metric : callable, optional
        A function that computes the distance between two data points. Required if `dist_matrix` is not provided.

    verbose : bool, default=False
        If True, prints information about the merging process.

    Returns:
    --------
    clusters : array-like of shape (n_samples,)
        Array of cluster labels after merging small clusters.
    """
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    small_clusters = [cluster for cluster, size in cluster_distribution.items() if size < min_cluster_size]

    if small_clusters and verbose:
        print(f"\nMerging small clusters: {small_clusters}")

    for small_cluster in small_clusters:
        small_cluster_indices = np.where(clusters == small_cluster)[0]
        nearest_cluster = None
        min_avg_distance = float('inf')

        for other_cluster in unique:
            if other_cluster == small_cluster or other_cluster in small_clusters:
                continue
            other_cluster_indices = np.where(clusters == other_cluster)[0]

            if dist_matrix is not None:
                # Calculate average distance using the distance matrix
                avg_distance = np.mean(dist_matrix[np.ix_(small_cluster_indices, other_cluster_indices)])
            else:
                # Calculate average distance using the metric function
                avg_distances = [metric(data[i], data[j]) for i in small_cluster_indices for j in other_cluster_indices]
                avg_distance = np.mean(avg_distances)

            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                nearest_cluster = other_cluster

        if nearest_cluster is not None:
            clusters[small_cluster_indices] = nearest_cluster

    # Remove empty clusters and reindex labels
    unique, counts = np.unique(clusters, return_counts=True)
    reindex_map = {old_label: new_label for new_label, old_label in enumerate(unique)}
    clusters = np.array([reindex_map[label] for label in clusters])

    return clusters
