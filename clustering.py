import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from kmeans import KMeansClustering, display_kmeans_results
from plotting import plot_clusters, plot_cluster_sizes


def kmeans_clustering(df, L, metric, averaging='mean', verbose=False):
    """
    Perform K-means clustering on the given data for L clusters using the specified metric.
    """
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
    title = f"K-means clusters for L={L} using {metric}"
    plot_clusters(kmeans.X, clusters, title, kmeans.cluster_centers)

    return clusters


def agglomerative_clustering(dist_matrix, L, min_cluster_size=1):
    """
    Perform agglomerative clustering on the given distance matrix for L clusters.
    """
    print("\nPerforming Agglomerative Clustering...")

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=L, metric='precomputed', linkage='average')
    clusters = clustering.fit_predict(dist_matrix)

    # Merge small clusters
    if min_cluster_size > 0:
        clusters = _merge_small_clusters(clusters, min_cluster_size, dist_matrix=dist_matrix)

    # Evaluate the clustering
    _, unique, counts = _evaluate_clustering(dist_matrix, clusters)

    # Plot results
    plot_cluster_sizes(unique, counts, 'Cluster Size Distribution')

    return clusters


def _evaluate_clustering(dist_matrix, clusters):
    """
    Evaluate the clustering using silhouette score and print cluster distribution.
    """
    # Cluster Size Distribution
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))

    silhouette_score_ = silhouette_score(dist_matrix, clusters, metric='precomputed')

    # print(f"\nSilhouette Score: {silhouette_score_}")
    print("\nCluster Size Distribution:")
    for cluster, size in cluster_distribution.items():
        print(f"Cluster {cluster}: {size} users")

    return silhouette_score_, unique, counts


def _merge_small_clusters(clusters, min_cluster_size, dist_matrix, verbose=False):
    """
    Merges small clusters into the nearest larger cluster based on a precomputed distance matrix.
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
            avg_distance = np.mean(dist_matrix[np.ix_(small_cluster_indices, other_cluster_indices)])

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
