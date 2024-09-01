import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from kmeans import KMeans, display_kmeans_results
from plotting import plot_clusters


def kmeans(df, L, metric, averaging='mean', min_cluster_size=10):
    """
    Run the clustering process on the given data using the K-means algorithm and display the results.

    :param df: The data to cluster (as a pandas DataFrame)
    :param L: The number of clusters to create
    :param metric: The distance metric to use
    :param averaging: The center averaging method to use
    :param min_cluster_size: The minimum size of a cluster
    """
    print(f"\n--------- K-means clustering: L={L}, metric={metric.__name__} ---------")

    # Initialize K-means and fit the data
    km = KMeans(
        metric=metric,
        k=L,
        # random_state=1,
        init='k-means++',
        # log_level=1,
        max_iter=30,
        averaging=averaging
    )
    km.fit(df)
    clusters = km.labels
    _merge_small_clusters(km.X, clusters, metric, min_cluster_size)

    # Check the sizes of the clusters
    unique, cluster_sizes = np.unique(clusters, return_counts=True)
    print("K-means clusters:", dict(zip(unique, cluster_sizes.tolist())))

    # Plot the clusters
    title = f"K-means clusters for L={L} using {metric.__name__}"
    plot_clusters(km.X, clusters, title, km.cluster_centers)

    display_kmeans_results(km)

    return clusters


def agglomerative_clustering(data, L, metric, dist_matrix, linkage_method, min_cluster_size=10):
    print(f"\nPerforming agglomerative clustering with {linkage_method} linkage...")
    Z = linkage(dist_matrix, method=linkage_method)

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram for Agglomerative Clustering')
    plt.xlabel('Users')
    plt.ylabel(f'{linkage_method.capitalize()} Distance')
    plt.show()

    # Create clusters
    clusters = fcluster(Z, t=0.9999, criterion='distance')
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"Clusters formed: {len(unique)}")
    # print("Cluster sizes:", dict(zip(unique, counts)))
    print("Sorted cluster sizes:", sorted(counts, key=lambda x: x, reverse=True))

    # Merge clusters
    print("Merging small clusters...")
    clusters = _merge_small_clusters(data, clusters, metric, min_cluster_size)
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"Merged clusters: {len(unique)}")
    # print("Merged cluster sizes:", dict(zip(unique, counts)))
    print("Sorted merged cluster sizes:", sorted(counts, key=lambda x: x, reverse=True))

    return clusters


def spectral_clustering(dist_matrix, L):
    # Convert to an affinity matrix using the RBF kernel
    delta = 1.0  # adjust delta based on the scale of the distances
    affinity_matrix = np.exp(-dist_matrix ** 2 / (2. * delta ** 2))

    # Perform spectral clustering
    print("\nPerforming Spectral Clustering...")
    spectral_clustering = SpectralClustering(
        n_clusters=L,
        affinity='precomputed',
        assign_labels='kmeans',  # 'kmeans' commonly used, can also try 'discretize'
        random_state=42
    )
    clusters = spectral_clustering.fit_predict(affinity_matrix)

    # Evaluate the clustering
    silhouette_avg = silhouette_score(dist_matrix, clusters, metric='precomputed')
    print(f"\nSilhouette Score: {silhouette_avg}")

    # Cluster Size Distribution
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))

    print("\nCluster Size Distribution:")
    for cluster, size in cluster_distribution.items():
        print(f"Cluster {cluster}: {size} users")

    # Plot the clusters
    plt.bar(unique, counts)
    plt.xlabel('Cluster Label')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution')
    plt.show()

    return clusters


def _merge_small_clusters(data, clusters, metric, min_cluster_size):
    # Identify small clusters
    unique, counts = np.unique(clusters, return_counts=True)
    small_clusters = unique[counts < min_cluster_size]

    # Iterate over small clusters and merge them into the nearest larger cluster
    for small_cluster in small_clusters:
        # Get indices of the small cluster
        small_cluster_indices = np.where(clusters == small_cluster)[0]

        # Compute distance to all other clusters
        large_cluster_indices = np.where(~np.isin(clusters, small_clusters))[0]
        large_cluster_labels = clusters[large_cluster_indices]

        # Find the nearest larger cluster
        for i in small_cluster_indices:
            min_distance = float('inf')
            nearest_large_cluster = None

            for j, large_cluster_index in enumerate(large_cluster_indices):
                distance = metric(data[i], data[large_cluster_index])
                if distance < min_distance:
                    min_distance = distance
                    nearest_large_cluster = large_cluster_labels[j]

            # Reassign the small cluster point to the nearest large cluster
            clusters[i] = nearest_large_cluster

    return clusters
