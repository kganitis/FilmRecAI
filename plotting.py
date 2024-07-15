import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_clusters(data, centroids, labels, n_iter):
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    pca_centroids = pca.transform(centroids)

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    plt.title(f'Custom k-means User Clusters after {n_iter} iterations')
    scatter = plt.scatter(x=pca_data[:, 0], y=pca_data[:, 1], c=labels)
    plt.scatter(x=pca_centroids[:, 0], y=pca_centroids[:, 1])
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()
