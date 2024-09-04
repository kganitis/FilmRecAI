import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_clusters(data, labels, title, centroids=None, dim_reduct='pca'):
    dim_reduct = dim_reduct.lower()
    if dim_reduct == 'pca':
        _plot_clusters_pca(data, labels, title, centroids)
    elif dim_reduct in ('pca3d', 'pca-3d'):
        _plot_clusters_pca_3d(data, labels, title, centroids)
    elif dim_reduct in ('tsne', 't-sne'):
        _plot_clusters_tsne(data, labels, title)
    else:
        raise ValueError(f"Invalid dimensionality reduction technique: {dim_reduct}")


def _plot_clusters_pca(data, labels, title, centroids=None):
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    pca_centroids = pca.transform(centroids) if centroids is not None else None

    # Adjust labels to start from 1 instead of 0
    adj_labels = labels + 1

    # Define a discrete colormap
    cmap = plt.get_cmap('tab20', np.max(adj_labels))

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=18)
    scatter = plt.scatter(x=pca_data[:, 0], y=pca_data[:, 1], c=adj_labels, cmap=cmap, s=30, alpha=0.7)

    # Plot the centroids
    if pca_centroids is not None:
        for idx, centroid in enumerate(pca_centroids):
            plt.scatter(x=centroid[0], y=centroid[1], c=[cmap(idx)], marker='X', s=100, edgecolor='k')

    # Create a colorbar with integer labels
    cbar = plt.colorbar(scatter, ticks=np.arange(1, np.max(adj_labels) + 1))
    cbar.set_label('Cluster Label')
    cbar.set_ticks(np.arange(1, np.max(adj_labels) + 1))
    cbar.set_ticklabels(np.arange(1, np.max(adj_labels) + 1))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def _plot_clusters_pca_3d(data, labels, title, centroids=None):
    # Perform dimensionality reduction using PCA to 3D
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)
    pca_centroids = pca.transform(centroids) if centroids is not None else None

    # Adjust labels to start from 1 instead of 0
    adj_labels = labels + 1

    # Define a discrete colormap
    cmap = plt.get_cmap('tab20', np.max(adj_labels))

    # Plot the clusters in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=adj_labels, cmap=cmap, s=30, alpha=0.7)

    # Plot the centroids with the same colors as their clusters
    if pca_centroids is not None:
        for idx, centroid in enumerate(pca_centroids):
            ax.scatter(centroid[0], centroid[1], centroid[2], c=[cmap(idx)], marker='X', s=100, edgecolor='k')

    # Create a colorbar with integer labels
    cbar = plt.colorbar(scatter, ticks=np.arange(1, np.max(adj_labels) + 1), ax=ax, pad=0.1)
    cbar.set_label('Cluster Label')
    cbar.set_ticks(np.arange(1, np.max(adj_labels) + 1))
    cbar.set_ticklabels(np.arange(1, np.max(adj_labels) + 1))

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()


def _plot_clusters_tsne(data, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)

    adj_labels = labels + 1
    cmap = plt.get_cmap('tab20', np.max(adj_labels))

    # Plot the reduced data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=adj_labels, cmap=cmap, s=30, alpha=0.6)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()


def plot_training_results(results_df, metrics):
    """
    Plots the results for each metric by cluster, dynamically adjusting the number of rows and columns
    based on the number of metrics, while fitting the plots to a (7, 10) figsize.
    """
    cluster_labels = results_df['Cluster'].values
    num_metrics = len(metrics)

    # Calculate the optimal number of rows and columns
    num_cols = math.ceil(math.sqrt(num_metrics))
    num_rows = math.ceil(num_metrics / num_cols)

    # Create the figure and axes with dynamic rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(7, 10))

    # Flatten axs for easy iteration (handle cases where it's a 2D array)
    if num_rows * num_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.bar(cluster_labels, results_df[f'Train {metric}'], alpha=0.6, label='Train')
        ax.bar(cluster_labels, results_df[f'Test {metric}'], alpha=0.6, label='Test')
        ax.set_xticks(cluster_labels)
        ax.set_xticklabels(cluster_labels)
        ax.set_title(f'{metric} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(metric)
        ax.legend()

    # Hide any unused subplots
    for i in range(num_metrics, num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
