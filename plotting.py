import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_ratings_per_user_histogram(df, R_min, R_max):
    """
    Display histogram of number of ratings per user for the given data.
    """
    ratings_per_user_filtered = df.groupby('username')['rating'].count()
    plt.figure(figsize=(10, 5))
    plt.hist(ratings_per_user_filtered, bins=range(R_min, R_max + 2), edgecolor='black', alpha=0.7)
    plt.title('Number of Ratings per User (Filtered Data)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.grid(True)
    plt.show()


def plot_time_ranges_histogram(df, days_interval):
    """
    Display histogram of time ranges for all ratings by users.
    """
    first_rating_date = df.groupby('username')['date'].min()
    last_rating_date = df.groupby('username')['date'].max()
    time_ranges = (last_rating_date - first_rating_date).dt.days
    plt.figure(figsize=(10, 5))
    plt.hist(time_ranges, bins=range(0, max(time_ranges) + 8, days_interval), edgecolor='black', alpha=0.7)
    plt.title('Time Ranges for All Ratings by Users (Filtered Data)')
    plt.xlabel('Time Range (days)')
    plt.ylabel('Number of Users')
    plt.grid(True)
    plt.show()


def plot_clusters(data, labels, title, centroids=None):
    """
    Plot the clusters in 2D using PCA for dimensionality reduction.
    """
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


def plot_cluster_sizes(unique, counts, title):
    """Plot the cluster size distribution."""
    plt.bar(unique, counts)
    plt.xlabel('Cluster Label')
    plt.ylabel('Cluster Size')
    plt.title(title)
    plt.show()


def plot_training_results(results_df, metrics):
    """
    Plots the results for each metric by cluster,
    """
    cluster_labels = results_df['Cluster'].values
    num_metrics = len(metrics)

    # Dynamically adjust the number of rows and columns based on the number of metrics.
    # num_cols = math.ceil(math.sqrt(num_metrics))

    # Current version has only 3 metrics, no need for dynamic adjustment
    # Just set num_cols to 1
    num_cols = 1
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
