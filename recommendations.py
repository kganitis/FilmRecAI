import csv
import os

import numpy as np
import pandas as pd
from keras import layers, models
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances, precision_recall_curve, auc, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

import clustering as cl


def run(ratings_df,
        # clustering
        clustering='agglomerative', L=100, linkage='average', delta=5.0, min_cluster_size=50,
        # training
        k=5, test_size=0.1, hidden_layer_units=(4096, 2048), large_cluster_threshold=0.,
        # evaluation
        bin_clas_threshold=0.3,
        # output
        verbose=True, plots=True):

    filename = 'experiment_results8.csv'

    # Convert to numpy ndarray for easier manipulation
    ratings_matrix = ratings_df.to_numpy()

    # Compute distance matrix
    if verbose:
        print("\nComputing distance matrix...")
    binary_ratings = ratings_matrix > 0  # convert to binary
    dist_matrix = pairwise_distances(binary_ratings, metric='jaccard')

    # Perform clustering
    clustering_results = perform_clustering(clustering, dist_matrix, L, linkage, delta, min_cluster_size, large_cluster_threshold, verbose, plots)
    clusters, silhouette_score, n_clusters, cluster_sizes, size_threshold = clustering_results

    # Train a model for each cluster
    train_results = {'Cluster': [], 'Train Accuracy': [], 'Test Accuracy': []}
    models_dict = {}
    for cluster_label in range(n_clusters):
        cluster_results = train_model_for_cluster(cluster_label, clusters, ratings_matrix, dist_matrix, k, test_size, hidden_layer_units,
                                large_cluster_threshold, size_threshold, verbose)
        # Store the results
        train_results['Cluster'].append(cluster_label)
        train_results['Train Accuracy'].append(cluster_results['train_accuracy'])
        train_results['Test Accuracy'].append(cluster_results['test_accuracy'])

        # Store the model and test data for later analysis
        models_dict[cluster_label] = {
            'model': cluster_results['model'],
            'X_test': cluster_results['X_test'],
            'y_test': cluster_results['y_test'],
        }

    # Display accuracy results for each cluster
    results_df = pd.DataFrame(train_results)
    if verbose:
        print("\nCluster-wise Accuracy Results:")
        print(results_df)

    # Collect cluster statistics
    cluster_stats_df = collect_cluster_statistics(models_dict, cluster_sizes, bin_clas_threshold, verbose)
    overall_averages = calculate_weighted_averages(cluster_stats_df)

    # Display the stats
    if verbose:
        print(cluster_stats_df)
        for metric, value in overall_averages.items():
            print(f"Weighted Overall {metric}: {value:.4f}")

    # Plot the stats
    if plots:
        plot_cluster_statistics(cluster_stats_df)

    # Prepare the result dictionary
    result = {
        "n": ratings_df.shape[0],
        "m": ratings_df.shape[1],
        "clustering": clustering,
        "L": L,
        "linkage": linkage if clustering == 'agglomerative' else '',
        "delta": delta if clustering == 'spectral' else '',
        "n_clusters": n_clusters,
        "k": k,
        "test_size": test_size,
        "hidden_layers": hidden_layer_units,
        "bin_clas_threshold": bin_clas_threshold,
        "Silhouette Score": f"{silhouette_score:.4f}",
    }
    result.update({f"{metric} Avg": f"{value:.4f}" for metric, value in overall_averages.items()})

    save_results_to_csv(result, filename)

    return result


def perform_clustering(clustering, dist_matrix, L, linkage, delta, min_cluster_size, large_cluster_threshold, verbose, plots):
    clustering = clustering.lower()
    if clustering == 'agglomerative':
        clusters, silhouette_score = cl.agglomerative_clustering(dist_matrix, L, linkage, min_cluster_size, verbose,
                                                                 plots)
    elif clustering == 'spectral':
        clusters, silhouette_score = cl.spectral_clustering(dist_matrix, L, delta, min_cluster_size, verbose, plots)
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'spectral'.")

    # Find the number of clusters actually formed
    n_clusters = len(np.unique(clusters))

    # Determine the size of each cluster
    cluster_sizes = [np.sum(clusters == cluster_label) for cluster_label in range(n_clusters)]

    size_threshold = max(cluster_sizes) + 1
    if large_cluster_threshold > 0:
        # Sort cluster sizes to determine the threshold for the largest clusters
        sorted_cluster_sizes = sorted(cluster_sizes)
        threshold_index = int((1 - large_cluster_threshold) * len(sorted_cluster_sizes))
        size_threshold = sorted_cluster_sizes[threshold_index]

        if verbose:
            print(f"\nCluster size threshold for large clusters: {size_threshold}")

    return clusters, silhouette_score, n_clusters, cluster_sizes, size_threshold


def train_model_for_cluster(cluster_label, clusters, ratings_matrix, dist_matrix, k, test_size, hidden_layer_units, large_cluster_threshold, size_threshold, verbose):
    if verbose:
        print(f"\nTraining model for cluster {cluster_label}")

    user_indices = np.where(clusters == cluster_label)[0]

    # Determine which architecture to use based on the cluster size
    if large_cluster_threshold > 0 and len(user_indices) >= size_threshold:
        # Add two extra layers for large clusters
        adjusted_hidden_layer_units = [4 * hidden_layer_units[0], 2 * hidden_layer_units[0]] + list(
            hidden_layer_units)
    else:
        adjusted_hidden_layer_units = hidden_layer_units

    # Prepare data for the cluster
    train_test_split_list = prepare_data_for_cluster(ratings_matrix, user_indices, dist_matrix, k, test_size)
    X_train, X_test, y_train, y_test = train_test_split_list

    # Create a neural network model
    model = models.Sequential()

    # Add the hidden layers specified in the hidden_layer_units parameter
    for units in adjusted_hidden_layer_units:
        model.add(layers.Dense(units, activation='relu'))

    # Output layer for binary classification
    model.add(layers.Dense(y_train.shape[1], activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "model": model,
        "X_test": X_test,
        "y_test": y_test
    }


def get_k_nearest_neighbors(k, user_idx, dist_matrix):
    """Get k nearest neighbors of a user based on the distance matrix."""
    distances = dist_matrix[user_idx]
    nearest_neighbors = np.argsort(distances)[:k + 1]  # +1 to exclude the user itself
    return nearest_neighbors[1:]  # exclude the user itself


def prepare_data_for_cluster(ratings_matrix, user_indices, dist_matrix, k, test_size):
    """Prepare the feature vectors and labels for the users in a cluster."""
    feature_vectors = []
    labels = []

    for user_idx in user_indices:
        neighbors = get_k_nearest_neighbors(k, user_idx, dist_matrix)
        label = (ratings_matrix[user_idx] > 0).astype(int)  # binary labels: 1 if watched, 0 otherwise
        feature_vector = np.concatenate([ratings_matrix[neighbor] for neighbor in neighbors], axis=0)
        feature_vectors.append(feature_vector)
        labels.append(label)

    X = np.array(feature_vectors)
    y = np.array(labels)

    return train_test_split(X, y, test_size=test_size)


def collect_cluster_statistics(models_dict, cluster_sizes, bin_clas_threshold, verbose):
    if verbose:
        print("\nCalculating cluster statistics...")

    cluster_stats = []
    for cluster_label in range(len(cluster_sizes)):
        stats = cluster_statistics(models_dict, cluster_label, threshold=bin_clas_threshold)
        stats['Size'] = cluster_sizes[cluster_label]
        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def cluster_statistics(models_dict, cluster_label, threshold=0.5):
    """Calculate and return various metrics for watched and non-watched movies for all users in a cluster."""
    model_info = models_dict[cluster_label]
    model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']

    # Lists to store probabilities and true labels for all users
    all_probs = []
    all_true_labels = []

    # Iterate over all users in the cluster
    for user_idx in range(X_test.shape[0]):
        actual_ratings = y_test[user_idx]
        predicted_probabilities = model.predict(X_test[user_idx].reshape(1, -1), verbose=0).flatten()

        # Collect all probabilities and true labels
        all_probs.extend(predicted_probabilities)
        all_true_labels.extend(actual_ratings)

    # Convert to numpy arrays for metric calculations
    all_probs = np.array(all_probs)
    all_true_labels = np.array(all_true_labels)

    # Calculate AUC and PRC AUC
    roc_auc = roc_auc_score(all_true_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_true_labels, all_probs)
    prc_auc = auc(recall, precision)

    # Perform binary classification based on the threshold
    binary_predictions = (all_probs >= threshold).astype(int)

    # Calculate precision, recall, F1-score, TPR, and FPR
    precision = precision_score(all_true_labels, binary_predictions)
    recall = recall_score(all_true_labels, binary_predictions)
    f1 = f1_score(all_true_labels, binary_predictions)

    # Calculate watched and non-watched averages
    watched_avg = np.mean(all_probs[all_true_labels == 1]) if np.any(all_true_labels == 1) else float('nan')
    non_watched_avg = np.mean(all_probs[all_true_labels == 0]) if np.any(all_true_labels == 0) else float('nan')

    return {
        'Cluster': cluster_label,
        'Size': 0,  # This will be set externally
        'Watched Avg': watched_avg,
        'Non-Watched Avg': non_watched_avg,
        'Gap': watched_avg - non_watched_avg,
        'ROC AUC': roc_auc,
        'PRC AUC': prc_auc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
    }


def calculate_weighted_averages(cluster_stats_df):
    def weighted_average(df, metric_column):
        return np.average(df[metric_column], weights=df['Size'])

    metrics = ['Watched Avg', 'Non-Watched Avg', 'Gap', 'ROC AUC', 'PRC AUC', 'Precision', 'Recall', 'F1-Score']
    return {metric: weighted_average(cluster_stats_df, metric) for metric in metrics}


def plot_cluster_statistics(cluster_results_df):
    clusters = cluster_results_df['Cluster']
    watched_avg = cluster_results_df['Watched Avg']
    non_watched_avg = cluster_results_df['Non-Watched Avg']
    gap = cluster_results_df['Gap']
    roc_auc = cluster_results_df['ROC AUC']
    prc_auc = cluster_results_df['PRC AUC']
    precision = cluster_results_df['Precision']
    recall = cluster_results_df['Recall']
    f1_score = cluster_results_df['F1-Score']

    # Number of subplots needed (for simplicity, we'll use one figure with multiple subplots)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    bar_width = 0.35
    index = np.arange(len(clusters))

    # Plot watched and non-watched averages
    axs[0, 0].bar(index, watched_avg, bar_width, label='Watched Avg', color='green', alpha=0.7)
    axs[0, 0].bar(index + bar_width, non_watched_avg, bar_width, label='Non-Watched Avg', color='red', alpha=0.7)
    axs[0, 0].plot(index + bar_width / 2, gap, label='Gap (Watched - Non-Watched)', color='blue', marker='o')
    axs[0, 0].set_xlabel('Cluster')
    axs[0, 0].set_ylabel('Average Probability')
    axs[0, 0].set_title('Watched vs Non-Watched Averages')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Plot ROC AUC
    axs[1, 0].plot(index, roc_auc, label='ROC AUC', color='purple', marker='o')
    axs[1, 0].set_xlabel('Cluster')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].set_title('ROC AUC per Cluster')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Plot PRC AUC
    axs[2, 0].plot(index, prc_auc, label='PRC AUC', color='orange', marker='o')
    axs[2, 0].set_xlabel('Cluster')
    axs[2, 0].set_ylabel('PRC AUC')
    axs[2, 0].set_title('PRC AUC per Cluster')
    axs[2, 0].legend()
    axs[2, 0].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Plot Precision
    axs[0, 1].plot(index, precision, label='Precision', color='green', marker='o')
    axs[0, 1].set_xlabel('Cluster')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].set_title('Precision per Cluster')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Plot Recall
    axs[1, 1].plot(index, recall, label='Recall', color='magenta', marker='o')
    axs[1, 1].set_xlabel('Cluster')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].set_title('Recall per Cluster')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Plot F1-Score
    axs[2, 1].plot(index, f1_score, label='F1-Score', color='cyan', marker='o')
    axs[2, 1].set_xlabel('Cluster')
    axs[2, 1].set_ylabel('F1-Score')
    axs[2, 1].set_title('F1-Score per Cluster')
    axs[2, 1].legend()
    axs[2, 1].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def save_results_to_csv(result, csv_file='experiment_results.csv'):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
