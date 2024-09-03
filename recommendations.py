import numpy as np
import pandas as pd
from keras import layers, models
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from keras import metrics

import clustering as cl


def run_movie_recommendation_pipeline(ratings_df,
                                      # Clustering parameters
                                      clustering_method='agglomerative',
                                      num_clusters=30,
                                      linkage_method='average',
                                      min_cluster_size=1000,
                                      large_cluster_threshold=0.1667,
                                      # KNN and Train-Test Split parameters
                                      k=10,
                                      test_size=0.3,
                                      # Model parameters
                                      hidden_layer_sizes=(512, 256, 128,),
                                      activation_function='relu',
                                      dropout_rate=0.15,
                                      learning_rate=0.001,
                                      # Training parameters
                                      epochs=200,
                                      batch_size=32,
                                      patience=40,
                                      # Evaluation parameters
                                      bin_class_threshold=0.5,  # other values are bugged at the moment
                                      # Output control
                                      verbose=True,
                                      show_plots=True):
    """
    Runs the entire pipeline for movie recommendation using clustering and a Multi-Layer Neural Network (MLNN).

    The pipeline includes:
    1. Clustering of user ratings using the specified clustering method.
    2. Training and evaluation of a neural network model for each cluster to make movie recommendations.

    Parameters:
        ratings_df (pd.DataFrame): The ratings DataFrame.

        # Clustering parameters
        clustering_method (str): The clustering algorithm to use ('agglomerative' or 'spectral').
        num_clusters (int): Number of clusters to form.
        linkage_method (str): Linkage method for agglomerative clustering.
        min_cluster_size (int): Minimum size for clusters.
        large_cluster_threshold (float): Proportion threshold for defining large clusters.

        # KNN and Train-Test Split parameters
        k (int): Number of neighbors for KNN feature generation.
        test_size (float): Proportion of the dataset to include in the test split.

        # Model parameters
        hidden_layer_sizes (tuple): Sizes of the hidden layers in the neural network.
        activation_function (str): Activation function to use in the network layers.
        dropout_rate (float): Dropout rate for regularization in the neural network.
        learning_rate (float): Learning rate for model training.

        # Training parameters
        epochs (int): Number of epochs for model training.
        batch_size (int): Batch size for model training.
        patience (int): Number of epochs with no improvement before stopping training.

        # Evaluation parameters
        bin_class_threshold (float): The binary classification threshold for model evaluation.

        # Output control
        verbose (bool): Whether to print progress information.
        show_plots (bool): Whether to display plots of the results.
    """

    # Convert ratings to numpy array and create binary ratings (1 if watched, 0 otherwise)
    ratings_matrix = ratings_df.to_numpy()
    binary_ratings = (ratings_matrix > 0).astype(int)

    # Compute the distance matrix using Jaccard metric
    if verbose:
        print("\nComputing distance matrix...")
    distance_matrix = pairwise_distances(ratings_matrix > 0, metric='jaccard')

    # Perform clustering on the distance matrix
    clusters, silhouette_score, num_clusters, cluster_sizes, size_threshold = perform_clustering(
        clustering_method, distance_matrix, num_clusters, linkage_method,
        min_cluster_size, large_cluster_threshold, verbose, show_plots)

    # Train and evaluate a model for each cluster
    results = train_and_evaluate(
        clusters, num_clusters, cluster_sizes, size_threshold, binary_ratings, distance_matrix,
        k, test_size, hidden_layer_sizes, activation_function, dropout_rate,
        learning_rate, epochs, batch_size, patience, bin_class_threshold, large_cluster_threshold, verbose)

    results_df = pd.DataFrame(results)

    # Display results
    if verbose:
        pd.set_option('display.max_columns', None)
        print("\nCluster-wise Results:")
        print(results_df)

    # Plot results if required
    if show_plots:
        plot_results(results_df)


def perform_clustering(method, dist_matrix, k, linkage, min_size, large_cluster_threshold, verbose, plots):
    """
    Perform clustering on the distance matrix using the specified method.
    """
    method = method.lower()
    if method == 'agglomerative':
        clusters, silhouette_score = cl.agglomerative_clustering(dist_matrix, k, linkage, min_size, verbose, plots)
    elif method == 'spectral':
        clusters, silhouette_score = cl.spectral_clustering(dist_matrix, k, min_size, verbose, plots)
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'spectral'.")

    # Determine the number of clusters and their sizes
    num_clusters = len(np.unique(clusters))
    cluster_sizes = [np.sum(clusters == label) for label in range(num_clusters)]

    # Determine the size threshold for large clusters
    size_threshold = max(cluster_sizes) + 1
    if large_cluster_threshold > 0:
        sorted_sizes = sorted(cluster_sizes)
        threshold_index = int((1 - large_cluster_threshold) * len(sorted_sizes))
        size_threshold = sorted_sizes[threshold_index]

        # if verbose:
        #     print(f"\nCluster size threshold for large clusters: {size_threshold}")

    return clusters, silhouette_score, num_clusters, cluster_sizes, size_threshold


def train_and_evaluate(clusters, num_clusters, cluster_sizes, size_threshold, binary_ratings, dist_matrix,
                       k, test_size, hidden_layer_sizes, activation, dropout_rate,
                       learning_rate, epochs, batch_size, patience, bin_class_threshold, large_cluster_threshold, verbose):
    """
    Train and evaluate a model for each cluster.
    """
    results = {'Cluster': [], 'Size': [], 'Train Accuracy': [], 'Test Accuracy': [],
               'Train Precision': [], 'Test Precision': [], 'Train Recall': [],
               'Test Recall': [], 'Train AUC': [], 'Test AUC': []}
    models_dict = {}

    for cluster_label in range(num_clusters):
        cluster_results = train_model_for_cluster(
            cluster_label, clusters, binary_ratings, dist_matrix,
            k, test_size, hidden_layer_sizes, activation,
            dropout_rate, learning_rate, epochs, batch_size, patience, bin_class_threshold,
            large_cluster_threshold, size_threshold, verbose)

        # Store the results for each cluster
        results['Cluster'].append(cluster_label)
        results['Size'].append(cluster_sizes[cluster_label])
        results['Train Accuracy'].append(cluster_results['train_accuracy'])
        results['Test Accuracy'].append(cluster_results['test_accuracy'])
        results['Train Precision'].append(cluster_results['train_precision'])
        results['Test Precision'].append(cluster_results['test_precision'])
        results['Train Recall'].append(cluster_results['train_recall'])
        results['Test Recall'].append(cluster_results['test_recall'])
        results['Train AUC'].append(cluster_results['train_auc'])
        results['Test AUC'].append(cluster_results['test_auc'])

        models_dict[cluster_label] = {
            'model': cluster_results['model'],
            'X_test': cluster_results['X_test'],
            'y_test': cluster_results['y_test'],
        }

    return results


def get_k_nearest_neighbors(k, user_idx, dist_matrix):
    """
    Get k nearest neighbors of a user based on the distance matrix.
    """
    distances = dist_matrix[user_idx]
    nearest_neighbors = np.argsort(distances)[:k + 1]
    return nearest_neighbors[1:]


def prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix, k, test_size):
    """
    Prepare the feature vectors and labels for the users in a cluster.
    """
    feature_vectors = []
    labels = []

    for user_idx in user_indices:
        neighbors = get_k_nearest_neighbors(k, user_idx, dist_matrix)
        label = binary_ratings[user_idx]
        feature_vector = np.concatenate([binary_ratings[neighbor] for neighbor in neighbors], axis=0)
        feature_vectors.append(feature_vector)
        labels.append(label)

    X = np.array(feature_vectors)
    y = np.array(labels)

    return train_test_split(X, y, test_size=test_size)


def create_neural_network(hidden_layer_sizes, activation, dropout_rate, input_dim, output_dim):
    """
    Creates and returns a neural network model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layer_sizes:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(output_dim, activation='sigmoid'))
    return model


def train_model_for_cluster(cluster_label, clusters, binary_ratings, dist_matrix, k, test_size, hidden_layer_units,
                            activation, dropout_rate, learning_rate, epochs, batch_size, patience, bin_class_threshold,
                            large_cluster_threshold, size_threshold, verbose):
    """
    Train a neural network model for a specific cluster.
    """
    if verbose:
        print(f"\nTraining model for cluster {cluster_label}")

    # Identify users in the cluster
    user_indices = np.where(clusters == cluster_label)[0]

    # Adjust the architecture based on the cluster size
    if large_cluster_threshold > 0 and len(user_indices) >= size_threshold:
        adjusted_hidden_layer_units = [2 * hidden_layer_units[0]] + list(hidden_layer_units)
    else:
        adjusted_hidden_layer_units = hidden_layer_units

    # Prepare the data for training
    X_train, X_test, y_train, y_test = prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix, k, test_size)

    # Create and compile the neural network model
    model = create_neural_network(adjusted_hidden_layer_units, activation, dropout_rate, X_train.shape[1], y_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    # Set up early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              callbacks=[early_stopping], validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model on training and test data
    train_accuracy, train_precision, train_recall, train_auc = evaluate_model(model, X_train, y_train, bin_class_threshold)
    test_accuracy, test_precision, test_recall, test_auc = evaluate_model(model, X_test, y_test, bin_class_threshold)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "model": model,
        "X_test": X_test,
        "y_test": y_test
    }


def evaluate_model(model, X, y, threshold=0.5):
    """
    Evaluates the model on given data and returns accuracy, precision, recall, and AUC.
    """
    if threshold != 0.5:
        return evaluate_model_with_threshold(model, X, y, threshold)
    else:
        metric_results = model.evaluate(X, y, verbose=0)
        return metric_results[1], metric_results[2], metric_results[3], metric_results[4]


def evaluate_model_with_threshold(model, X, y, threshold=0.5):
    """
    Evaluates the model on given data and returns accuracy, precision, recall, and AUC,
    applying a custom threshold for classification.
    """
    # Get the predicted probabilities
    predicted_probs = model.predict(X)

    # Apply the threshold to get binary predictions
    predicted_classes = (predicted_probs >= threshold).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y)

    # Calculate precision, recall, and AUC
    # TODO find where these functions are implemented
    precision = metrics.precision_score(y, predicted_classes)
    recall = metrics.recall_score(y, predicted_classes)
    auc = metrics.roc_auc_score(y, predicted_probs)

    return accuracy, precision, recall, auc


def plot_results(results_df):
    """
    Plots the results for each metric by cluster.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.title(f'{metric} by Cluster')
        plt.bar(results_df['Cluster'], results_df[f'Train {metric}'], alpha=0.6, label='Train')
        plt.bar(results_df['Cluster'], results_df[f'Test {metric}'], alpha=0.6, label='Test')
        plt.xlabel('Cluster')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
