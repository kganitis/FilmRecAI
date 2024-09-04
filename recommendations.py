import numpy as np
import pandas as pd
from keras import layers, models
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import clustering as cl
from plotting import plot_training_results

config = {
    # Agglomerative Clustering parameters
    'L': 30,  # Initial number of clusters to form (before merging small clusters into larger ones)
    'min_cluster_size': 500,  # Minimum size for clusters
    'large_cluster_threshold': 0.1667,  # Threshold for defining large clusters (model adjusts for those)

    # KNN and Train-Test Split parameters
    'k': 10,  # Number of neighbors for KNN feature generation
    'test_size': 0.3,  # Proportion of the dataset to include in the test split

    # Neural Network Training parameters
    'hidden_layers': (512, 256, 128),  # Sizes of the hidden layers in the neural network
    'metrics': ['Accuracy', 'Precision', 'Recall', 'AUC'],  # Metrics to evaluate the model
    'epochs': 200,  # Number of epochs for model training
    'learning_rate': 0.001,  # Learning rate for model training
    'patience': 40,  # Number of epochs with no improvement before stopping training
    'dropout': 0.15,  # Dropout rate for regularization in the neural network
    'batch_size': 32,  # Batch size for model training
}


def run_recommendation_pipeline(ratings_df):
    """
    Runs the entire pipeline for movie recommendation using clustering and a Multi-Layer Neural Network.
    """
    # Convert ratings to numpy array and create binary ratings (1 if rated, 0 otherwise)
    ratings_matrix = ratings_df.to_numpy()
    binary_ratings = (ratings_matrix > 0).astype(int)

    # Compute the distance matrix using Jaccard metric
    print("\nComputing distance matrix...")
    dist_matrix = pairwise_distances(ratings_matrix > 0, metric='jaccard')

    # Perform clustering on the distance matrix
    clusters, num_clusters, size_threshold = perform_clustering(dist_matrix)

    # Train and evaluate a model for each cluster
    results = train_and_evaluate(clusters, num_clusters, size_threshold, binary_ratings, dist_matrix)

    # Display results
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    print("\nCluster-wise Results:")
    print(results_df)

    plot_training_results(results_df, config['metrics'])


def perform_clustering(dist_matrix):
    """
    Perform agglomerative clustering on the distance matrix.
    """
    clusters = cl.agglomerative_clustering(dist_matrix, config['L'], config['min_cluster_size'])

    # Determine the number of clusters formed and their sizes
    num_clusters = len(np.unique(clusters))
    cluster_sizes = [np.sum(clusters == label) for label in range(num_clusters)]

    # Determine the size threshold for large clusters
    size_threshold = max(cluster_sizes) + 1
    if config['large_cluster_threshold'] > 0:
        sorted_sizes = sorted(cluster_sizes)
        threshold_index = int((1 - config['large_cluster_threshold']) * len(sorted_sizes))
        size_threshold = sorted_sizes[threshold_index]

    return clusters, num_clusters, size_threshold


def train_and_evaluate(clusters, num_clusters, size_threshold, binary_ratings, dist_matrix):
    """
    Train and evaluate a model for each cluster.
    """
    results = []

    for cluster_label in range(num_clusters):
        print(f"\nTraining model for cluster {cluster_label}")
        user_indices = np.where(clusters == cluster_label)[0]  # Identify users in the cluster
        cluster_result = {"Cluster": cluster_label, "Size": len(user_indices)}
        # Train model for the specific user indices
        train_result = train_model_for_cluster(user_indices, size_threshold, binary_ratings, dist_matrix)
        cluster_result.update(train_result)
        results.append(cluster_result)

    return results


def get_k_nearest_neighbors(user_idx, dist_matrix):
    """
    Get k nearest neighbors of a user based on the distance matrix.
    """
    distances = dist_matrix[user_idx]
    nearest_neighbors = np.argsort(distances)[:config['k'] + 1]
    return nearest_neighbors[1:]


def prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix):
    """
    Prepare the feature vectors and labels for the users in a cluster.
    """
    feature_vectors, labels = [], []

    for user_idx in user_indices:
        neighbors = get_k_nearest_neighbors(user_idx, dist_matrix)
        label = binary_ratings[user_idx]
        feature_vector = np.concatenate([binary_ratings[neighbor] for neighbor in neighbors], axis=0)
        feature_vectors.append(feature_vector)
        labels.append(label)

    X = np.array(feature_vectors)
    y = np.array(labels)

    return train_test_split(X, y, test_size=config['test_size'])


def create_neural_network(hidden_layers, input_dim, output_dim):
    """
    Creates and returns a neural network model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(config['dropout']))

    model.add(layers.Dense(output_dim, activation='sigmoid'))
    return model


def adjust_hidden_layers(cluster_size, size_threshold):
    """
    Add an extra hidden layer for large clusters.
    """
    hidden_layers = config['hidden_layers']
    if config['large_cluster_threshold'] > 0 and cluster_size >= size_threshold:
        return [2 * hidden_layers[0]] + list(hidden_layers)
    return hidden_layers


def train_model_for_cluster(user_indices, size_threshold, binary_ratings, dist_matrix):
    """
    Train a neural network model for the users of a specific cluster.
    """
    # Prepare the data for training
    X_train, X_test, y_train, y_test = prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix)

    # Adjust the model architecture based on the cluster size
    hidden_layers = adjust_hidden_layers(len(user_indices), size_threshold)

    # Create and compile the neural network model
    metrics = config['metrics']
    model = create_neural_network(hidden_layers, X_train.shape[1], y_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='binary_crossentropy', metrics=metrics)

    # Train the model
    train_and_fit_model(model, X_train, y_train, X_test, y_test)

    # Evaluate the model on training and test data
    train_metrics = model.evaluate(X_train, y_train, verbose=0)[1:]
    test_metrics = model.evaluate(X_test, y_test, verbose=0)[1:]

    cluster_result = {}
    for i in range(len(metrics)):
        cluster_result[f"Train {metrics[i]}"] = train_metrics[i]
        cluster_result[f"Test {metrics[i]}"] = test_metrics[i]

    return cluster_result


def train_and_fit_model(model, X_train, y_train, X_test, y_test):
    """
    Fits the neural network model with early stopping.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        verbose=1,
        restore_best_weights=True
    )

    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
              callbacks=[early_stopping], validation_data=(X_test, y_test), verbose=1)
