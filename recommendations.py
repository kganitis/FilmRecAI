import numpy as np
import pandas as pd
from keras import layers, models
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from keras.src.metrics import Precision, Recall
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from clustering import agglomerative_clustering
from plotting import plot_training_results


config = {
    # Agglomerative Clustering parameters
    'L': 5,  # Number of clusters to form
    'large_cluster_threshold': 0.2,  # Threshold for defining large clusters (model adjusts for those)

    # KNN and Train-Test Split parameters
    'k': 10,  # Number of neighbors for KNN feature generation
    'test_size': 0.1,  # Proportion of the dataset to include in the test split

    # Neural Network Training parameters
    'hidden_layers': (8192, 4096),  # Sizes of the hidden layers in the neural network
    'metrics': [Precision(name="Precision"), Recall(name="Recall")],  # Metrics to evaluate the model
    'epochs': 50,  # Number of epochs for model training
    'learning_rate': 0.0001,  # Learning rate for model training
    'patience': 5,  # Number of epochs with no improvement before stopping training
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
    clusters = agglomerative_clustering(dist_matrix, config['L'])

    # Train and evaluate a model for each cluster
    results = train_and_evaluate(clusters, binary_ratings, dist_matrix)

    # Display results
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    print("\nCluster-wise Results:")
    print(results_df)

    # Plot results
    metric_names = [metric.name for metric in config['metrics']] + ["F1 Score"]
    plot_training_results(results_df, metric_names)


def determine_large_cluster_size_threshold(clusters):
    """
    Determine the size threshold for defining large clusters.
    Training models for clusters larger than this threshold will have different hyperparameters.
    """
    num_clusters = len(np.unique(clusters))
    cluster_sizes = [np.sum(clusters == label) for label in range(num_clusters)]

    size_threshold = np.inf
    if config['large_cluster_threshold'] > 0:
        sorted_sizes = sorted(cluster_sizes)
        threshold_index = int((1 - config['large_cluster_threshold']) * num_clusters)
        size_threshold = sorted_sizes[threshold_index]

    return size_threshold


def get_k_nearest_neighbors(user_idx, dist_matrix):
    """
    Get k nearest neighbors of a user based on the distance matrix.
    """
    distances = dist_matrix[user_idx]
    nearest_neighbors = np.argsort(distances)[:config['k'] + 1]
    return nearest_neighbors[1:]


def prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix):
    """
    Prepare the data for training and testing a model for a cluster.
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


def train_and_evaluate_for_cluster(X_train, X_test, y_train, y_test, is_large_cluster=False):
    """
    Train and evaluate a neural network model on training and test data for a cluster of users.
    """
    # Create a neural network model
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for units in config['hidden_layers']:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='sigmoid'))

    metrics = config['metrics']

    # Compile the neural network model
    model.compile(optimizer=Adam(
        learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=metrics)

    # Define early stopping to prevent overfitting
    # Parameters are adjusted depending on the cluster size
    early_stopping = EarlyStopping(
        monitor='val_loss',
        # Set double patience for small clusters
        patience=config['patience'] if is_large_cluster else config['patience'] * 2,
        verbose=1,
        # Don't restore weights for the large clusters
        # Metrics continue improving for about 5 epochs even if loss is increasing
        restore_best_weights=not is_large_cluster
    )

    # Train the model
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=32,
              callbacks=[early_stopping], validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model on training and test data
    train_metrics = model.evaluate(X_train, y_train, verbose=0)[1:]
    test_metrics = model.evaluate(X_test, y_test, verbose=0)[1:]

    train_precision, train_recall = train_metrics[0], train_metrics[1]
    test_precision, test_recall = test_metrics[0], test_metrics[1]

    # Also evaluate F1 Score
    def f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    # Return a dictionary with the evaluation results
    cluster_result = {}
    for i in range(len(metrics)):
        cluster_result[f"Train {metrics[i].name}"] = train_metrics[i]
        cluster_result[f"Train F1 Score"] = f1_score(train_precision, train_recall)
        cluster_result[f"Test {metrics[i].name}"] = test_metrics[i]
        cluster_result[f"Test F1 Score"] = f1_score(test_precision, test_recall)

    return cluster_result


def train_and_evaluate(clusters, binary_ratings, dist_matrix):
    """
    Train and evaluate a neural network model for each cluster of users.
    """
    results = []
    large_cluster_size_threshold = determine_large_cluster_size_threshold(clusters)

    for cluster_label in range(len(np.unique(clusters))):
        print(f"\nTraining model for cluster {cluster_label}")

        # Get the indices of users in the cluster
        user_indices = np.where(clusters == cluster_label)[0]

        # Determine if it's a large cluster
        cluster_size = len(user_indices)
        is_large_cluster = cluster_size >= large_cluster_size_threshold

        # Prepare the data for training and testing
        X_train, X_test, y_train, y_test = prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix)

        # Train and evaluate a model for the cluster
        train_result = train_and_evaluate_for_cluster(X_train, X_test, y_train, y_test, is_large_cluster)

        # Update the results
        cluster_result = {"Cluster": cluster_label, "Size": cluster_size}
        cluster_result.update(train_result)
        results.append(cluster_result)

    return results
