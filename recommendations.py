import numpy as np
import pandas as pd
from keras import layers, models
from keras.src.metrics import Precision, Recall
from keras.src.optimizers import Adam
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from clustering import agglomerative_clustering
from f1_score import F1ScoreCallback, f1_score
from plotting import plot_training_results

config = {
    'L': 5,  # Number of clusters to form
    'k': 20,  # Number of neighbors for KNN feature generation
    'test_size': 0.1,  # Proportion of the dataset to include in the test split
    'hidden_layers': (512, 256),  # Sizes of the hidden layers in the neural network
    'metrics': [Precision(name="Precision"), Recall(name="Recall")],  # Metrics to evaluate the model
    'epochs': 50,  # Number of epochs for model training
    'learning_rate': 0.0001,  # Learning rate for model training
    'patience': 10,  # Number of epochs with no improvement before stopping training
    'threshold': 0.5  # Binary classification threshold for calculating F1 Score
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


def train_and_evaluate_for_cluster(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a neural network model on training and test data for a cluster of users.
    """
    # Create a neural network model
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for units in config['hidden_layers']:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='sigmoid'))

    val_data = (X_test, y_test)
    metrics = config['metrics']

    # Compile the neural network model
    model.compile(optimizer=Adam(
        learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=metrics)

    # Set a callback for monitoring F1 score, implementing patience-based stopping
    # and applying a custom binary classification threshold
    f1_score_callback = F1ScoreCallback(
        validation_data=val_data,
        patience=config['patience'],
        threshold=config['threshold'])

    # Train the model with the F1 score callback
    model.fit(X_train, y_train, epochs=config['epochs'], validation_data=val_data, callbacks=[f1_score_callback])

    # Evaluate the model on training and test data
    train_metrics = model.evaluate(X_train, y_train, verbose=0)[1:]
    test_metrics = model.evaluate(X_test, y_test, verbose=0)[1:]
    train_precision, train_recall = train_metrics[0], train_metrics[1]
    test_precision, test_recall = test_metrics[0], test_metrics[1]

    # Return a dictionary with the evaluation results
    cluster_result = {}
    for i in range(len(metrics)):
        cluster_result[f"Train {metrics[i].name}"] = train_metrics[i]
        cluster_result[f"Test {metrics[i].name}"] = test_metrics[i]
    cluster_result[f"Train F1 Score"] = f1_score(train_precision, train_recall)
    cluster_result[f"Test F1 Score"] = f1_score(test_precision, test_recall)

    return cluster_result


def train_and_evaluate(clusters, binary_ratings, dist_matrix):
    """
    Train and evaluate a neural network model for each cluster of users.
    """
    results = []

    for cluster_label in range(len(np.unique(clusters))):
        print(f"\nTraining model for cluster {cluster_label}")

        # Get the indices of users in the cluster
        user_indices = np.where(clusters == cluster_label)[0]

        # Prepare the data for training and testing
        X_train, X_test, y_train, y_test = prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix)

        # Train and evaluate a model for the cluster
        train_result = train_and_evaluate_for_cluster(X_train, X_test, y_train, y_test)

        # Update the results
        cluster_result = {"Cluster": cluster_label, "Size": len(user_indices)}
        cluster_result.update(train_result)
        results.append(cluster_result)

    return results
