import numpy as np
import pandas as pd
from keras import layers, models
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import clustering as cl


config = {
    # Clustering parameters
    'L': 30,  # Number of clusters to form
    'min_cluster_size': 1000,  # Minimum size for clusters
    'large_cluster_threshold': 0.1667,  # Threshold for defining large clusters

    # KNN and Train-Test Split parameters
    'k': 10,  # Number of neighbors for KNN feature generation
    'test_size': 0.3,  # Proportion of the dataset to include in the test split

    # Model parameters
    'hidden_layers': (512, 256, 128),  # Sizes of the hidden layers in the neural network
    'dropout': 0.15,  # Dropout rate for regularization in the neural network
    'learning_rate': 0.001,  # Learning rate for model training

    # Training parameters
    'epochs': 200,  # Number of epochs for model training
    'batch_size': 32,  # Batch size for model training
    'patience': 40,  # Number of epochs with no improvement before stopping training

    # Evaluation parameters
    'bin_class_threshold': 0.5,  # The binary classification threshold for model evaluation  # TODO Only 0.5 works

    # Output control
    'verbose': True,  # Whether to print progress information
    'show_plots': True  # Whether to display plots of the results
}


def run_recommendation_pipeline(ratings_df):
    """
    Runs the entire pipeline for movie recommendation using clustering and a Multi-Layer Neural Network (MLNN).
    """

    # Convert ratings to numpy array and create binary ratings (1 if watched, 0 otherwise)
    ratings_matrix = ratings_df.to_numpy()
    binary_ratings = (ratings_matrix > 0).astype(int)

    # Compute the distance matrix using Jaccard metric
    log("\nComputing distance matrix...")
    dist_matrix = pairwise_distances(ratings_matrix > 0, metric='jaccard')

    # Perform clustering on the distance matrix
    clusters, silhouette_score, num_clusters, cluster_sizes, size_threshold = perform_clustering(dist_matrix)

    # Train and evaluate a model for each cluster
    results = train_and_evaluate(clusters, num_clusters, cluster_sizes, size_threshold, binary_ratings, dist_matrix)

    results_df = pd.DataFrame(results)

    # Display results
    log("\nCluster-wise Results:")
    log(results_df)

    # Plot results
    plot_results(results_df)


def perform_clustering(dist_matrix):
    """
    Perform agglomerative clustering on the distance matrix.
    """
    clusters, silhouette_score = cl.agglomerative_clustering(
        dist_matrix, config['L'], config['min_cluster_size'], config['verbose'], config['show_plots'])

    # Determine the number of clusters formed and their sizes
    num_clusters = len(np.unique(clusters))
    cluster_sizes = [np.sum(clusters == label) for label in range(num_clusters)]

    # Determine the size threshold for large clusters
    size_threshold = max(cluster_sizes) + 1
    if config['large_cluster_threshold'] > 0:
        sorted_sizes = sorted(cluster_sizes)
        threshold_index = int((1 - config['large_cluster_threshold']) * len(sorted_sizes))
        size_threshold = sorted_sizes[threshold_index]

    return clusters, silhouette_score, num_clusters, cluster_sizes, size_threshold


def train_and_evaluate(clusters, num_clusters, cluster_sizes, size_threshold, binary_ratings, dist_matrix):
    """
    Train and evaluate a model for each cluster.
    """
    results = []

    for cluster_label in range(num_clusters):
        result = train_model_for_cluster(
            cluster_label, clusters, size_threshold, binary_ratings, dist_matrix)
        result['Cluster'] = cluster_label
        result['Size'] = cluster_sizes[cluster_label]
        results.append(result)

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


def adjust_hidden_layers(cluster_size, size_threshold):
    """
    Adjust hidden layers based on cluster size.
    """
    hidden_layers = config['hidden_layers']
    if config['large_cluster_threshold'] > 0 and cluster_size >= size_threshold:
        return [2 * hidden_layers[0]] + list(hidden_layers)
    return hidden_layers


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


def train_model_for_cluster(cluster_label, clusters, size_threshold, binary_ratings, dist_matrix):
    """
    Train a neural network model for a specific cluster.
    """
    log(f"\nTraining model for cluster {cluster_label}")

    # Identify users in the cluster
    user_indices = np.where(clusters == cluster_label)[0]

    # Prepare the data for training
    X_train, X_test, y_train, y_test = prepare_data_for_cluster(binary_ratings, user_indices, dist_matrix)

    # Adjust the architecture based on the cluster size
    adjusted_hidden_layers = adjust_hidden_layers(len(user_indices), size_threshold)

    # Create and compile the neural network model
    model = create_neural_network(adjusted_hidden_layers, X_train.shape[1], y_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    # Train the model
    train_and_fit_model(model, X_train, y_train, X_test, y_test)

    # Evaluate the model on training and test data
    train_metrics = evaluate_model(model, X_train, y_train, config['bin_class_threshold'])
    test_metrics = evaluate_model(model, X_test, y_test, config['bin_class_threshold'])

    return {
        "train_accuracy": train_metrics[0],
        "test_accuracy": test_metrics[0],
        "train_precision": train_metrics[1],
        "test_precision": test_metrics[1],
        "train_recall": train_metrics[2],
        "test_recall": test_metrics[2],
        "train_auc": train_metrics[3],
        "test_auc": test_metrics[3]
    }


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


def evaluate_model(model, X, y, threshold=0.5):
    """
    Evaluates the model on given data and returns accuracy, precision, recall, and AUC.
    """
    if threshold != 0.5:
        return evaluate_model_with_threshold(model, X, y, threshold)
    else:
        return model.evaluate(X, y, verbose=0)[1:]


def evaluate_model_with_threshold(model, X, y, threshold=0.5):
    """
    Evaluates the model on given data and returns accuracy, precision, recall, and AUC,
    applying a custom threshold for classification.
    """
    # Generate predicted probabilities
    y_pred_prob = model.predict(X)

    # Apply the threshold to convert probabilities into binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Calculate metrics
    # TODO Debug calculation
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_pred_prob, average='macro', multi_class='ovr')

    return accuracy, precision, recall, auc


def plot_results(results_df):
    """
    Plots the results for each metric by cluster.
    """
    if not config['show_plots']:
        return

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


def log(statement):
    if config['verbose']:
        print(statement)
