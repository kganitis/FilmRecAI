import random

import numpy as np
import pandas as pd
from keras import layers, models
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import clustering


def run(ratings_df, L=5, k=5, hidden_layer_sizes=None, test_size=0.1):
    # Convert the DataFrame to a numpy array for easier manipulation
    ratings_matrix = ratings_df.to_numpy()

    # Results dictionary to store cluster-wise MAE
    results = {
        'Cluster': [],
        'Train MAE': [],
        'Test MAE': []
    }

    # Dictionary to store models and test data for each cluster
    models_dict = {}

    # Compute distance matrix
    print("\nComputing distance matrix...")
    binary_ratings = ratings_matrix > 0  # convert to boolean
    dist_matrix = pairwise_distances(binary_ratings, metric='jaccard')

    # Perform spectral clustering
    clusters = clustering.spectral_clustering(dist_matrix, L)

    # Train a model for each cluster
    for cluster_label in range(L):
        print(f"\nTraining model for cluster {cluster_label}\n")

        user_indices = np.where(clusters == cluster_label)[0]

        # Prepare data for the cluster
        train_test_split_list = prepare_data_for_cluster(ratings_matrix, user_indices, dist_matrix, k, test_size)
        X_train, X_test, y_train, y_test = train_test_split_list

        # Create a neural network model
        model = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(y_train.shape[1])  # output layer should match the number of movies
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Predict on the training and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate the model
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Store the results
        results['Cluster'].append(cluster_label)
        results['Train MAE'].append(train_mae)
        results['Test MAE'].append(test_mae)

        # Store the model and test data for later analysis
        models_dict[cluster_label] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }

    # Present results as a table
    results_df = pd.DataFrame(results)
    print("\nCluster-wise MAE Results:")
    print(results_df)

    # Test on a number of random users from each cluster
    for cluster_label in range(L):
        test_random_users(models_dict, cluster_label, n_users=1)


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


def test_random_users(models_dict, cluster_label, n_users, n_ratings=10):
    """Test the model on n_users random users from the cluster, showing n_ratings non-zero ratings for each user."""
    model_info = models_dict[cluster_label]
    model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']

    # Select n_users random users
    user_indices = random.sample(range(X_test.shape[0]), min(n_users, X_test.shape[0]))

    for user_idx in user_indices:
        actual_ratings = y_test[user_idx]
        predicted_ratings = model.predict(X_test[user_idx].reshape(1, -1), verbose=0).flatten()

        # Filter non-zero actual ratings
        non_zero_indices = np.nonzero(actual_ratings)
        actual_ratings_non_zero = actual_ratings[non_zero_indices]
        predicted_ratings_non_zero = predicted_ratings[non_zero_indices]

        # Select n_ratings random ratings
        selected_indices = random.sample(range(len(actual_ratings_non_zero)),
                                         min(n_ratings, len(actual_ratings_non_zero)))

        print(f"\nUser {user_idx} in Cluster {cluster_label}:")
        print("Actual Ratings:    ", [f"{actual_ratings_non_zero[i]:.2f}" for i in selected_indices])
        print("Predicted Ratings: ", [f"{predicted_ratings_non_zero[i]:.2f}" for i in selected_indices])
