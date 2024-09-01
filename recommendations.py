import numpy as np
import pandas as pd
from keras import layers, models
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import clustering


def run(ratings_df, L=5, k=5, test_size=0.2):
    # Convert the DataFrame to a numpy array for easier manipulation
    ratings_matrix = ratings_df.to_numpy()

    # Results dictionary to store cluster-wise accuracy
    results = {
        'Cluster': [],
        'Train Accuracy': [],
        'Test Accuracy': []
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
        print(f"\nTraining model for cluster {cluster_label}")

        user_indices = np.where(clusters == cluster_label)[0]

        # Prepare data for the cluster
        train_test_split_list = prepare_data_for_cluster(ratings_matrix, user_indices, dist_matrix, k, test_size)
        X_train, X_test, y_train, y_test = train_test_split_list

        # Create a neural network model
        model = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(4096, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(y_train.shape[1], activation='sigmoid')  # output layer for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # Evaluate the model
        train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

        # Store the results
        results['Cluster'].append(cluster_label)
        results['Train Accuracy'].append(train_accuracy)
        results['Test Accuracy'].append(test_accuracy)

        # Store the model and test data for later analysis
        models_dict[cluster_label] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }

    # Present results as a table
    results_df = pd.DataFrame(results)
    print("\nCluster-wise Accuracy Results:")
    print(results_df)

    # Calculate statistics for each cluster and store the results in a list
    cluster_results = []
    for cluster_label in range(L):
        stats = cluster_statistics(models_dict, cluster_label)
        cluster_results.append(stats)

    # Convert the results to a DataFrame for better readability
    cluster_results_df = pd.DataFrame(cluster_results)

    # Calculate the gap between watched and non-watched averages
    cluster_results_df['Gap'] = cluster_results_df['Watched Avg'] - cluster_results_df['Non-Watched Avg']

    # Calculate overall averages
    overall_watched_avg = cluster_results_df['Watched Avg'].mean()
    overall_non_watched_avg = cluster_results_df['Non-Watched Avg'].mean()
    overall_gap_avg = cluster_results_df['Gap'].mean()

    # Display the results
    print(cluster_results_df)

    print(f"\nOverall Watched Avg: {overall_watched_avg:.3f}")
    print(f"Overall Non-Watched Avg: {overall_non_watched_avg:.3f}")
    print(f"Overall Gap Avg: {overall_gap_avg:.3f}")

    # Plot the results
    plot_cluster_statistics(cluster_results_df, overall_watched_avg, overall_non_watched_avg, overall_gap_avg)


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


def cluster_statistics(models_dict, cluster_label):
    """Calculate and return average and median probabilities for watched and non-watched movies for all users in a cluster."""
    model_info = models_dict[cluster_label]
    model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']

    # Lists to store probabilities for all users
    all_watched_probs = []
    all_non_watched_probs = []

    # Iterate over all users in the cluster
    for user_idx in range(X_test.shape[0]):
        actual_ratings = y_test[user_idx]
        predicted_probabilities = model.predict(X_test[user_idx].reshape(1, -1), verbose=0).flatten()

        # Separate watched and non-watched movies
        watched_indices = np.nonzero(actual_ratings)[0]
        non_watched_indices = np.where(actual_ratings == 0)[0]

        # Add the probabilities to the lists
        all_watched_probs.extend(predicted_probabilities[watched_indices])
        all_non_watched_probs.extend(predicted_probabilities[non_watched_indices])

    # Calculate statistics for watched movies
    watched_avg = np.mean(all_watched_probs) if all_watched_probs else float('nan')

    # Calculate statistics for non-watched movies
    non_watched_avg = np.mean(all_non_watched_probs) if all_non_watched_probs else float('nan')

    return {
        'Cluster': cluster_label,
        'Watched Avg': watched_avg,
        'Non-Watched Avg': non_watched_avg,
    }


def plot_cluster_statistics(cluster_results_df, overall_watched_avg, overall_non_watched_avg, overall_gap_avg):
    clusters = cluster_results_df['Cluster']
    watched_avg = cluster_results_df['Watched Avg']
    non_watched_avg = cluster_results_df['Non-Watched Avg']
    gap = cluster_results_df['Gap']

    # Create a bar plot for watched and non-watched averages
    plt.figure(figsize=(14, 8))

    bar_width = 0.35
    index = np.arange(len(clusters))

    # Add horizontal lines for overall averages and gap
    plt.axhline(y=overall_non_watched_avg, color='green', linestyle='-', linewidth=2,
                label=f'Overall Watched Avg: {overall_watched_avg:.3f}')
    plt.axhline(y=overall_watched_avg, color='red', linestyle='-', linewidth=2,
                label=f'Overall Non-Watched Avg: {overall_non_watched_avg:.3f}')
    plt.axhline(y=overall_gap_avg, color='blue', linestyle='-', linewidth=2,
                label=f'Overall Gap Avg: {overall_gap_avg:.3f}')

    plt.bar(index, watched_avg, bar_width, label='Watched Avg', color='green', alpha=0.7)
    plt.bar(index + bar_width, non_watched_avg, bar_width, label='Non-Watched Avg', color='red', alpha=0.7)

    # Plot the gap as a line plot
    plt.plot(index + bar_width / 2, gap, label='Gap (Watched - Non-Watched)', color='blue', marker='o')

    # Add labels and titles
    plt.xlabel('Cluster')
    plt.ylabel('Average Probability')
    plt.title('Cluster-Wise Watched vs Non-Watched Averages')
    plt.xticks(index + bar_width / 2, clusters)
    plt.legend()

    # Show only horizontal grid lines
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    # Show the plot
    plt.tight_layout()
    plt.show()
