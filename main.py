import argparse

import pandas as pd

import clustering
import preprocessing
import recommendations
from metrics import euclidean_distance_generalized, cosine_distance_generalized

# Constants for execution speed
SLOW = 'slow'  # only for final submission
MEDIUM = 'medium'  # for testing
FAST = 'fast'  # default, for quick results

EXEC_SPEED = FAST  # set for the desired execution speed


def run(R_min, R_max, M_min):
    # Run data preprocessing
    # R_min, R_max, M_min = 50, 100, 0
    ratings_df = preprocessing.run(R_min, R_max, M_min, False)

    # Run K-means clustering using the two metrics and different values of L
    # L_values = [5, 7, 10, 15, 20]
    # metrics = [euclidean_distance_generalized, cosine_distance_generalized]
    # for L in L_values:
    #     for metric in metrics:
    #         clustering.kmeans_clustering(data, metric, L)

    recommendations.run(ratings_df)


def run_experiments(ratings_df):
    # Define the parameter grid
    L_values = [5, 15, 30, 45]  # Different values of L (number of clusters)
    k_values = [3, 5, 7]  # Different values of k (number of neighbors)
    test_size_values = [0.1, 0.2]  # Different values of test_size (proportion of test data)
    hidden_layer_units_values = [
        (256, 128),
        (1024, 512),
        (4096, 2048),
    ]  # Different configurations of hidden layers

    results = []

    # Loop over each combination of parameters
    for L in L_values:
        for k in k_values:
            for test_size in test_size_values:
                for hidden_layer_units in hidden_layer_units_values:
                    print(f"Running with L={L}, k={k}, test_size={test_size}, hidden_layers={hidden_layer_units}")
                    # Run the model with the current set of parameters
                    result = recommendations.run(
                        ratings_df,
                        L=L,
                        delta=5.0,
                        k=k,
                        test_size=test_size,
                        hidden_layer_units=hidden_layer_units,
                        verbose=False,
                        plots=False
                    )
                    # Append the result with the current parameters
                    result.update({
                        "L": L,
                        "k": k,
                        "test_size": test_size,
                        "hidden_layers": hidden_layer_units
                    })
                    results.append(result)

                    print(result)

    # Convert the results to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    print("\nExperiment Results:")
    print(results_df)

    # Optionally, save the results to a CSV file for further analysis
    results_df.to_csv('experiment_results.csv', index=False)

    return results_df


def get_parameters_for_speed(speed=FAST):
    """Get parameters based on execution speed."""
    speed = speed.lower()
    if speed == MEDIUM:
        R_min, R_max, M_min = 50, 200, 10
    elif speed == SLOW:
        R_min, R_max, M_min = 50, 200, 5
    else:  # FAST
        R_min, R_max, M_min = 50, 200, 17
    return R_min, R_max, M_min


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run with different execution speeds.")
    parser.add_argument(
        '--speed',
        type=str,
        choices=[SLOW, MEDIUM, FAST],
        default=EXEC_SPEED,
        help=f'Execution speed: slow, medium, fast (default: {EXEC_SPEED})'
    )

    args = parser.parse_args()
    run(*get_parameters_for_speed(args.speed))
