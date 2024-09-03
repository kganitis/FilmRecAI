import argparse

import preprocessing
import clustering
import recommendations

# Constants for execution speed
SLOW = 'slow'  # for final submission
MEDIUM = 'medium'  # for testing
FAST = 'fast'  # default; for quick results
VERY_FAST = 'very fast'  # for instant results

EXEC_SPEED = FAST  # Set the execution speed here

RUN_KMEANS_CLUSTERING_PIPELINE = False
RUN_RECOMMENDATIONS_PIPELINE = True


def run(R_min_kmeans, R_min, R_max, M_min):
    """
    Executes the data preprocessing, clustering, and recommendation pipelines.

    Args:
        R_min_kmeans (int): Minimum number of ratings for users/movies used in K-means clustering preprocessing.
        R_min (int): Minimum number of ratings required for users/movies in Jaccard clustering preprocessing.
        R_max (int): Maximum number of ratings considered for users/movies in Jaccard clustering preprocessing.
        M_min (int): Minimum number of movies considered for clustering in Jaccard clustering preprocessing.

    Steps:
    1. Runs the preprocessing pipeline for K-means clustering.
    2. Performs K-means clustering with different numbers of clusters (L) and two distance metrics.
    3. Runs the preprocessing pipeline for Jaccard clustering.
    4. Executes the movie recommendation pipeline using the processed data.
    """
    if RUN_KMEANS_CLUSTERING_PIPELINE:
        ratings_df = preprocessing.run_kmeans_clustering_pipeline(R_min=R_min_kmeans, R_max=200, M_min=50)

        L_values = [5, 7, 10, 15, 20]
        metrics = ['euclidean_generalized', 'cosine_generalized']

        # Perform K-means clustering with the specified metrics and L values
        for L in L_values:
            for metric in metrics:
                clustering.kmeans_clustering(ratings_df, L, metric)

    if RUN_RECOMMENDATIONS_PIPELINE:
        ratings_df = preprocessing.run_jaccard_clustering_pipeline(R_min, R_max, M_min)
        recommendations.run_recommendation_pipeline(ratings_df)


def get_parameters_for_speed(speed=FAST):
    """
    Retrieves the preprocessing parameters based on the specified execution speed.

    Args:
        speed (str): The execution speed, which determines the parameter values. Can be 'slow', 'medium', 'fast', or 'very fast'.

    Returns:
        tuple: A tuple containing four integers:
            - R_min_kmeans: Minimum ratings for K-means clustering preprocessing.
            - R_min: Minimum ratings for Jaccard clustering preprocessing.
            - R_max: Maximum ratings for Jaccard clustering preprocessing.
            - M_min: Minimum movies considered for Jaccard clustering.
    """
    speed = speed.lower()
    if speed == MEDIUM:
        R_min_kmeans = 100
        R_min, R_max, M_min = 5, 15, 42
    elif speed == SLOW:
        R_min_kmeans = 50
        R_min, R_max, M_min = 5, 15, 40
    elif speed == VERY_FAST:
        R_min_kmeans = 150
        R_min, R_max, M_min = 5, 15, 46
    else:  # FAST
        R_min_kmeans = 125
        R_min, R_max, M_min = 5, 15, 44

    return R_min_kmeans, R_min, R_max, M_min


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run the clustering and recommendation pipelines with different execution speeds.")

    parser.add_argument(
        '--speed',
        type=str,
        choices=[SLOW, MEDIUM, FAST],
        default=EXEC_SPEED,
        help=f'Execution speed: slow, medium, fast (default: {EXEC_SPEED})'
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main pipeline with parameters based on the chosen execution speed
    run(*get_parameters_for_speed(args.speed))
