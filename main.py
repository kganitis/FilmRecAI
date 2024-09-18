import preprocessing
import clustering
import recommendations

# Constants for execution speed
SLOW = 'slow'  # for final submission
MEDIUM = 'medium'  # for performance testing
FAST = 'fast'  # default; for quick results
MAX = 'max'  # for error testing

EXEC_SPEED = MEDIUM  # Set the execution speed here

RUN_KMEANS_CLUSTERING_PIPELINE = True
RUN_RECOMMENDATIONS_PIPELINE = True


def run(parameters):
    """
    Executes the data preprocessing, clustering, and recommendation pipelines.

    Steps:
    1. Runs the preprocessing pipeline for K-means clustering.
    2. Performs K-means clustering with different numbers of clusters (L) and distance metrics.
    3. Runs the preprocessing pipeline for the recommendations model.
    4. Executes the movie recommendation pipeline using the processed data.
    """
    if RUN_KMEANS_CLUSTERING_PIPELINE:
        R_min, R_max, M_min = parameters[0]
        ratings_df = preprocessing.run_for_kmeans_clustering_pipeline(R_min, R_max, M_min)

        L_values = [5, 7, 10, 15, 20]
        metrics = ['euclidean_generalized', 'cosine_generalized']

        # Perform K-means clustering with the specified metrics and L values
        for L in L_values:
            for metric in metrics:
                clustering.kmeans_clustering(ratings_df, L, metric)

    if RUN_RECOMMENDATIONS_PIPELINE:
        R_min, R_max, M_min = parameters[1]
        ratings_df = preprocessing.run_for_recommendations_pipeline(R_min, R_max, M_min)
        recommendations.run_recommendation_pipeline(ratings_df)


def get_parameters_for_speed(speed='fast'):
    """
    Retrieves the preprocessing parameters based on the specified execution speed.
    """
    speed = speed.lower()
    if speed == 'slow':
        R_min_1, R_max_1, M_min_1 = 50, 200, 50
        R_min_2, R_max_2, M_min_2 = 5, 15, 40
    elif speed == 'medium':
        R_min_1, R_max_1, M_min_1 = 100, 200, 50
        R_min_2, R_max_2, M_min_2 = 5, 15, 42
    elif speed == 'fast':
        R_min_1, R_max_1, M_min_1 = 125, 200, 50
        R_min_2, R_max_2, M_min_2 = 5, 15, 46
    elif speed == 'max':
        R_min_1, R_max_1, M_min_1 = 150, 200, 50
        R_min_2, R_max_2, M_min_2 = 5, 15, 50
    else:
        raise ValueError(f"Invalid execution speed. Please use one of the following: slow, medium, fast, max.")

    return (R_min_1, R_max_1, M_min_1), (R_min_2, R_max_2, M_min_2)


if __name__ == '__main__':
    run(get_parameters_for_speed(EXEC_SPEED))
