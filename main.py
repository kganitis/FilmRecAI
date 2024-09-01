import argparse

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
    #         clustering.kmeans(data, metric, L)

    recommendations.run(ratings_df)


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
