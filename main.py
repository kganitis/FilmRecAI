import preprocessing
import clustering
from metrics import euclidean_distance_generalized, cosine_distance_generalized
import argparse


# Constants for execution speed
SLOW = 'slow'  # only for final submission
MEDIUM = 'medium'  # suitable for debugging
FAST = 'fast'  # default, for quick demonstrations

EXEC_SPEED = FAST  # set for the desired execution speed


def run(R_min, R_max, M_min):
    # Preprocess the data
    data = preprocessing.run(R_min, R_max, M_min)

    # K-means clustering
    L_values = [5, 7, 10, 15, 20]
    metrics = [euclidean_distance_generalized, cosine_distance_generalized]
    for metric in metrics:
        for L in L_values:
            clustering.run(data, L, metric)


def get_parameters_for_speed(speed=FAST):
    """Get parameters based on execution speed."""
    if speed == MEDIUM:
        R_min, R_max, M_min = 100, 200, 50
    elif speed == SLOW:
        R_min, R_max, M_min = 50, 200, 50
    else:  # FAST
        R_min, R_max, M_min = 150, 200, 50
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
    exec_speed = args.speed.lower()
    run(*get_parameters_for_speed(exec_speed))
