import preprocessing
import clustering
from metrics import *

if __name__ == '__main__':
    # Preprocess the data
    data = preprocessing.run(R_min=50, R_max=200, M_min=50, display_graphs=True)

    # Cluster the data
    L_values = [5, 7, 10, 15, 20]

    for L in L_values:
        clustering.run(data, L, metric=euclidean_distance_generalized, seed=1)

    for L in L_values:
        clustering.run(data, L, metric=cosine_distance_generalized, seed=1)
