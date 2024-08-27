import preprocessing
import clustering
from metrics import euclidean_distance_generalized, cosine_distance_generalized

if __name__ == '__main__':
    # Preprocess the data
    data = preprocessing.run(R_min=50, R_max=200, M_min=50)

    # K-means clustering
    L_values = [5, 7, 10, 15, 20]
    metrics = [euclidean_distance_generalized, cosine_distance_generalized]
    for metric in metrics:
        for L in L_values:
            clustering.run(data, L, metric)
