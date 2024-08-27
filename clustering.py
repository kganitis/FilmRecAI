import preprocessing
from kmeans import KMeans
from metrics import *
from plotting import *


def run():
    # Preprocess the data
    data = preprocessing.run(R_min=50, R_max=200, M_min=50, display_graphs=True)
    # np.random.seed(42)
    # data = np.random.rand(1000, 3)

    # Initialize k-means and fit the data
    rnd = np.random.randint(0, 50)
    seed = 1
    kmeans = KMeans(
        metric=euclidean_distance_generalized,
        n_clusters=5,
        init='random',
        # max_iter=30,
        random_state=seed,
        averaging='mean',
        log_level=1,
        # # plotting
        # plot_iters=False,
        # plot_results=True,
        # plot_normalized=False,
        # dim_reduct='pca',
        # normalize_x=False,
    )
    kmeans.fit(data)

    # Print clustering results
    print(f"\nK-MEANS CLUSTERING")
    print(f"Metric used: {kmeans.metric.__name__}")
    print(f"Number of clusters: {kmeans.n_clusters}")
    print(f"Initialization method used: {kmeans.init}")
    print(f"Number of initializations: {kmeans.n_init}, seed: {seed}")
    print(f"Max number of iterations per run: {kmeans.max_iter}")
    print(f"Tolerance: {kmeans.tol}")

    print(f"\nDATA:")
    print(f"Number of samples: {kmeans.n_samples}")
    print(f"Number of features: {kmeans.n_features}")

    print(f"\nRESULTS")
    print(f"Cluster sizes: {kmeans.cluster_sizes('desc')}")
    print(f"Inertia: {kmeans.inertia}")
    print(f"Iterations run: {kmeans.n_iter}")
    print("Centroids:")
    for centroid in kmeans.cluster_centers:
        n_values_printed = 10
        if len(centroid) <= n_values_printed:
            formatted_centroid = "[" + " ".join("{:.2f}".format(x) for x in centroid) + "]"
        else:
            first_values = " ".join("{:.2f}".format(x) for x in centroid[:n_values_printed // 2])
            last_values = " ".join("{:.2f}".format(x) for x in centroid[-n_values_printed // 2:])
            formatted_centroid = "[" + first_values + " ... " + last_values + "]"
        print(formatted_centroid)

    # Plot clusters
    title = f"Clusters for L={kmeans.n_clusters} using {kmeans.metric.__name__}"
    plot_clusters(data, kmeans.cluster_centers, kmeans.labels, title, dim_reduct='pca')
