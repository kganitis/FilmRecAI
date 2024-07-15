import numpy as np
from sklearn.utils.validation import check_array, check_random_state
from metrics import euclidean_distance
from plotting import plot_clusters


class KMeans:
    """
    K-means clustering implementation for dense data using Lloyd's algorithm.

    This implementation is based on scikit-learn's KMeans but includes some custom modifications:
    - Only supports dense data format (no scipy sparse format).
    - Uses Lloyd's algorithm for clustering (no Elkan algorithm).
    - Does not include parallel processing, sample weights, or advanced center initialization (e.g., k-means++).
    - Allows for a custom distance function.

    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.

    max_iter : int, default=100
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance in regard to inertia to declare convergence.

    distance_func : callable, default=euclidean_distance
        Custom distance function. The function should take two arrays as input and return a float.

    mean_centering : bool, default=True
        If True, the data will be mean centered before clustering.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    verbose : bool, default=False
        If True, prints information about the clustering process.

    plot_iters : bool, default=False
        If True, plots the clusters after each centers update.

    plot_results : bool, default=False
        If True, plots the final results of the clustering.

    Attributes:
    -----------
    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : ndarray of shape (n_samples,)
        Labels of each point.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter : int
        Number of iterations run.
    """

    def __init__(self, n_clusters=8, n_init=10, max_iter=100, tol=1e-4,
                 distance_func=euclidean_distance, mean_centering=True,
                 random_state=None, verbose=False, plot_iters=False,
                 plot_results=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.distance_func = distance_func
        self.mean_centering = mean_centering
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.plot_iters = plot_iters
        self.plot_results = plot_results
        self.labels = None

    def log(self, message):
        """Prints a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    @property
    def labels_value_counts(self):
        """Returns the sorted count of each label."""
        return sorted(np.bincount(self.labels), reverse=True)

    def _init_centroids(self, X):
        """Initializes centroids by randomly selecting samples from the dataset."""
        seeds = self.random_state.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[seeds]

    def fit(self, X):
        """
        Computes k-means clustering.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        Returns:
        --------
        self : KMeans
            Fitted instance of self.
        """
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        if self.mean_centering:
            X = X.copy()  # ensure X is not modified in-place
            X_mean = X.mean(axis=0)
            X -= X_mean

        best_inertia, best_labels = None, None

        for _ in range(self.n_init):
            self.log(f"\nInitialization {_ + 1} out of {self.n_init}")
            centers_init = self._init_centroids(X)

            # Run k-means once
            labels, inertia, centers, n_iter = self._kmeans_single(X, centers_init)
            self.log(f"Labels distribution: {sorted(np.bincount(labels), reverse=True)}")

            # Determine if these results are the best so far.
            if best_inertia is None or (
                    inertia < best_inertia and not self._is_same_clustering(labels, best_labels)
            ):
                best_labels, best_inertia, best_centers, best_n_iter = labels, inertia, centers, n_iter

        if self.plot_results:
            plot_clusters(X, best_centers, best_labels, best_n_iter)

        self.cluster_centers = best_centers
        self.labels = best_labels
        self.inertia = best_inertia
        self.n_iter = best_n_iter
        return self

    def _kmeans_single(self, X, centers_init):
        """
        Executes a single run of k-means algorithm.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        centers_init : ndarray of shape (n_clusters, n_features)
            Initial centers.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Labels of each point.

        inertia : float
            Sum of squared distances of samples to their closest cluster center.

        centers : ndarray of shape (n_clusters, n_features)
            Updated centers after convergence.

        n_iter : int
            Number of iterations run.
        """
        n_clusters = centers_init.shape[0]
        centers = centers_init
        centers_new = np.zeros_like(centers)
        labels = np.full(X.shape[0], -1, dtype=np.int32)
        labels_old = labels.copy()
        weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)

        strict_convergence = False

        for i in range(self.max_iter):
            center_shift = self._lloyd_iter_single(X, centers, centers_new, labels, weight_in_clusters)
            centers, centers_new = centers_new, centers

            # First check the labels for strict convergence.
            if np.array_equal(labels, labels_old):
                self.log(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tolerance-based convergence.
                total_center_shift = np.sum(center_shift ** 2)
                if total_center_shift <= self.tol:
                    self.log(
                        f"Converged at iteration {i}: center shift "
                        f"{total_center_shift} within tolerance {self.tol}."
                    )
                    break

            labels_old[:] = labels
            self.log(f"Update {i + 1}: {sorted(np.bincount(labels), reverse=True)}")

            if self.plot_iters:
                plot_clusters(X, centers, labels, i + 1)

        if not strict_convergence:
            # Rerun E-step so that predicted labels match cluster centers
            self._lloyd_iter_single(X, centers, centers, labels, weight_in_clusters, update_centers=False)

        inertia = sum(self.distance_func(X[i], centers[labels[i]]) for i in range(X.shape[0]))

        return labels, inertia, centers, i + 1

    def _lloyd_iter_single(self, X, centers_old, centers_new, labels, weight_in_clusters, update_centers=True):
        """
        Performs a single iteration of Lloyd's algorithm.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        centers_old : ndarray of shape (n_clusters, n_features)
            The old cluster centers.

        centers_new : ndarray of shape (n_clusters, n_features)
            Array to store the new cluster centers.

        labels : ndarray of shape (n_samples,)
            Labels of each point.

        weight_in_clusters : ndarray of shape (n_clusters,)
            Weight of each cluster.

        update_centers : bool, default=True
            Whether to update the cluster centers.

        Returns:
        --------
        center_shift : ndarray of shape (n_clusters,)
            Shift of cluster centers.
        """
        n_samples = X.shape[0]

        if n_samples == 0:
            return

        if update_centers:
            centers_new.fill(0)
            weight_in_clusters.fill(0)

        # Calculate pairwise distances
        pairwise_distances = self._calc_pairwise_distances(X, centers_old)

        # Update labels and centers
        for i in range(n_samples):
            label = np.argmin(pairwise_distances[i])
            labels[i] = label

            if update_centers:
                weight_in_clusters[label] += 1
                centers_new[label] += X[i]

        if update_centers:
            self._relocate_empty_clusters(X, centers_old, centers_new, labels, weight_in_clusters)
            _average_centers(centers_new, weight_in_clusters)
            center_shift = np.array(
                [self.distance_func(centers_new[j], centers_old[j]) for j in range(centers_old.shape[0])]
            )
            return center_shift

    def _calc_pairwise_distances(self, X, centers_old):
        """
        Calculates pairwise distances between each sample and each cluster center.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        centers_old : ndarray of shape (n_clusters, n_features)
            The old cluster centers.

        Returns:
        --------
        pairwise_distances : ndarray of shape (n_samples, n_clusters)
            Pairwise distances between samples and cluster centers.
        """
        n_samples = X.shape[0]
        pairwise_distances = np.zeros((n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            for i in range(n_samples):
                pairwise_distances[i, j] = self.distance_func(X[i], centers_old[j])
        return pairwise_distances

    def _relocate_empty_clusters(self, X, centers_old, centers_new, labels, weight_in_clusters):
        """
        Relocates centers which have no sample assigned to them.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        centers_old : ndarray of shape (n_clusters, n_features)
            The old cluster centers.

        centers_new : ndarray of shape (n_clusters, n_features)
            The new cluster centers.

        labels : ndarray of shape (n_samples,)
            Labels of each point.

        weight_in_clusters : ndarray of shape (n_clusters,)
            Weight of each cluster.
        """
        empty_clusters = np.where(weight_in_clusters == 0)[0]
        n_empty = empty_clusters.shape[0]

        if n_empty == 0:
            return

        # Find points that are farthest from their assigned centers
        distances = np.array(
            [self.distance_func(X[i], centers_old[labels[i]]) for i in range(X.shape[0])]
        )
        if np.max(distances) == 0:
            return

        n_features = X.shape[1]
        far_from_centers = np.argpartition(distances, -n_empty)[-n_empty:]

        for idx in range(n_empty):
            new_cluster_id = empty_clusters[idx]
            far_idx = far_from_centers[idx]
            old_cluster_id = labels[far_idx]

            for k in range(n_features):
                centers_new[old_cluster_id, k] -= X[far_idx, k]
                centers_new[new_cluster_id, k] = X[far_idx, k]

            weight_in_clusters[new_cluster_id] = 1
            weight_in_clusters[old_cluster_id] -= 1

    def _is_same_clustering(self, labels1, labels2):
        """
        Checks if two arrays of labels represent the same clustering up to a permutation of the labels.

        Parameters:
        -----------
        labels1 : ndarray of shape (n_samples,)
            The first set of labels.

        labels2 : ndarray of shape (n_samples,)
            The second set of labels.

        Returns:
        --------
        bool
            True if the two label sets represent the same clustering, False otherwise.
        """
        mapping = np.full(fill_value=-1, shape=(self.n_clusters,), dtype=np.int32)
        for i in range(labels1.shape[0]):
            if mapping[labels1[i]] == -1:
                mapping[labels1[i]] = labels2[i]
            elif mapping[labels1[i]] != labels2[i]:
                return False
        return True


def _average_centers(centers, weight_in_clusters):
    """
    Averages new centers with respect to weights.

    Parameters:
    -----------
    centers : ndarray of shape (n_clusters, n_features)
        The new cluster centers.

    weight_in_clusters : ndarray of shape (n_clusters,)
        Weight of each cluster.
    """
    for j in range(centers.shape[0]):
        if weight_in_clusters[j] > 0:
            centers[j] /= weight_in_clusters[j]
        else:
            # If a cluster is empty, it takes the position of the largest cluster center.
            centers[j] = centers[np.argmax(weight_in_clusters)]
