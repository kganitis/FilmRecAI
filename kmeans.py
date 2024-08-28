import numpy as np
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_array, check_random_state
from tqdm import tqdm

from logger import Logger
from metrics import euclidean_distance_generalized


class KMeans:
    """
    K-means clustering implementation based on scikit-learn's KMeans,
    with several simplifications and customizations.

    Simplifications
    ---------------
    - Supports only the fit method, not the predict and transform methods.

    - Uses Lloyd's algorithm for clustering (no Elkan algorithm).

    - Only supports k-means++ and random center initialization.

    - Does not perform special handling of sparse matrices.

    - Cython code replaced with equivalent Python code.

    - Skips most data checks and validations.

    - Does not perform parallel processing.

    - Does not support sample weights.

    Customizations
    --------------
    - Allows the use of a callable distance metric defined by the user.

    - Allows for a weighted method of averaging the cluster centers.

    Parameters
    ----------
    k : int, default=8
        The number of clusters to form as well as the number of centroids to generate.

    init : {'k-means++', 'random'}, default='random'
        Method for initialization:
            * 'k-means++':
            selects initial cluster centroids using sampling based on an empirical probability
            distribution of the points' contribution to the overall inertia.
            The algorithm implemented is "greedy k-means++". It differs from the vanilla k-means++
            by making several trials at each sampling step and choosing the best centroid among them.

            * 'random':
            choose `n_clusters` observations (rows) at random from data for the initial centroids.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.

    max_iter : int, default=100
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance in regard to inertia to declare convergence.

    metric : callable, default=euclidean_distance_generalized
        What distance metric to use. The function should take two arrays as input and return a float.

    averaging : {'mean', 'weighted'}, default='mean'
        Method for averaging the cluster centers after each iteration:
            * 'mean':
            computes the mean of the feature vectors in each cluster.

            * 'weighted':
            Computes the weighted mean for each feature within each cluster,
            where weights are determined by the presence of non-zero values.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.

    log_level : int, default=2 (INFO)
        The logging level: DEBUG=0, VERBOSE=1, INFO=2, WARNING=3, ERROR=4, CRITICAL=5.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        The fitted data.

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : ndarray of shape (n_samples,)
        Labels of each point.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter : int
        Number of iterations run.
    """

    def __init__(self, k=5, init='random', n_init=10, max_iter=100, tol=1e-4, random_state=None,
                 metric=euclidean_distance_generalized, averaging='mean', log_level=Logger.ERROR):
        self.n_clusters = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.averaging = averaging
        self.random_state = check_random_state(random_state)
        self.logger = Logger(log_level)

    def _kmeans_plusplus(self, X):
        """K-means++ initialization of centroids."""
        centers = np.empty((self.n_clusters, self.n_features), dtype=X.dtype)
        n_local_trials = 2 + int(np.log(self.n_clusters))

        # Pick the first center randomly
        centers[0] = X[self.random_state.choice(self.n_samples)]

        # Compute the initial closest distances to the first center
        closest_dist_sq = np.array([self.metric(X[i], centers[0]) for i in range(self.n_samples)])
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.n_clusters):
            # Select new center candidates proportional to the squared distances
            rand_vals = self.random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)

            # Calculate distances from all points to candidate centers
            distance_to_candidates = np.array([
                [self.metric(X[i], X[candidate]) for i in range(self.n_samples)]
                for candidate in candidate_ids
            ])

            # Find the candidate that gives the lowest potential
            min_distances = np.minimum(closest_dist_sq, distance_to_candidates).min(axis=0)
            best_candidate = np.argmin(min_distances.sum(axis=0))
            current_pot = min_distances.sum()
            closest_dist_sq = min_distances

            # Add the best candidate as the new center
            centers[c] = X[candidate_ids[best_candidate]]

        return centers

    def _init_centroids(self, X):
        """
        Compute the initial centroids.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        if self.init == 'k-means++':
            centers = self._kmeans_plusplus(X)
        else:
            # default: 'random'
            seeds = self.random_state.choice(self.n_samples, size=self.n_clusters, replace=False)
            centers = X[seeds]
        return centers

    def fit(self, X):
        """
        Computes k-means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : KMeans
            Fitted instance of self.
        """
        X = X.copy()

        X = check_array(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self.tol = self._tolerance(X)
        self.n_samples, self.n_features = X.shape  # type: ignore

        best_centers, best_inertia, best_labels, best_n_iter = None, None, None, 1

        if self.init == 'k-means++':
            self.n_init = 1

        for i in range(self.n_init):
            with tqdm(total=self.max_iter, desc=f"Initialization {i + 1}/{self.n_init}", leave=False) as pbar:
                # Initialize random centers
                centers_init = self._init_centroids(X)
                self.__log("Initial centroids:", Logger.DEBUG, nl=True)
                self.__log_centroids(centers_init)

                # Run k-means once
                labels, inertia, centers, n_iter = self._kmeans_single(X, centers_init, pbar)
                self.__log(f"Cluster sizes: {self.__cluster_sizes(labels, sort='desc')}", nl=True)
                self.__log(f"Inertia: {inertia}\n")

                # Determine if these results are the best so far.
                if best_inertia is None or (
                        inertia < best_inertia and not self._is_same_clustering(labels, best_labels)
                ):
                    best_labels, best_inertia, best_centers, best_n_iter = labels, inertia, centers, n_iter

        self.X = X
        self.cluster_centers = best_centers
        self.labels = best_labels
        self.inertia = best_inertia
        self.n_iter = best_n_iter
        return self

    def _kmeans_single(self, X, centers_init, pbar=None):
        """
        Executes a single run of k-means algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        centers_init : ndarray of shape (n_clusters, n_features)
            Initial centers.

        pbar : tqdm, optional
            Progress bar to update during the iterations.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Labels of each point.

        inertia : float
            Sum of distances of samples to their closest cluster center.

        centers : ndarray of shape (n_clusters, n_features)
            Updated centers after convergence.

        n_iter : int
            Number of iterations run.

        Notes
        -----
        If the algorithm stops before fully converging (because of ``tol`` or ``max_iter``),
        ``labels`` and ``cluster_centers`` will not be consistent,
        i.e. the ``cluster_centers`` will not be the means of the points in each cluster.
        So scikit-learn re-runs the E-step, reassigning ``labels`` after the last iteration,
        to make ``labels`` consistent with ``predict`` on the training set.
        Since we did not implement ``predict``, we decided to skip this step.
        """
        centers = centers_init
        centers_new = np.zeros_like(centers)
        labels = np.full(self.n_samples, -1, dtype=np.int32)
        labels_old = labels.copy()
        inertia, n_iter = np.inf, 0

        for i in range(self.max_iter):
            n_iter = i + 1
            self.__log(f"Iteration {n_iter}", nl=True)

            center_shift = self._lloyd_iter(X, centers, centers_new, labels)
            inertia = self._inertia(X, centers_new, labels)
            centers, centers_new = centers_new, centers
            pbar.update(1)

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                self.__log(f"Converged at iteration {n_iter}: strict convergence.", nl=True)
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift ** 2).sum()
                if center_shift_tot <= self.tol:
                    self.__log(
                        f"Converged at iteration {n_iter}: "
                        f"center shift {center_shift_tot} within tolerance {self.tol}.",
                        nl=True)
                    break

        labels_old[:] = labels

        return labels, inertia, centers, n_iter

    def _lloyd_iter(self, X, centers_old, centers_new, labels, update_centers=True):
        """
        Performs a single iteration of Lloyd's algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        centers_old : ndarray of shape (n_clusters, n_features)
            The old cluster centers.

        centers_new : ndarray of shape (n_clusters, n_features)
            Array to store the new cluster centers.

        labels : ndarray of shape (n_samples,)
            Labels of each point.

        update_centers : bool, default=True
            Whether to update the cluster centers.

        Returns
        -------
        center_shift : ndarray of shape (n_clusters,)
            Shift of cluster centers.
        """
        # Calculate and log pairwise distances
        pairwise_distances = self.__calc_pairwise_distances(X, centers_old)
        pairwise_distances_str = np.array2string(pairwise_distances, formatter={
            'float_kind': lambda x: f"{x:.5f}" if x % 1 else f"{int(x)}"})
        self.__log(f"Pairwise distances:\n{pairwise_distances_str}", Logger.DEBUG)

        # Update labels
        labels[:] = np.argmin(pairwise_distances, axis=1)

        # Update centers
        if update_centers:
            self._relocate_empty_clusters(X, centers_old, centers_new, labels)
            self._average_centers(X, labels, centers_new)
            center_shift = np.array(
                [self.metric(centers_new[j], centers_old[j]) for j in range(centers_old.shape[0])]
            )

            # Log center update results
            self.__log("New centroids:", Logger.DEBUG, nl=True)
            self.__log_centroids(centers_new)
            self.__log(f"Total center shift: {np.sum(center_shift ** 2)}", Logger.VERBOSE)
            self.__log(f"Cluster sizes: {self.__cluster_sizes(labels, sort='desc')}", Logger.VERBOSE)
            self.__log(f"Inertia: {self._inertia(X, centers_new, labels)}")

            return center_shift

    def __calc_pairwise_distances(self, X, centers):
        """
        Calculates pairwise distances between each sample and each cluster center.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        centers : ndarray of shape (n_clusters, n_features)
            The cluster centers to calculate the distances from.

        Returns
        -------
        pairwise_distances : ndarray of shape (n_samples, n_clusters)
            Pairwise distances between samples and cluster centers.
        """
        pairwise_distances = np.zeros((self.n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            for i in range(self.n_samples):
                pairwise_distances[i, j] = self.metric(X[i], centers[j])

        return pairwise_distances

    def _relocate_empty_clusters(self, X, centers_old, centers_new, labels):
        """
        Relocates cluster centers which have no samples assigned to them.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        centers_old : ndarray of shape (n_clusters, n_features)
            The previous iteration's cluster centers.

        centers_new : ndarray of shape (n_clusters, n_features)
            The current iteration's updated cluster centers.

        labels : ndarray of shape (n_samples,)
            Cluster assignments of each data point.

        Returns
        -------
        n_relocated : int
            Number of clusters that were relocated.
        """
        # Identify empty clusters
        empty_clusters = np.where(np.array(self.__cluster_sizes(labels)) == 0)[0]
        n_empty = empty_clusters.shape[0]

        if n_empty == 0:
            return 0

        # Compute the distances of each point to its assigned center
        distances = np.array([self.metric(X[i], centers_old[labels[i]]) for i in range(self.n_samples)])

        # Identify the farthest points (those with the highest distances)
        farthest_points = np.argpartition(distances, -n_empty)[-n_empty:]

        for i, cluster_id in enumerate(empty_clusters):
            # Reassign the farthest point to the empty cluster
            farthest_point = farthest_points[i]
            labels[farthest_point] = cluster_id

            # Update the new center to the position of this farthest point
            centers_new[cluster_id] = X[farthest_point]

    def _average_centers(self, X, labels, centers):
        """
        Averages new centers using the method specified.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        labels : ndarray of shape (n_samples,)
            Cluster assignments of each data point.

        centers : ndarray of shape (n_clusters, n_features)
            The new cluster centers.
        """
        for j in range(self.n_clusters):
            # Select samples that belong to the current cluster
            cluster_points = X[labels == j]
            n_points = cluster_points.shape[0]

            if n_points > 0:
                if self.averaging == 'weighted':
                    # Weights are determined by the presence of non-zero values
                    lambda_matrix = (cluster_points != 0).astype(float)
                    weighted_sum = np.sum(cluster_points * lambda_matrix, axis=0)
                    lambda_sum = np.sum(lambda_matrix, axis=0)
                    centers[j] = np.where(lambda_sum != 0, np.divide(weighted_sum, lambda_sum), 0)
                else:  # default: 'mean'
                    centers[j] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, relocate it to the location of the largest cluster
                largest_cluster_idx = np.argmax([np.sum(labels == i) for i in range(self.n_clusters)])
                centers[j] = centers[largest_cluster_idx]

    def _is_same_clustering(self, labels1, labels2):
        """
        Checks if two arrays of labels represent the same clustering up to a permutation of the labels.

        Parameters
        ----------
        labels1 : ndarray of shape (n_samples,)
            The first set of labels.

        labels2 : ndarray of shape (n_samples,)
            The second set of labels.

        Returns
        -------
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

    def _inertia(self, X, centers, labels):
        """Sum of squared distances between each sample and its assigned center."""
        return sum(self.metric(X[i], centers[labels[i]]) for i in range(centers.shape[0]))

    def _tolerance(self, X):
        """Return a tolerance which is dependent on the dataset."""
        if self.tol == 0:
            return 0
        variances = np.var(X, axis=0)
        return np.mean(variances) * self.tol

    def cluster_sizes(self, sort=None):
        """
        Returns the sizes of each cluster, optionally sorted.

        Parameters
        ----------
        sort : str or None, default=None
            Specifies the sort order for the cluster sizes.
            * 'asc' or 'ascending': sort the sizes in ascending order.
            * 'desc' or 'descending': sort the sizes in descending order.
            *  None: do not sort the sizes.

        Returns
        -------
        sizes : list of int
            List of cluster sizes, optionally sorted.
        """
        return self.__cluster_sizes(self.labels, sort)

    @staticmethod
    def __cluster_sizes(labels, sort=None):
        """Returns the sizes of each cluster."""
        counts = np.bincount(labels)
        if sort:
            sort = str(sort).lower()
            if sort in ("asc", "ascending"):
                return sorted(counts)
            elif sort in ("desc", "descending"):
                return sorted(counts, reverse=True)
        return counts.tolist()

    def __log(self, message, level=2, filename=None, filemode='a', nl=False):
        self.logger.log(message, level, filename, filemode, nl)

    def __log_centroids(self, centers, log_zeros=False, log_level=Logger.DEBUG):
        if self.logger.log_level > log_level:
            return
        if not log_zeros:
            self.__log("*** Printing only non-zero values ***", log_level)
        for i, center in enumerate(centers):
            self.__log(f"Centroid {i + 1}:", log_level)
            feature_strings = []
            n_non_zero = 0
            for j, feature_value in enumerate(center):
                if log_zeros:
                    feature_strings.append(f"{feature_value:.4f}")
                elif feature_value != 0:
                    feature_strings.append(f"({i + 1}, {j + 1}): {feature_value:.4f}")
                    n_non_zero += 1

            if len(feature_strings) <= 10:
                self.__log(", ".join(feature_strings), log_level)
            else:
                first_five = feature_strings[:5]
                last_five = feature_strings[-5:]
                self.__log(", ".join(first_five + ["..."] + last_five), log_level)

            if not log_zeros:
                self.__log(f"Non-zero count: {n_non_zero}", log_level)
