import numpy as np
from kmeans import KMeans
import preprocessing
from metrics import euclidean_distance_custom


def generate_user_movie_ratings(n_users=500, n_movies=2000, n_ratings_per_user=10, rating_scale=(1, 10)):
    # Initialize the dataset with zeros
    user_movie_ratings = np.zeros((n_users, n_movies))

    # Generate movie popularity using a Zipf distribution
    movie_popularity = np.random.zipf(a=2.0, size=n_movies)
    movie_popularity = movie_popularity / np.sum(movie_popularity)  # Normalize to get probabilities

    for user in range(n_users):
        # Determine the number of movies this user will rate
        n_ratings = np.random.randint(5, n_ratings_per_user + 1)
        # Select movies for this user to rate based on their popularity
        rated_movies = np.random.choice(n_movies, n_ratings, replace=False, p=movie_popularity)
        # Assign random ratings to the selected movies
        user_movie_ratings[user, rated_movies] = np.random.randint(rating_scale[0], rating_scale[1] + 1, n_ratings)

    return user_movie_ratings


# Preprocessing variables
R_min = 90
R_max = 100
display_graphs = False

# Clustering variables
L = 6
n_init = 10
rnd = np.random.randint(0, 50)
seed = 0
verbose = False
plot_iters = False
plot_results = True

data = preprocessing.run(R_min, R_max, display_graphs)
# data = generate_user_movie_ratings()

# # KMeans clustering with standard Euclidean distance and mean centering
# kmeans_standard = KMeans(
#     n_clusters=L,
#     n_init=n_init,
#     random_state=seed,
#     verbose=verbose,
#     plot_iters=plot_iters,
#     plot_results=plot_results)
# kmeans_standard.fit(data)
#
# print("\nResults with standard Euclidean distance and mean centering:")
# print(kmeans_standard.labels_value_counts)
#
# # KMeans clustering with custom Euclidean distance and mean centering
# kmeans_custom = KMeans(
#     n_clusters=L,
#     n_init=n_init,
#     distance_func=euclidean_distance_custom,
#     random_state=seed,
#     verbose=verbose,
#     plot_iters=plot_iters,
#     plot_results=plot_results)
# kmeans_custom.fit(data)
#
# print("\nResults with custom Euclidean distance and mean centering:")
# print(kmeans_custom.labels_value_counts)
#
# # KMeans clustering with standard Euclidean distance, but no mean centering
# kmeans_standard_nomean = KMeans(
#     n_clusters=L,
#     n_init=n_init,
#     mean_centering=False,
#     random_state=seed,
#     verbose=verbose,
#     plot_iters=plot_iters,
#     plot_results=plot_results)
# kmeans_standard_nomean.fit(data)
#
# print("\nResults with standard Euclidean distance, but no mean centering:")
# print(kmeans_standard_nomean.labels_value_counts)

# KMeans clustering with custom Euclidean distance, but no mean centering
kmeans_custom_nomean = KMeans(
    n_clusters=L,
    n_init=n_init,
    max_iter=20,
    distance_func=euclidean_distance_custom,
    mean_centering=False,
    random_state=seed,
    verbose=True,
    plot_iters=True,
    plot_results=plot_results)
kmeans_custom_nomean.fit(data)

print("\nResults with custom Euclidean distance, but no mean centering:")
print(kmeans_custom_nomean.labels_value_counts)
