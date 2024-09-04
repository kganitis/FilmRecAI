import numpy as np
import pandas as pd

from plotting import plot_ratings_per_user_histogram, plot_time_ranges_histogram


def run_for_kmeans_clustering_pipeline(R_min, R_max, M_min):
    """Run the preprocessing pipeline specifically configured for K-means clustering."""
    return _run(R_min, R_max, M_min, plot_histograms=True, refiltering=False)


def run_for_recommendations_pipeline(R_min, R_max, M_min):
    """Run the preprocessing pipeline specifically configured for training the recommendations ANN model."""
    return _run(R_min, R_max, M_min, plot_histograms=False, refiltering=True)


def _run(R_min, R_max, M_min, plot_histograms=True, refiltering=False) -> pd.DataFrame:
    """
    Preprocess the dataset to generate user preference vectors, with optional filtering and graph display.

    Args:
        R_min (int): Minimum required number of ratings per user.
        R_max (int): Maximum allowed number of ratings per user.
        M_min (int): Minimum required number of ratings per movie.
        plot_histograms (bool, optional): Whether to display histograms of the filtered data. Defaults to True.
        refiltering (bool, optional): Whether to apply iterative refiltering. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing user preference vectors.
    """
    # Load the dataset from a .npy file
    dataset = np.load('dataset.npy')

    # Split the dataset into components: username, movie, rating, and date
    data_split = [row.split(',') for row in dataset]

    # Convert the split data into a pandas DataFrame
    df = pd.DataFrame(data_split, columns=['username', 'movie', 'rating', 'date'])

    # Convert the DataFrame columns to appropriate data types
    df['username'] = df['username'].astype(str)
    df['movie'] = df['movie'].astype(str)
    df['rating'] = df['rating'].astype(int)
    df['date'] = pd.to_datetime(df['date'])

    print("\n---------------- Preprocessing: Find Unique Users and Movies Sets ----------------\n")

    # Identify unique users and movies in the dataset
    unique_users = df['username'].unique()
    unique_movies = df['movie'].unique()
    print("Number of Unique Users (U):", len(unique_users))
    print("Number of Unique Movies (I):", len(unique_movies))

    print("\n------------- Preprocessing: Filter Data based on Number of Ratings -------------\n")

    print(f"R_min: {R_min}, R_max: {R_max}, M_min: {M_min}")

    # Apply filtering based on the specified criteria
    if refiltering:
        filtered_df = _filter_df_with_refiltering(df, R_min, R_max, M_min)
    else:
        filtered_df = _filter_df(df, R_min, R_max, M_min)

    # Identify the filtered sets of users and movies
    filtered_unique_users = filtered_df['username'].unique()
    filtered_unique_movies = filtered_df['movie'].unique()
    n = len(filtered_unique_users)
    m = len(filtered_unique_movies)
    print("Number of Filtered Users (Û):", len(filtered_unique_users))
    print("Number of Filtered Movies (Î):", len(filtered_unique_movies))

    # Calculate the total number of non-zero ratings in the filtered dataset
    non_zero_ratings_count = (filtered_df['rating'] > 0).sum()
    print(f"Total number of ratings in the filtered dataset: {non_zero_ratings_count}")

    # Calculate the sparsity of the ratings matrix
    total_possible_entries = n * m
    sparsity = 1 - (non_zero_ratings_count / total_possible_entries)
    print(f"Sparsity of the ratings matrix: {sparsity:.4f}")

    # Display histograms
    if plot_histograms:
        plot_ratings_per_user_histogram(filtered_df, R_min, R_max)
        plot_time_ranges_histogram(filtered_df, days_interval=90)

    print("\n------- Preprocessing: Generate Preference (Feature) Vectors for each User -------\n")

    # Generate user preference vectors using a pivot table
    ratings_df = filtered_df.pivot_table(index='username', columns='movie', values='rating', aggfunc='last')

    # Fill missing ratings with 0
    ratings_df.fillna(0, inplace=True)

    print("Preprocessing completed successfully.")
    print(f"Shape of the user preference matrix: {ratings_df.shape}")
    print(ratings_df.head())

    return ratings_df


def _filter_df(df, R_min, R_max, M_min):
    """
    Filter the DataFrame to include only users and movies that meet the specified rating criteria.
    """
    # Filter users based on the number of ratings
    ratings_per_user = df.groupby('username')['rating'].count()
    filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)].index
    filtered_df = df[df['username'].isin(filtered_users)]

    # Filter movies based on the number of ratings
    ratings_per_movie = filtered_df.groupby('movie')['rating'].count()
    movies_filtered = ratings_per_movie[ratings_per_movie >= M_min].index
    filtered_df = filtered_df[filtered_df['movie'].isin(movies_filtered)]

    return filtered_df


def _filter_df_with_refiltering(df, R_min, R_max, M_min):
    """
    Filter the DataFrame iteratively, applying user and movie filtering repeatedly until the dataset stabilizes.
    """
    # Initial filter for movies
    ratings_per_movie = df.groupby('movie')['rating'].count()
    movies_filtered = ratings_per_movie[ratings_per_movie >= M_min].index
    filtered_df = df[df['movie'].isin(movies_filtered)]

    # Initial filter for users
    ratings_per_user = filtered_df.groupby('username')['rating'].count()
    filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)].index
    filtered_df = filtered_df[filtered_df['username'].isin(filtered_users)]

    # Iteratively apply filtering until the dataset stabilizes
    while True:
        new_ratings_per_movie = filtered_df.groupby('movie')['rating'].count()
        new_movies_filtered = new_ratings_per_movie[new_ratings_per_movie >= M_min].index
        new_filtered_df = filtered_df[filtered_df['movie'].isin(new_movies_filtered)]

        new_ratings_per_user = new_filtered_df.groupby('username')['rating'].count()
        new_filtered_users = new_ratings_per_user[
            (new_ratings_per_user >= R_min) & (new_ratings_per_user <= R_max)].index
        new_filtered_df = new_filtered_df[new_filtered_df['username'].isin(new_filtered_users)]

        # Check if the filtering has stabilized
        if new_filtered_df.shape == filtered_df.shape:
            break  # Exit loop if no changes occurred
        filtered_df = new_filtered_df  # Update the DataFrame for the next iteration

    return filtered_df
