import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_kmeans_clustering_pipeline(R_min, R_max, M_min):
    """
    Run the preprocessing pipeline specifically configured for K-means clustering.

    Args:
        R_min (int): Minimum required number of ratings per user.
        R_max (int): Maximum allowed number of ratings per user.
        M_min (int): Minimum required number of ratings per movie.

    Returns:
        pd.DataFrame: User preference vectors as a pandas DataFrame.
    """
    return _run(R_min, R_max, M_min, display_graphs=True, refiltering=False)


def run_jaccard_clustering_pipeline(R_min, R_max, M_min):
    """
    Run the preprocessing pipeline specifically configured for Jaccard clustering.

    Args:
        R_min (int): Minimum required number of ratings per user.
        R_max (int): Maximum allowed number of ratings per user.
        M_min (int): Minimum required number of ratings per movie.

    Returns:
        pd.DataFrame: User preference vectors as a pandas DataFrame.
    """
    return _run(R_min, R_max, M_min, display_graphs=False, refiltering=True)


def _run(R_min: int, R_max: int, M_min: int, display_graphs: bool = True, refiltering: bool = False, days_interval: int = 90) -> pd.DataFrame:
    """
    Preprocess the dataset to generate user preference vectors, with optional filtering and graph display.

    Args:
        R_min (int): Minimum required number of ratings per user.
        R_max (int): Maximum allowed number of ratings per user.
        M_min (int): Minimum required number of ratings per movie.
        display_graphs (bool, optional): Whether to display histograms of the filtered data. Defaults to True.
        refiltering (bool, optional): Whether to apply iterative refiltering. Defaults to False.
        days_interval (int, optional): Interval of days to group data for the histogram of time ranges. Defaults to 90.

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
    N = len(unique_users)
    M = len(unique_movies)

    print("Number of Unique Users (U):", N)
    print("Number of Unique Movies (I):", M)

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
    print("Number of Filtered Users (Û):", n)
    print("Number of Filtered Movies (Î):", m)

    # Calculate the total number of non-zero ratings in the filtered dataset
    non_zero_ratings_count = (filtered_df['rating'] > 0).sum()
    print(f"Total number of ratings in the filtered dataset: {non_zero_ratings_count}")

    # Calculate the sparsity of the ratings matrix
    total_possible_entries = len(filtered_unique_users) * len(filtered_unique_movies)
    sparsity = 1 - (non_zero_ratings_count / total_possible_entries)
    print(f"Sparsity of the ratings matrix: {sparsity:.4f}")

    if display_graphs:
        # Display histogram of number of ratings per user
        ratings_per_user_filtered = filtered_df.groupby('username')['rating'].count()
        plt.figure(figsize=(10, 5))
        plt.hist(ratings_per_user_filtered, bins=range(R_min, R_max + 2), edgecolor='black', alpha=0.7)
        plt.title('Number of Ratings per User (Filtered Data)')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        plt.grid(True)
        plt.show()

        # Display histogram of time ranges for all ratings by users
        first_rating_date = filtered_df.groupby('username')['date'].min()
        last_rating_date = filtered_df.groupby('username')['date'].max()
        time_ranges = (last_rating_date - first_rating_date).dt.days
        plt.figure(figsize=(10, 5))
        plt.hist(time_ranges, bins=range(0, max(time_ranges) + 8, days_interval), edgecolor='black', alpha=0.7)
        plt.title('Time Ranges for All Ratings by Users (Filtered Data)')
        plt.xlabel('Time Range (days)')
        plt.ylabel('Number of Users')
        plt.grid(True)
        plt.show()

    print("\n------- Preprocessing: Generate Preference (Feature) Vectors for each User -------\n")

    # Generate user preference vectors using a pivot table
    ratings_df = filtered_df.pivot_table(index='username', columns='movie', values='rating', aggfunc='last')

    # Fill missing ratings with 0
    ratings_df.fillna(0, inplace=True)

    print(ratings_df)

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
