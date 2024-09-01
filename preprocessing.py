import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(R_min: int, R_max: int, M_min: int, display_graphs: bool = True, days_interval: int = 90) -> pd.DataFrame:
    """
    Preprocess the dataset and generate user preference vectors.

    :param R_min: Minimum required number of ratings per user
    :param R_max: Maximum allowed number of ratings per user
    :param M_min: Minimum required number of ratings per movie
    :param display_graphs: Whether to display graphs
    :param days_interval: Interval of days to group data for the histogram of time ranges

    :return: User preference vectors as a pandas DataFrame
    """
    # Load the dataset from dataset.npy
    dataset = np.load('dataset.npy')

    # Split each row of data into username, movie, rating, and date
    data_split = [row.split(',') for row in dataset]

    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data_split, columns=['username', 'movie', 'rating', 'date'])

    # Convert types: username to string, movie to string, rating to int, date to datetime
    df['username'] = df['username'].astype(str)
    df['movie'] = df['movie'].astype(str)
    df['rating'] = df['rating'].astype(int)
    df['date'] = pd.to_datetime(df['date'])

    print("\n---------------- Preprocessing: Find Unique Users and Movies Sets ----------------\n")

    # Get the sets of unique users and movies
    unique_users = df['username'].unique()
    unique_movies = df['movie'].unique()
    N = len(unique_users)
    M = len(unique_movies)

    print("Number of Unique Users (U):", N)
    print("Number of Unique Movies (I):", M)

    print("\n------------- Preprocessing: Filter Data based on Number of Ratings -------------\n")

    # Initial Filter for movies
    ratings_per_movie = df.groupby('movie')['rating'].count()
    movies_filtered = ratings_per_movie[ratings_per_movie >= M_min].index
    filtered_df = df[df['movie'].isin(movies_filtered)]

    # Initial Filter for users
    ratings_per_user = filtered_df.groupby('username')['rating'].count()
    filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)].index
    filtered_df = filtered_df[filtered_df['username'].isin(filtered_users)]

    # Iterative re-filtering process
    while True:
        new_ratings_per_movie = filtered_df.groupby('movie')['rating'].count()
        new_movies_filtered = new_ratings_per_movie[new_ratings_per_movie >= M_min].index
        new_filtered_df = filtered_df[filtered_df['movie'].isin(new_movies_filtered)]

        new_ratings_per_user = new_filtered_df.groupby('username')['rating'].count()
        new_filtered_users = new_ratings_per_user[
            (new_ratings_per_user >= R_min) & (new_ratings_per_user <= R_max)].index
        new_filtered_df = new_filtered_df[new_filtered_df['username'].isin(new_filtered_users)]

        # Check if the new filters result in the same data
        if new_filtered_df.shape == filtered_df.shape:
            break  # Exit loop if no change in data
        filtered_df = new_filtered_df  # Update filtered data for the next iteration

    # Final statistics after filtering
    final_ratings_per_user = filtered_df.groupby('username')['rating'].count()
    final_ratings_per_movie = filtered_df.groupby('movie')['rating'].count()
    min_ratings_per_user = final_ratings_per_user.min()
    max_ratings_per_user = final_ratings_per_user.max()
    min_ratings_per_movie = final_ratings_per_movie.min()

    # Display the statistics
    print(f"R_min: {R_min}, R_max: {R_max}, M_min: {M_min}")
    print("Statistics for Filtered Dataset:")
    print(f"Minimum number of ratings per user: {min_ratings_per_user} (Expected >= {R_min})")
    print(f"Maximum number of ratings per user: {max_ratings_per_user} (Expected <= {R_max})")
    print(f"Minimum number of ratings per movie: {min_ratings_per_movie} (Expected >= {M_min})")

    # Check if the filtering conditions are met
    conditions_met = (
            min_ratings_per_user >= R_min and
            max_ratings_per_user <= R_max and
            min_ratings_per_movie >= M_min
    )
    print("Filtering Successful:", "Yes" if conditions_met else "No")

    # Get the sets of filtered users and movies
    filtered_unique_users = filtered_df['username'].unique()
    filtered_unique_movies = filtered_df['movie'].unique()
    n = len(filtered_unique_users)
    m = len(filtered_unique_movies)
    print("Number of Filtered Users (Û):", n)
    print("Number of Filtered Movies (Î):", m)

    # print("\n------------------ Preprocessing: Generate Frequency Histograms ------------------\n")

    if display_graphs:
        # Recalculate the number of ratings per user based on the filtered DataFrame
        ratings_per_user_filtered = filtered_df.groupby('username')['rating'].count()

        # First Histogram: Number of Ratings per User (Filtered Data)
        plt.figure(figsize=(10, 5))
        plt.hist(ratings_per_user_filtered, bins=range(R_min, R_max + 2), edgecolor='black', alpha=0.7)
        plt.title('Number of Ratings per User (Filtered Data)')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        plt.grid(True)
        plt.show()

        # Calculate time ranges (in days) for each user based on the filtered DataFrame
        first_rating_date = filtered_df.groupby('username')['date'].min()
        last_rating_date = filtered_df.groupby('username')['date'].max()
        time_ranges = (last_rating_date - first_rating_date).dt.days

        # Second Histogram: Time Ranges for All Ratings by Users (Filtered Data)
        plt.figure(figsize=(10, 5))
        plt.hist(time_ranges, bins=range(0, max(time_ranges) + 8, days_interval), edgecolor='black', alpha=0.7)
        plt.title('Time Ranges for All Ratings by Users (Filtered Data)')
        plt.xlabel('Time Range (days)')
        plt.ylabel('Number of Users')
        plt.grid(True)
        plt.show()

    print("\n------- Preprocessing: Generate Preference (Feature) Vectors for each User -------\n")

    # Use pivot_table to aggregate duplicate ratings by taking the maximum rating for each user-movie combination
    ratings_df = filtered_df.pivot_table(index='username', columns='movie', values='rating', aggfunc='last')

    # Fill missing values (movies not rated by users) with 0
    ratings_df.fillna(0, inplace=True)

    print(ratings_df)

    return ratings_df
