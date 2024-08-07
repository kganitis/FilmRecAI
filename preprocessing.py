import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(R_min: int, R_max: int, display_graphs=False) -> pd.DataFrame:
    """
    :param R_min: Minimum required number of ratings per user
    :param R_max: Maximum allowed number of ratings per user
    :param display_graphs: Whether to display graphs
    :return: User preference vectors as a pandas dataframe
    """
    # ------------ Preprocessing Task 1: Find Unique Users and Movies Sets ------------

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

    # Find the unique users and movies
    unique_users = df['username'].unique()
    unique_movies = df['movie'].unique()
    N = len(unique_users)
    M = len(unique_movies)
    print("\n------------ Preprocessing Task 1: Find Unique Users and Movies Sets ------------\n")
    print("Number of Unique Users (U):", N)
    print("Number of Unique Movies (I):", M)

    # ------------ Preprocessing Task 2: Filter Users based on Number of Ratings ------------

    # Calculate the number of ratings per user
    ratings_per_user = df.groupby('username')['rating'].count()

    # Filter users based on the number of ratings
    filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)]

    # Filter the DataFrame based on the filtered users
    filtered_df = df[df['username'].isin(filtered_users.index)]

    # Get the sets of filtered users and movies
    filtered_unique_users = filtered_df['username'].unique()
    filtered_unique_movies = filtered_df['movie'].unique()
    n = len(filtered_unique_users)
    m = len(filtered_unique_movies)

    print("\n------------ Preprocessing Task 2: Filter Users based on Number of Ratings ------------\n")
    print("R_min:", R_min)
    print("R_max:", R_max)
    print("Number of Filtered Users (Û):", n)
    print("Number of Filtered Movies (Î):", m)

    # ------------ Preprocessing Task 3: Generate Frequency Histograms ------------

    if display_graphs:
        # Recalculate the number of ratings per user based on the filtered DataFrame
        ratings_per_user_filtered = filtered_df.groupby('username')['rating'].count()

        # First Histogram: Number of Ratings per User
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
        plt.hist(time_ranges, bins=range(0, max(time_ranges) + 8, 7), edgecolor='black', alpha=0.7)
        plt.title('Time Ranges for All Ratings by Users (Filtered Data)')
        plt.xlabel('Time Range (days)')
        plt.ylabel('Number of Users')
        plt.grid(True)
        plt.show()

    # ------------ Preprocessing Task 4: Generate Preference (Feature) Vectors for each User ------------

    # Use pivot_table to aggregate duplicate ratings by taking the maximum rating for each user-movie combination
    pivot_df = filtered_df.pivot_table(index='username', columns='movie', values='rating', aggfunc='last')

    # Fill missing values (movies not rated by users) with 0
    pivot_df.fillna(0, inplace=True)

    # Sort the data so that the users with the most ratings appear in the first rows
    # and the most rated movies in the first columns
    ratings_per_user_filtered = filtered_df.groupby('username')['rating'].count()
    sorted_users = ratings_per_user_filtered.sort_values(ascending=False).index
    pivot_df = pivot_df.loc[sorted_users]

    ratings_count_movies = filtered_df.groupby('movie')['rating'].count()
    sorted_movies = ratings_count_movies.sort_values(ascending=False).index
    pivot_df = pivot_df[sorted_movies]

    # Print the first few preference vectors
    print("\n------------ Preprocessing Task 4: Generate Preference (Feature) Vectors for each User ------------\n")
    print(pivot_df.head())

    return pivot_df
