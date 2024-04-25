import numpy as np
import pandas as pd

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
users = df['username'].unique()
movies = df['movie'].unique()
print("Unique Users (U):", len(users))
print("Unique Movies (I):", len(movies))

# Display the first few rows of the DataFrame
print("\nDataFrame Head:")
print(df.head())
