import numpy as np
import pandas as pd

# ------------ Task 1 ---------------

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

# ------------ Task 2 ---------------

# Calculate the number of ratings per user
ratings_per_user = df.groupby('username')['rating'].count()

# Define the bins for different ranges of reviews per user
bins = [0, 4, 50, 100, 500, 1000, float('inf')]

# Create labels for the bins
labels = ['1-4', '5-50', '51-100', '101-500', '501-1000', '1000+']

# Cut the data into bins
ratings_bins = pd.cut(ratings_per_user, bins=bins, labels=labels, right=False)

# Count the frequency of users in each bin
bin_counts = ratings_bins.value_counts().sort_index()

# Display the results
for bin_label, count in bin_counts.items():
    print(f"{bin_label} reviews: {count} users")


# # Choose initial values for R_min and R_max
# R_min = 10
# R_max = 100
#
# # Filter users based on the number of ratings
# filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)]
#
# # Filter the DataFrame based on the selected users
# filtered_df = df[df['username'].isin(filtered_users.index)]
#
# # Update the sets of unique users and objects
# users_filtered = filtered_df['username'].unique()
# movies_filtered = filtered_df['movie'].unique()
#
# print("Number of unique users after filtering:", len(users_filtered))
# print("Number of unique objects after filtering:", len(movies_filtered))
