import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ Task 1: Find Unique Users and Movies Sets ------------

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
print("\n------------ Task 1: Find Unique Users and Movies Sets ------------\n")
print("Number of Unique Users (U):", len(unique_users))
print("Number of Unique Movies (I):", len(unique_movies))

# ------------ Task 2: Filter Users based on Number of Ratings ------------

# Calculate the number of ratings per user
ratings_per_user = df.groupby('username')['rating'].count()

# Define the bins for different ranges of reviews per user
bins = [1, 10, 101, float('inf')]

# Create labels for the bins
labels = ['1-9', '10-100', '101+']

# Cut the data into bins
ratings_bins = pd.cut(ratings_per_user, bins=bins, labels=labels, right=False)

# Count the frequency of users in each bin
bin_counts = ratings_bins.value_counts().sort_index()

# Display the results
print("\n------------ Task 2: Filter Users based on Number of Ratings ------------\n")
for bin_label, count in bin_counts.items():
    print(f"{bin_label} reviews: {count} users")

# Choose initial values for R_min and R_max
R_min = 10  # Minimum required number of ratings per user
R_max = 100  # Maximum allowed number of ratings per user

# Filter users based on the number of ratings
filtered_users = ratings_per_user[(ratings_per_user >= R_min) & (ratings_per_user <= R_max)]

# Filter the DataFrame based on the selected users
filtered_df = df[df['username'].isin(filtered_users.index)]

# Get the sets of filtered users and movies
filtered_unique_users = filtered_df['username'].unique()
filtered_unique_movies = filtered_df['movie'].unique()

print("R_min:", R_min)
print("R_max:", R_max)
print("Number of Filtered Users (Û):", len(filtered_unique_users))
print("Number of Filtered Movies (Î):", len(filtered_unique_movies))

# ------------ Task 3: Generate Frequency Histograms ------------

# Recalculate the number of ratings per user based on the filtered DataFrame
ratings_per_user_filtered = filtered_df.groupby('username')['rating'].count()

# First Histogram: Number of Ratings per User
plt.figure(figsize=(10, 5))
plt.hist(ratings_per_user_filtered, bins=range(1, max(ratings_per_user_filtered) + 2), edgecolor='black', alpha=0.7)
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
