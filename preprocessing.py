import os
import csv

# Directory paths for the CSV files
movies_dir = "dataset/1_movies_per_genre"
reviews_dir = "dataset/2_reviews_per_movie_raw"


# Function to read and extract objects from CSV files
def extract_data_from_csv(directory, field_name):
    objects = set()
    files = os.listdir(directory)
    for file in files:
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                objects.add(row[field_name])
    return objects


# Extract users and movie objects from both directories
unique_users = extract_data_from_csv(reviews_dir, 'username')
unique_movies = extract_data_from_csv(movies_dir, 'name')

# Print the sets of unique users and movie objects
print("Unique Users:", unique_users)
print("Unique Movies:", unique_movies)
