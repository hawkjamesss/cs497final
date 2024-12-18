import pandas as pd

def remove_genres(csv_file, genres_to_remove, output_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Filter out rows with genres to remove
    filtered_data = data[~data['genre'].isin(genres_to_remove)]
    
    # Save the cleaned data to a new CSV file
    filtered_data.to_csv(output_file, index=False)

# Replace 'music_data.csv' with the actual file name
csv_file = 'devall.csv'
output_file = 'dev.csv'
genres_to_remove = [
    'rock', 'pop'
]

remove_genres(csv_file, genres_to_remove, output_file)

print(f"Genres removed. Cleaned data saved to {output_file}.")
