import pandas as pd

def get_unique_genres(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Extract the unique genres
    unique_genres = data['genre'].unique()
    
    # Convert to a list for easier manipulation
    return list(unique_genres)

# Replace 'your_file.csv' with the actual file name
csv_file = 'devall.csv'
unique_genres = get_unique_genres(csv_file)

print("Unique genres:", unique_genres)

