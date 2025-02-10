import csv

# Path to the CSV file
csv_file = "image_metadata.csv"  # Change this to your file path

# Open and read the CSV file
with open(csv_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Reads as a dictionary
    keys = reader.fieldnames  # Extracts column names

# Print all keys (column names)
print("Keys (Column Names):", keys)

