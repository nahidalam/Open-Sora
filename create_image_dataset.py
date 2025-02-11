import os
import csv
from PIL import Image

# Root dataset directory
root_dir = "/home/shapla/image_dataset"
output_csv = "image_metadata.csv"

# Collect image metadata
image_data = []

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
            image_path = os.path.join(subdir, file)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = round(width / height, 4)  # Rounded for better readability
                    
                    image_data.append([
                        os.path.abspath(image_path),
                        "this is an image",
                        1,
                        width,
                        height,
                        aspect_ratio
                    ])
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["path", "text", "num_frames", "width", "height", "aspect_ratio"])  # Header
    writer.writerows(image_data)

print(f"CSV file created: {output_csv}")

