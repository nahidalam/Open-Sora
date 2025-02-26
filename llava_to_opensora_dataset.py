import json
import csv
import os
from PIL import Image

# Input JSON file
json_file = "blip_laion_cc_sbu_558k.json"
image_base_dir = "/dev/data/images/"  # Base directory for images
output_csv = "output.csv"

# Load the JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Prepare CSV file
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["path", "text", "num_frames", "width", "height", "aspect_ratio"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item in data:
        image_path = os.path.join(image_base_dir, item["image"])  # Construct absolute path
        text = next((conv["value"] for conv in item["conversations"] if conv["from"] == "gpt"), "")
        num_frames = 1  # Since it's a single image

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = round(width / height, 2) if height > 0 else 0
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            width, height, aspect_ratio = None, None, None  # Handle missing/corrupt images

        # Write to CSV
        writer.writerow({
            "path": image_path,
            "text": text,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
        })

print(f"CSV file saved as {output_csv}")

