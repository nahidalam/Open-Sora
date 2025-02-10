#!/bin/bash
set -e  # Exit immediately if a command fails

echo "1. Converting dataset to CSV..."
python -m tools.datasets.convert video ~/dataset --output meta.csv

echo "2. Getting video information..."
python -m tools.datasets.datautil meta.csv --info --fmin 1

echo "3.1. Generating captions..."
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava meta_info_fmin1.csv --dp-size 8 --tp-size 1 --model-path liuhaotian/llava-v1.6-mistral-7b --prompt video

echo "3.1.1. Merging generated captions..."
python -m tools.datasets.datautil meta_info_fmin1_caption_part*.csv --output meta_caption.csv

echo "3.1.2. Merging captions and video info..."
python -m tools.datasets.datautil meta_info_fmin1.csv --intersection meta_caption.csv --output meta_caption_info.csv

echo "3.1.3. Cleaning captions..."
python -m tools.datasets.datautil meta_caption_info.csv --clean-caption --refine-llm-caption --remove-empty-caption --output meta_caption_processed.csv

echo "3.2. Extracting captions..."
python -m tools.datasets.datautil meta_info_fmin1.csv --load-caption json --remove-empty-caption --clean-caption

echo "4.1. Running aesthetic scoring..."
torchrun --standalone --nproc_per_node 8 -m tools.scoring.aesthetic.inference meta_caption_processed.csv
python -m tools.datasets.datautil meta_caption_processed_part*.csv --output meta_caption_processed_aes.csv

echo "4.2. Running optical flow scoring..."
torchrun --standalone --nproc_per_node 8 -m tools.scoring.optical_flow.inference meta_caption_processed.csv

echo "4.3. Running matching scoring..."
torchrun --standalone --nproc_per_node 8 -m tools.scoring.matching.inference meta_caption_processed.csv

echo "4.4. Detecting camera motion..."
python -m tools.caption.camera_motion_detect meta_caption_processed.csv

echo "Pipeline completed successfully!"

