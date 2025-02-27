import os
import subprocess
import pandas as pd
import torch
import clip
import lpips
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from torchvision.models import inception_v3
from torchvision import models

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# LPIPS model
lpips_model = lpips.LPIPS(net='vgg').to(device)

# Load InceptionV3 for FID
inception = inception_v3(pretrained=True, transform_input=False).to(device).eval()

# Load sample.xlsx
df = pd.read_excel("sample.xlsx")

# WAN2.1 Command Template
WAN2_CMD_TEMPLATE = (
    "python generate.py --task t2v-1.3B --size 832*480 "
    "--ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 "
    '--prompt "{description}" --output ./sample/{filename}'
)

# Ensure output directory
os.makedirs("sample", exist_ok=True)

# Generate videos
for _, row in df.iterrows():
    sample_id = row["Sample Id"]
    description = row["Description"]
    output_file = f"{sample_id}.mp4"

    # Run WAN2.1
    command = WAN2_CMD_TEMPLATE.format(description=description, filename=output_file)
    subprocess.run(command, shell=True, check=True)

print("Text-to-video generation completed.")

# Convert frames to feature vectors for FID
def compute_inception_features(images):
    images = torch.stack([transforms.ToTensor()(img).to(device) for img in images])
    images = transforms.Resize((299, 299))(images)
    with torch.no_grad():
        features = inception(images).cpu().numpy()
    return features

# Compute CLIPScore
def compute_clip_score(frames, text):
    text_features = clip_model.encode_text(clip.tokenize([text]).to(device)).detach().cpu().numpy()
    frame_features = [clip_model.encode_image(clip_preprocess(frame).unsqueeze(0).to(device)).detach().cpu().numpy() for frame in frames]
    
    similarities = [np.dot(text_features, frame_feat.T)[0][0] for frame_feat in frame_features]
    return np.mean(similarities)

# Compute LPIPS
def compute_lpips(frames):
    scores = []
    for i in range(len(frames) - 1):
        img1 = transforms.ToTensor()(frames[i]).to(device)
        img2 = transforms.ToTensor()(frames[i + 1]).to(device)
        scores.append(lpips_model(img1.unsqueeze(0), img2.unsqueeze(0)).item())
    return np.mean(scores)

# Compute Temporal Consistency (SSIM)
def compute_temporal_consistency(frames):
    scores = []
    for i in range(len(frames) - 1):
        img1 = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(frames[i + 1]), cv2.COLOR_RGB2GRAY)
        scores.append(ssim(img1, img2))
    return np.mean(scores)

# Compute FID
def compute_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

# Evaluation function
def evaluate_generated_videos(sample_dir="sample", real_dir="real_videos"):
    results = []
    real_videos = os.listdir(real_dir)

    for video in os.listdir(sample_dir):
        if not video.endswith(".mp4"):
            continue
        
        video_path = os.path.join(sample_dir, video)
        sample_id = video.split(".")[0]
        text_prompt = df[df["Sample Id"] == sample_id]["Description"].values[0]

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames:
            continue

        # Compute metrics
        clip_score = compute_clip_score(frames, text_prompt)
        lpips_score = compute_lpips(frames)
        temp_consistency = compute_temporal_consistency(frames)

        # Compute FID with a random real video
        real_video_path = os.path.join(real_dir, np.random.choice(real_videos))
        cap = cv2.VideoCapture(real_video_path)
        real_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            real_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        gen_features = compute_inception_features(frames)
        real_features = compute_inception_features(real_frames)
        fid_score = compute_fid(real_features, gen_features)

        results.append((sample_id, clip_score, lpips_score, temp_consistency, fid_score))

    # Save results
    results_df = pd.DataFrame(results, columns=["Sample Id", "CLIPScore", "LPIPS", "Temporal Consistency", "FID"])
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation completed. Results saved to evaluation_results.csv.")

# Run evaluation
evaluate_generated_videos()

