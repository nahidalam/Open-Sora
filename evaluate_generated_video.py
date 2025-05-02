'''
python video_eval.py --samples_dir /path/to/samples --outputs_dir /path/to/generated --output_csv results.csv
'''

import os
import cv2
import numpy as np
import torch
import lpips
import requests
import base64
import csv
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- CONFIG ---
GEMINI_API_KEY = "your_gemini_key_here"
USE_GPU = torch.cuda.is_available()

# --- INIT MODELS ---
lpips_model = lpips.LPIPS(net='alex')
if USE_GPU:
    lpips_model = lpips_model.cuda()
lpips_model.eval()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Gemma 3 model from Huggingface
print("Loading Gemma 3 model...")
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it", device_map="auto")
gemma_pipeline = pipeline("text-generation", model=model_gemma, tokenizer=tokenizer_gemma)

# --- LOW-LEVEL METRICS ---
def compute_video_metrics(src_path, gen_path):
    cap_src = cv2.VideoCapture(src_path)
    cap_gen = cv2.VideoCapture(gen_path)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    while True:
        ret_src, frame_src = cap_src.read()
        ret_gen, frame_gen = cap_gen.read()

        if not ret_src or not ret_gen:
            break

        frame_src = cv2.resize(frame_src, (frame_gen.shape[1], frame_gen.shape[0]))

        gray_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

        psnr = peak_signal_noise_ratio(gray_src, gray_gen, data_range=255)
        ssim = structural_similarity(gray_src, gray_gen, data_range=255)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        tensor_src = torch.from_numpy(frame_src).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        tensor_gen = torch.from_numpy(frame_gen).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        if USE_GPU:
            tensor_src = tensor_src.cuda()
            tensor_gen = tensor_gen.cuda()

        lpips_val = lpips_model(tensor_src, tensor_gen)
        lpips_scores.append(lpips_val.item())

    cap_src.release()
    cap_gen.release()

    return {
        "psnr": np.mean(psnr_scores) if psnr_scores else 0.0,
        "ssim": np.mean(ssim_scores) if ssim_scores else 0.0,
        "lpips": np.mean(lpips_scores) if lpips_scores else 1.0
    }

# --- VLM INTERFACE (GEMINI) ---
def describe_frame_with_vlm(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    b64_img = base64.b64encode(img_encoded).decode('utf-8')

    payload = {
        "contents": [
            {
                "parts": [{
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_img
                    }
                }]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
        headers=headers,
        json=payload
    )

    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return "Error or empty response"

def compare_descriptions(desc1, desc2):
    emb1 = embed_model.encode(desc1, convert_to_tensor=True)
    emb2 = embed_model.encode(desc2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def evaluate_semantics_with_vlm_multi(src_path, gen_path):
    cap_src = cv2.VideoCapture(src_path)
    cap_gen = cv2.VideoCapture(gen_path)

    total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [0, total_frames // 2, max(total_frames - 1, 0)]

    scores, src_descs, gen_descs = [], [], []

    for idx in indices:
        cap_src.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_gen.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, frame1 = cap_src.read()
        ret2, frame2 = cap_gen.read()
        if not ret1 or not ret2:
            continue
        desc1 = describe_frame_with_vlm(frame1)
        desc2 = describe_frame_with_vlm(frame2)
        src_descs.append(desc1)
        gen_descs.append(desc2)
        scores.append(compare_descriptions(desc1, desc2))

    cap_src.release()
    cap_gen.release()

    return {
        "vlm_score": float(np.mean(scores)) if scores else 0.0,
        "src_desc": " | ".join(src_descs),
        "gen_desc": " | ".join(gen_descs)
    }

# --- GEMMA 3 DESCRIPTION FUNCTION ---
def summarize_video_with_gemma(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [0, total_frames // 2, total_frames - 1]

    descriptions = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_resized = cv2.resize(frame, (224, 224))
        _, img_encoded = cv2.imencode(".jpg", frame_resized)
        img_bytes = base64.b64encode(img_encoded).decode("utf-8")

        prompt = f"This is a frame from a video. Describe what is happening in the scene:\n[image base64: {img_bytes}]"
        response = gemma_pipeline(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        descriptions.append(response.strip())

    cap.release()
    return " | ".join(descriptions)

# --- FULL EVAL ---
def evaluate_all(samples_dir, outputs_dir):
    all_results = {}
    for filename in tqdm(os.listdir(samples_dir)):
        if not filename.endswith(".mp4"):
            continue

        src_path = os.path.join(samples_dir, filename)
        gen_path = os.path.join(outputs_dir, filename)

        if not os.path.exists(gen_path):
            print(f"Missing: {filename}")
            continue

        low_level = compute_video_metrics(src_path, gen_path)
        high_level = evaluate_semantics_with_vlm_multi(src_path, gen_path)
        gemma_src = summarize_video_with_gemma(src_path)
        gemma_gen = summarize_video_with_gemma(gen_path)
        gemma_score = compare_descriptions(gemma_src, gemma_gen)

        all_results[filename] = {
            **low_level,
            **high_level,
            "gemma_score": gemma_score,
            "gemma_src_desc": gemma_src,
            "gemma_gen_desc": gemma_gen
        }

    return all_results

# --- EXPORT ---
def export_results_to_csv(results, output_path="video_eval_report.csv"):
    fieldnames = [
        "filename", "psnr", "ssim", "lpips",
        "vlm_score", "src_desc", "gen_desc",
        "gemma_score", "gemma_src_desc", "gemma_gen_desc"
    ]

    with open(output_path, mode="w", newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for filename, metrics in results.items():
            row = {"filename": filename}
            row.update(metrics)
            writer.writerow(row)

# --- MAIN ENTRY ---
def main():
    parser = argparse.ArgumentParser(description="Video Evaluation Script")
    parser.add_argument("--samples_dir", required=True, help="Directory containing original sample videos")
    parser.add_argument("--outputs_dir", required=True, help="Directory containing generated videos to evaluate")
    parser.add_argument("--output_csv", default="video_eval_report.csv", help="Path to save the evaluation CSV")
    args = parser.parse_args()

    results = evaluate_all(args.samples_dir, args.outputs_dir)
    export_results_to_csv(results, args.output_csv)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
