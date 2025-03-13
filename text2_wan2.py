import torch
import pandas as pd
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# Read data from Excel
df = pd.read_excel("t2v_Benchmark_1000_Trimmed.xlsx")

# Load model and VAE
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

# Configure flow shift and move pipeline to GPU
flow_shift = 3.0  # Use 5.0 for 720p, 3.0 for 480p
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.to("cuda")

# Fixed negative prompt
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, "
    "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)

# Generate videos row by row
for idx, row in df.iterrows():
    prompt_text = row["Description"]      # Prompt from Description column
    sample_id = row["Sample Id"]          # Output filename prefix from Sample Id column

    print(f"Processing row {idx}: {sample_id}")

    # Run the pipeline
    output_frames = pipe(
        prompt=prompt_text,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
    ).frames[0]

    # Export video using sample_id as the filename
    output_filename = f"{sample_id}.mp4"
    export_to_video(output_frames, output_filename, fps=16)

    print(f"Video saved as {output_filename}")

