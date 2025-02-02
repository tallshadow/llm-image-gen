# Copyright © 2023 Apple Inc.
# Modified for structured output directory

import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from stable_diffusion import StableDiffusion, StableDiffusionXL

# Ensure the images folder exists
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "../images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def generate_image(
    prompt: str,
    model_type: str = "sdxl",
    negative_prompt: str = "",
    num_images: int = 1,
    num_steps: int = 20,
    cfg_weight: float = 7.5,
    seed: int = None,
    output_filename: str = "generated_image.png",
    verbose: bool = True
):
    """
    Generate images using Stable Diffusion with MLX acceleration
    
    Args:
        prompt: Text prompt for image generation
        model_type: Model variant (sd or sdxl)
        negative_prompt: Text prompt to avoid in generation
        num_images: Number of images to generate
        num_steps: Number of diffusion steps
        cfg_weight: Classifier-free guidance weight
        seed: Random seed for reproducibility
        output_filename: Name of the saved image file
        verbose: Show generation details and memory usage
    """
    output_path = os.path.join(IMAGES_DIR, output_filename)

    # Initialize model
    if model_type == "sdxl":
        sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
        cfg_weight = cfg_weight or 0.0
    else:
        sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
        cfg_weight = cfg_weight or 7.5

    # Generate latents
    latents = sd.generate_latents(
        prompt,
        n_images=num_images,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        seed=seed,
        negative_text=negative_prompt,
    )

    # Process diffusion steps
    for x_t in tqdm(latents, total=num_steps, desc="Generating latents"):
        mx.eval(x_t)

    # Memory optimization
    if model_type == "sdxl":
        del sd.text_encoder_1, sd.text_encoder_2
    else:
        del sd.text_encoder
    del sd.unet, sd.sampler

    # Decode images
    decoded = []
    for i in tqdm(range(0, num_images, 1), desc="Decoding images"):
        decoded.append(sd.decode(x_t[i:i+1]))
        mx.eval(decoded[-1])

    # Convert and save image
    x = mx.concatenate(decoded, axis=0)
    x = (x * 255).astype(mx.uint8)
    im = Image.fromarray(np.array(x[0]))  # Take first image for single output
    im.save(output_path)

    if verbose:
        print(f"✅ Image successfully saved to {output_path}")
        peak_mem = mx.metal.get_peak_memory() / 1024**3
        print(f"💾 Peak memory used: {peak_mem:.2f}GB")

# Example usage
if __name__ == "__main__":
    generate_image(
        prompt="A Moroccan dog playing in the city of Salé with children, in the beach",
        model_type="sdxl",
        num_steps=4,
        output_filename="sale_city.png",
        seed=42
    )
