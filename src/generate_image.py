import ollama
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Function to generate image description using Ollama
def generate_image_prompt(topic="a futuristic city at sunset"):
    system_prompt = "Describe an image in a way that an AI model can generate it. Include lighting, colors, and mood."
    response = ollama.chat(model="llama3.2", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate an image of {topic}"}
    ])
    return response['message']['content']

# Function to generate image using Stable Diffusion
def generate_image_from_prompt(prompt, output_path="generated_image.png"):
    model_id = "runwayml/stable-diffusion-v1-5"  # Open-source model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    # Generate the image
    image = pipe(prompt).images[0]

    # Save and display the image
    image.save(output_path)
    image.show()

    return output_path

# Run the process
image_prompt = generate_image_prompt("a cyberpunk city at night with neon lights")
print("Generated Image Prompt:", image_prompt)

output_file = generate_image_from_prompt(image_prompt)
print(f"Image saved to {output_file}")
