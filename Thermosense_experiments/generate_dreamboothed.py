import argparse
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def apply_colormap_viridis(image):
    """
    Apply the viridis colormap to an image.
    Assumes the input is a PIL image.
    """
    # Convert the PIL image to a grayscale numpy array
    image = image.convert('L')  # Ensure the image is in grayscale (L mode)
    image = np.array(image)

    # Apply the viridis colormap
    colormap_image = plt.cm.viridis(image / 255.0)  # Normalize the image to [0, 1]

    # Convert to RGB by dropping the alpha channel
    colormap_image_rgb = (colormap_image[:, :, :3] * 255).astype(np.uint8)

    # Convert back to a PIL image
    return Image.fromarray(colormap_image_rgb)



def generate_images(model_id, prompt, vault, num_images=10, num_inference_steps=50, guidance_scale=7.5, apply_colormap=False):
    """
    Generate num_images images using the model_id diffuser with . Images saved in the vault folder.
    """
    # Load image generator
    pipe = DiffusionPipeline.from_pretrained(model_id, dtype=torch.float16).to("cuda")

    # Generate n images
    for i in range(num_images):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        # If colormap is requested, apply it to the image
        if apply_colormap:
            image = apply_colormap_viridis(image)

        # Save the image
        image.save(os.path.join(vault, f'img_{i}.png'))

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Script to generate images using a pretrained DiffusionPipeline.")
    parser.add_argument('--model_id', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt to generate the image. See the git for the prompt approriate for each diffuser trained.')
    parser.add_argument('--output_vault', type=str, required=True, help='Directory where generated images will be saved.')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps for the pipeline.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for image synthesis. Default value is 7.5 for diffusers.')
    parser.add_argument('--save_colormap_viridis', type=int, default=0, required=False, help='Boolean to apply the viridis colormap. 0 = False, 1 (or anything else :) ) = True.')

    # Get arguments
    args = parser.parse_args()

    # Generate the images with or without colormap
    generate_images(
        args.model_id,
        args.prompt,
        args.output_vault,
        args.num_images,
        args.num_inference_steps,
        args.guidance_scale,
        apply_colormap=bool(args.save_colormap_viridis)
    )

if __name__ == "__main__":
    main()
