import argparse
import os
from rembg import remove
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

def process_image(input_path, output_path):
    """
    Dummy image processing function that just resaves the image
    """
    img = Image.open(input_path)

    # step 1: remove background
    output = remove(img, only_mask=False)

    # step 2: generate new background using stable diffusion
    # Initialize the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Convert mask to proper format (black and white, where white is the area to inpaint)
    mask_array = np.array(output.split()[3])  # Get alpha channel
    inpaint_mask = Image.fromarray(255 - mask_array)  # Invert mask since we want to fill background
    
    # Convert RGBA to RGB for stable diffusion input
    rgb_image = Image.new("RGB", output.size, (255, 255, 255))
    rgb_image.paste(output, mask=output.split()[3])

    # Generate new background
    result = pipe(
        prompt="construction side view",
        image=rgb_image,
        mask_image=inpaint_mask,
        num_inference_steps=200,
        guidance_scale=10.0
    ).images[0]

    result = result.resize(output.size)
    
    # Composite original foreground with new background
    output = Image.composite(output, result, output.split()[3])

    # step 3: save image

    output.save(output_path)

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Process images from input directory to output directory')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process all images in input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            
            try:
                process_image(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
