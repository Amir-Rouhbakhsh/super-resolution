import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import gradio as gr
import random
import os

# Load your trained model (adjust the path to your model)
generator = torch.load('generator_model.pt')
generator.eval()

# Preprocessing function for the test
def preprocess_image(image):
    # Resize input image to 256x256 as expected by the model
    image = image.resize((256, 256))
    image = np.array(image).astype(np.float32)
    
    # Normalize the image to [-1, 1]
    image = (image / 127.5) - 1.0
    
    # Convert to tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    return image

# Function for generating super-resolved image (inference)
def generate_sr_image(lr_image):
    # Preprocess the low-resolution image
    lr_image = preprocess_image(lr_image)
    
    # Run through the model
    with torch.no_grad():
        sr_image = generator(lr_image)
    
    # Convert output back to numpy for display
    sr_image = sr_image.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Rescale the output image to [0, 1] for display
    sr_image = (sr_image + 1.0) / 2.0
    return sr_image

# Gradio interface
interface = gr.Interface(
    fn=generate_sr_image, 
    inputs=gr.Image(type="pil", label="Upload Low-Resolution Image (256x256)"), 
    outputs=gr.Image(type="numpy", label="Super-Resolved Image (1024x1024)"), 
    live=False,  # Update only after clicking the button
    title="SRGAN Super-Resolution",
    description="Upload a 256x256 low-resolution image, and the model will upscale it to 1024x1024."
)

# Launch the Gradio app
interface.launch()