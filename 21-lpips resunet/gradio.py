import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# Load your trained model (the generator part of SRGAN)
generator = tf.keras.models.load_model('H:/GPU/srgan thesis/21-lpips resunet/generator_model.h5')

# Define the evaluation function
def evaluate(lr_image):
    # Preprocess the input image (resize and normalize)
    lr_image = tf.image.resize(lr_image, (256, 256))  # Resize if needed
    lr_image = lr_image / 127.5 - 1  # Normalize to [-1, 1]
    lr_image = np.expand_dims(lr_image, axis=0)  # Add batch dimension

    # Generate super-resolved image
    sr_image = generator(lr_image, training=False)

    # Post-process the output image
    sr_image = (sr_image + 1.0) / 2.0  # Rescale to [0, 1]
    sr_image = np.squeeze(sr_image)  # Remove batch dimension

    return sr_image

# Create the Gradio interface
interface = gr.Interface(fn=evaluate, 
                         inputs=gr.inputs.Image(type="array"), 
                         outputs=gr.outputs.Image(type="numpy"),
                         live=True,  # Updates live as the user uploads an image
                         title="SRGAN Super-Resolution",
                         description="Upload a low-resolution image to get the super-resolved output.")

# Launch the Gradio app
interface.launch()