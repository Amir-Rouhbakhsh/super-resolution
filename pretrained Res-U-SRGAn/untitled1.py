import tensorflow as tf
from tensorflow.keras import layers
import os

import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

from tensorflow.keras import models
import random

import pandas as pd

from tensorflow.keras.callbacks import Callback
from tensorflow.image import psnr, ssim
import numpy as np


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path


def Res_UNet(input_tensor, dropout_rate=0.0, batch_norm=True):
    """
    U-Net architecture used for SRGAN generator.
    """
    FILTER_NUM = 64  # number of filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters

    #inputs = layers.Input(input_shape, dtype=tf.float32)
    x = layers.Conv2D(64, (1, 1), padding='same')(input_tensor)  # Expand channels to 64

    # Downsampling layers
    conv_128 = res_conv_block(x, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Final 1x1 convolution to map to the number of channels (RGB output)
    conv_final = layers.Conv2D(64, kernel_size=(1, 1), padding="same")(up_conv_128)
    conv_final = layers.Activation('tanh')(conv_final)

    # Model
    model = models.Model(input_tensor, conv_final, name="UNet")
    return model
    #return conv_final
    
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    


def Res_Autoencoder(input_tensor, dropout_rate=0.0, batch_norm=True):
    """
    Autoencoder architecture based on the encoder and decoder of the Res-UNet.
    """
    FILTER_NUM = 64  # number of filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters

    #inputs = layers.Input(input_shape, dtype=tf.float32)
    x = layers.Conv2D(64, (1, 1), padding='same')(input_tensor)  # Expand channels to 64
    # Encoder (downsampling path)
    conv_128 = res_conv_block(x, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)

    # Bottleneck
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Decoder (upsampling path)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_8)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_16)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_32)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_64)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Final output layer
    output = layers.Conv2D(input_shape[2], kernel_size=(1, 1), padding="same", activation="tanh")(up_conv_128)

    # Model
    autoencoder = models.Model(input_tensor, output, name="Res_Autoencoder")
    return autoencoder

# Example usage
input_shape = (64, 64, 3)  # Adjust as needed
inputs = layers.Input(input_shape, dtype=tf.float32)
autoencoder = Res_Autoencoder(inputs)
autoencoder.summary()
   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
matplotlib.use('TkAgg')   
def preprocess_image(image, hr_size=(256, 256)):
    """
    Preprocess the high-resolution image by resizing and normalizing.
    Also, downsample to create a low-resolution image.
    """
    # Resize the HR image
    hr_image = tf.image.resize(image, hr_size)

    # Downsample the HR image to create the LR image
    lr_image = tf.image.resize(hr_image, (hr_size[0] // 4, hr_size[1] // 4))  # 4x downsampling
    

    # Normalize the images to [-1, 1] for compatibility with tanh activation
    hr_image = hr_image / 127.5 - 1
    lr_image = lr_image / 127.5 - 1 
    # Normalize the images to [0, 1] range
    #hr_image = hr_image / 255.0
    #lr_image = lr_image / 255.0

    return lr_image, hr_image
'''
# Example usage with a TFRecord or image dataset:
def load_dataset(data_dir, batch_size=16):
    """
    Loads and preprocesses the dataset of images from a directory.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, image_size=(128, 128), batch_size=batch_size)

    # Apply preprocessing to each image
    dataset = dataset.map(lambda x, y: preprocess_image(x))

    return dataset

# Load your dataset (replace `data_dir` with the path to your dataset)
# dataset = load_dataset(data_dir, batch_size=16)
'''

def load_dataset(data_dir, batch_size=1, hr_size=(256, 256)):
    """
    Loads and preprocesses the dataset of images from a directory.
    The images are resized, downsampled to create LR images, and then normalized.
    """
    # Load dataset from directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=None,  # We don't need labels for this task
        image_size=hr_size,  # Resize images to HR size
        batch_size=batch_size,
        shuffle=True
    )

    # Apply preprocessing function to each image in the dataset
    dataset = dataset.map(lambda image: preprocess_image(image, hr_size))
    
    # Prefetch to improve performance during training
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

#Example of Loading the Dataset

data_dir = "H:/GPU/srgan thesis/5-Srgan tensorflow/hr_train_LR"  # Replace this with the actual dataset path
batch_size = 1
hr_size = (256, 256)

# Load and preprocess the dataset
dataset = load_dataset(data_dir, batch_size=batch_size, hr_size=hr_size)

# Example of checking one batch of preprocessed LR and HR images
for lr_images, hr_images in dataset.take(1):
    print(f"Low-resolution image shape: {lr_images.shape}")
    print(f"High-resolution image shape: {hr_images.shape}")
    
#Visualizing the Data (Optional)    


def visualize_lr_hr(lr_image, hr_image):
    """
    Displays a low-resolution and high-resolution image side by side.
    """
    
    # Rescale images to [0, 1] for display
    lr_image = (lr_image + 1.0) / 2.0
    hr_image = (hr_image + 1.0) / 2.0

    plt.figure(figsize=(8, 4))

    # Display LR image
    plt.subplot(1, 2, 1)
    plt.imshow(tf.clip_by_value(lr_image, 0.0, 1.0))
    plt.title("Low-Resolution Image")
    plt.axis("off")

    # Display HR image
    plt.subplot(1, 2, 2)
    plt.imshow(tf.clip_by_value(hr_image, 0.0, 1.0))
    plt.title("High-Resolution Image")
    plt.axis("off")

    plt.show()

# Example of visualizing images from the dataset
for lr_images, hr_images in dataset.take(1):
    visualize_lr_hr(lr_images[0], hr_images[0])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lr_image_list = []

# Extract high-resolution images from the dataset and store them in a list
for lr_images, hr_images in dataset:
    lr_image_list.append(lr_images.numpy())

# Convert the list of high-resolution images to a numpy array for training
lr_image_array = np.vstack(lr_image_list)



hr_image_list = []

# Extract high-resolution images from the dataset and store them in a list
for lr_images, hr_images in dataset:
    hr_image_list.append(hr_images.numpy())

# Convert the list of high-resolution images to a numpy array for training
hr_image_array = np.vstack(hr_image_list)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_random_images(image_array, num_images=5):
    """
    Plots a specified number of random images from the given image array.
    
    Parameters:
    - image_array: The array of images (assumed to be in the range [-1, 1])
    - num_images: Number of random images to plot
    """
    # Rescale the images to [0, 1] for display
    rescaled_images = (image_array + 1.0) / 2.0

    # Randomly choose `num_images` indices
    random_indices = np.random.choice(image_array.shape[0], num_images, replace=False)

    plt.figure(figsize=(12, 6))

    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(np.clip(rescaled_images[idx], 0.0, 1.0))
        #plt.imshow(np.clip(image_array[idx], 0.0, 1.0))  # No need to rescale, just clip
        plt.title(f"Image {idx}")
        plt.axis("off")

    plt.show()

# Now, you can plot some random images
plot_random_images(lr_image_array, num_images=5)

plot_random_images(hr_image_array, num_images=5)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from tensorflow.keras.callbacks import Callback

class StopAtAccuracy(Callback):
    def __init__(self, target_accuracy=0.95):
        super(StopAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy has reached the target
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy >= self.target_accuracy:
            print(f"\nReached {self.target_accuracy * 100}% accuracy, stopping training!")
            self.model.stop_training = True


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Train the autoencoder using high-resolution images
autoencoder_model = Res_Autoencoder(inputs)  # Shape of a single image
autoencoder_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(autoencoder_model.summary())


stop_callback = StopAtAccuracy(target_accuracy=0.95)
# Train the autoencoder (using hr_image_array for both input and output)
#history = autoencoder_model.fit(lr_image_array, lr_image_array, epochs=100, verbose=1)
# Train the autoencoder and stop when accuracy > 90%
history = autoencoder_model.fit(
    lr_image_array, lr_image_array, 
    epochs=3000,  # Set a high number of epochs, but training will stop earlier
    verbose=1, 
    callbacks=[stop_callback]
)



def plot_training_history(history):
    """
    Plots the accuracy and loss from the training history.
    
    Parameters:
    - history: The history object from the model.fit() function.
    """
    # Retrieve accuracy and loss values
    accuracy = history.history.get('accuracy', [])
    loss = history.history.get('loss', [])
    epochs = range(1, len(accuracy) + 1)

    # Plot Accuracy
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Call the function to plot the accuracy and loss
plot_training_history(history)


    

save_path = r'H:/GPU/srgan thesis/14/autoencoder_trained64.h5'
autoencoder_model.save(save_path)

# Save the trained autoencoder model
autoencoder_model.save('autoencoder_trained64.h5')
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg, which supports interactivity
#autoencoder_model = load_model("autoencoder_trained.h5", compile=False)

autoencoder_model = load_model(save_path, compile=False)

num = random.randint(0, len(lr_image_array) - 1)
test_img = np.expand_dims(lr_image_array[num], axis=0)  # Select a random image and expand dimensions

# Predict using the autoencoder (reconstruct the image)
pred = autoencoder_model.predict(test_img)

# Since images were rescaled to [-1, 1] during training, we need to rescale them back to [0, 1] for visualization
test_img_rescaled = (test_img[0] + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
pred_rescaled = (pred[0] + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]



'''
# Plot the original and reconstructed images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)

#plt.imshow(np.clip(test_img_rescaled, 0.0, 1.0))  # Clip values to [0, 1]
plt.imshow(np.clip(test_img[0], 0.0, 1.0))  # No need to rescale, just clip
plt.title('Original Image')

plt.subplot(1, 2, 2)
#plt.imshow(np.clip(pred_rescaled, 0.0, 1.0))  # Clip values to [0, 1]
plt.imshow(np.clip(pred[0], 0.0, 1.0))  # No need to rescale, just cli
plt.title('Reconstructed Image')

plt.show()
'''
# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(np.clip(test_img_rescaled, 0.0, 1.0))  # Clip values to [0, 1]
plt.title('Original Image')
plt.axis('off')

# Display the reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(np.clip(pred_rescaled, 0.0, 1.0))  # Clip values to [0, 1]
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Now let us define a Unet with same encoder part as out autoencoder. 
#Then load weights from the original autoencoder for the first 35 layers (encoder)
input_shape = (64, 64, 3)

inputs = layers.Input(input_shape, dtype=tf.float32)
unet_model =Res_UNet(inputs)

#Print layer names for each model to verify the layers....
#First 35 layers should be the same in both models. 
unet_layer_names=[]
for layer in unet_model.layers:
    unet_layer_names.append(layer.name)

autoencoder_layer_names = []
for layer in autoencoder_model.layers:
    autoencoder_layer_names.append(layer.name)
    
#Make sure the first 35 layers are the same. Remember that the exct names of the layers will be different.
###########

#Set weights to encoder part of the U-net (first 35 layers)
for l1, l2 in zip(unet_model.layers[0:51], autoencoder_model.layers[0:51]):
    l1.set_weights(l2.get_weights())



unet_save_path = r'H:/GPU/srgan thesis/14/unet_with_transferred_weights64.h5'
unet_model.save(unet_save_path)

print(f"U-Net model saved with transferred weights at {unet_save_path}")
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np

# Verify weight transfer by comparing weights for the first 50 layers
weights_match = True  # Flag to check if all weights match

for i, (layer_unet, layer_autoencoder) in enumerate(zip(unet_model.layers[:51], autoencoder_model.layers[:51])):
    weights_unet = layer_unet.get_weights()
    weights_autoencoder = layer_autoencoder.get_weights()

    # Check if weights and biases are identical for each layer
    if all(np.array_equal(w_u, w_a) for w_u, w_a in zip(weights_unet, weights_autoencoder)):
        print(f"Layer {i + 1}: Weights match.")
    else:
        print(f"Layer {i + 1}: Weights do not match.")
        weights_match = False

if weights_match:
    print("All transferred weights match between the autoencoder and U-Net model.")
else:
    print("Some weights do not match. Please verify your weight transfer.")













    
    
    
    