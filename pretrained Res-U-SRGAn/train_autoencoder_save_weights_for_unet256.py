# https://youtu.be/hTpq9lzAb8M
"""
@author: Sreenivas Bhattiprolu

Train an autoencoder for a given type of images. e.g. EM images of cells.
Use the encoder part of the trained autoencoder as the encoder for a U-net.
Use pre-trained weights from autoencoder as starting weights for encoder in the Unet. 
Train the Unet.

Training with initial encoder pre-trained weights would dramatically speed up 
the training process of U-net. 

"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential
import os
from keras.models import Model
from matplotlib import pyplot as plt
       
import random
SIZE=256


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    #hr_image = hr_image / 127.5 - 1
    #lr_image = lr_image / 127.5 - 1 
    # Normalize the images to [0, 1] range
    hr_image = hr_image / 255.0
    lr_image = lr_image / 255.0

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
    #lr_image = (lr_image + 1.0) / 2.0
    #hr_image = (hr_image + 1.0) / 2.0

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
        #plt.imshow(np.clip(rescaled_images[idx], 0.0, 1.0))
        plt.imshow(np.clip(image_array[idx], 0.0, 1.0))  # No need to rescale, just clip
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
from models import build_autoencoder, build_encoder, build_unet , build_resunet 
# Now, `hr_image_array` contains all the high-resolution images needed for training the autoencoder

# Train the autoencoder using high-resolution images
autoencoder_model = build_autoencoder(hr_image_array.shape[1:])  # Shape of a single image
autoencoder_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(autoencoder_model.summary())


stop_callback = StopAtAccuracy(target_accuracy=0.95)
# Train the autoencoder (using hr_image_array for both input and output)
#history = autoencoder_model.fit(lr_image_array, lr_image_array, epochs=100, verbose=1)
# Train the autoencoder and stop when accuracy > 90%
history = autoencoder_model.fit(
    hr_image_array, hr_image_array, 
    epochs=10000,  # Set a high number of epochs, but training will stop earlier
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


















save_path = r'H:/GPU/srgan thesis/13-pretrainedunet/autoencoder_trained255.h5'
autoencoder_model.save(save_path)

# Save the trained autoencoder model
autoencoder_model.save('autoencoder_trained255.h5')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg, which supports interactivity
#autoencoder_model = load_model("autoencoder_trained.h5", compile=False)

autoencoder_model = load_model(save_path, compile=False)

num = random.randint(0, len(hr_image_array) - 1)
test_img = np.expand_dims(hr_image_array[num], axis=0)  # Select a random image and expand dimensions

# Predict using the autoencoder (reconstruct the image)
pred = autoencoder_model.predict(test_img)

# Since images were rescaled to [-1, 1] during training, we need to rescale them back to [0, 1] for visualization
#test_img_rescaled = (test_img[0] + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
#pred_rescaled = (pred[0] + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

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
#test on a few images
#Load the model 
from keras.models import load_model
autoencoder_model = load_model("autoencoder_trained.h5", compile=False)

num=random.randint(0, len(img_array2)-1)
test_img = np.expand_dims(img_array[num], axis=0)
pred = autoencoder_model.predict(test_img)

plt.subplot(1,2,1)
plt.imshow(test_img[0])
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(pred[0].reshape(SIZE,SIZE,3))
plt.title('Reconstructed')
plt.show()

'''
########################################################
###############################

#Extract weights only for the encoder part of the Autoencoder. 

#from models import build_autoencoder
#from keras.models import load_model
#autoencoder_model = load_model("autoencoder_mito_500imgs_100epochs.h5", compile=False)
autoencoder_model = load_model(save_path, compile=False)       
#Now define encoder model only, without the decoder part. 
input_shape = (256, 256, 3)
input_img = Input(shape=input_shape)

encoder = build_encoder(input_img)
encoder_model = Model(input_img, encoder)
print(encoder_model.summary())

num_encoder_layers = len(encoder_model.layers) #16 layers in our encoder. 

#Get weights for the 35 layers from trained autoencoder model and assign to our new encoder model 
for l1, l2 in zip(encoder_model.layers[:16], autoencoder_model.layers[0:16]):
    l1.set_weights(l2.get_weights())



# Compare weights of the first layer of the autoencoder and encoder
autoencoder_weights = autoencoder_model.layers[0].get_weights()  # Weights of the first layer of autoencoder
encoder_weights = encoder_model.layers[0].get_weights()  # Weights of the first layer of encoder

# Print the weights to verify if they are the same
print("Autoencoder Weights (First Layer):", autoencoder_weights)
print("Encoder Weights (First Layer):", encoder_weights)

# Optionally, check if the weights are equal
if np.array_equal(autoencoder_weights, encoder_weights):
    print("Weights match!")
else:
    print("Weights do not match!")










'''

#Verify if the weights are the same between autoencoder and encoder only models. 
autoencoder_weights = autoencoder_model.get_weights()[0][1]
encoder_weights = encoder_model.get_weights()[0][1]

#Save encoder weights for future comparison
np.save('pretrained_encoder-weights.npy', encoder_weights )

weights_path = r'H:/GPU/srgan thesis/13-pretrainedunet/pretrained_encoder-weights.npy'
np.save(weights_path, encoder_weights)
'''
'''
#Check the output of encoder_model on a test image
#Should be of size 16x16x1024 for our model
temp_img = cv2.imread('data/mito/images/img9.tif',1)
temp_img = temp_img.astype('float32') / 255.
temp_img = np.expand_dims(temp_img, axis=0)
temp_img_encoded=encoder_model.predict(temp_img)

#Plot a few encoded channels
'''
################################################
#Now let us define a Unet with same encoder part as out autoencoder. 
#Then load weights from the original autoencoder for the first 35 layers (encoder)
input_shape = (256, 256, 3)
unet_model = build_resunet(input_shape)

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
for l1, l2 in zip(unet_model.layers[0:16], autoencoder_model.layers[0:16]):
    l1.set_weights(l2.get_weights())



unet_save_path = r'H:/GPU/srgan thesis/13-pretrainedunet/unet_with_transferred_weights256.h5'
unet_model.save(unet_save_path)

print(f"U-Net model saved with transferred weights at {unet_save_path}")


'''
from keras.optimizers import Adam
import segmentation_models as sm
unet_model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
#unet_model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()
print(unet_model.output_shape)

unet_model.save('unet_model_weights.h5')

'''
#Now use train_unet to load these weights for encoder and train a unet model. 
###################################################################
