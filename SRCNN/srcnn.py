# Importing libraries
import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import os

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Defining Image Quality Matric Functions

def compare_ssim(target, ref):
    # Ensure the images are the same size
    if target.shape != ref.shape:
        print("Target and reference images must have the same dimensions.")
        return None

    # Calculate SSIM with appropriate window size and channel handling
    try:
        # Set win_size to 3 (you can adjust this based on your image size)
        ssim_value, _ = ssim(target, ref, full=True, multichannel=True, win_size=3, channel_axis=-1)
        return ssim_value
    except ValueError as e:
        print(f"SSIM calculation failed: {e}")
        return None








def psnr(target, ref):

    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err

# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(compare_ssim(target, ref))

    return scores

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#degraded images by downscaling and then upscaling them. 

# Preparing Images (degraded) by resizing
cwd = os.getcwd()  # Get the current working directory (cwd)
def prepare_images(path, factor):

    # loop through the files in the directory
    for file in os.listdir(path):
        try:
          # open the file
          img = cv2.imread(path + '/' + file)

          # find old and new image dimensions
          h, w, _ = img.shape
          new_height = int(h / factor)
          new_width = int(w / factor)

          # resize the image - down
          img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

          # resize the image - up
          img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)

          # save the image
          print('Saving {}'.format(file))
          cv2.imwrite('H:/GPU/srgan thesis/4-SRCNN/lr_train/{}'.format(file), img)
        except:
          print('ERROR for file-', file, '!')
          pass

prepare_images('H:/GPU/srgan thesis/4-SRCNN/hr_train', 2)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Testing Quality difference between source and image (degraded)
for file in os.listdir('H:/GPU/srgan thesis/4-SRCNN/lr_train/'):
    try:
      # open target and reference images
      target = cv2.imread('H:/GPU/srgan thesis/4-SRCNN/lr_train/{}'.format(file))
      ref = cv2.imread('H:/GPU/srgan thesis/4-SRCNN/hr_train/{}'.format(file))

      # calculate score
      scores = compare_images(target, ref)

      # print all three scores with new line characters (\n)
      print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    except Exception as e:
        print(f"An error occurred with file {file}: {e}")
        pass
   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''You will need to load both high-resolution and low-resolution images for training. 
Here is a way to prepare your training data, assuming your images are in separate directories for high-resolution 
and low-resolution images.
Since SRCNN is trained only on the luminance (Y) channel of images, the code converts the image 
from BGR (standard color space in OpenCV) to YCrCb color space. The Y channel represents brightness,
 while the Cr and Cb channels represent color information.
Only the Y channel will be fed into the SRCNN model for super-resolution.'''

hr_path='H:/GPU/srgan thesis/4-SRCNN/hr_train/'
lr_path='H:/GPU/srgan thesis/4-SRCNN/lr_train/'

# Load images and prepare training data
def load_training_data(hr_path, lr_path):
    hr_images = []
    lr_images = []
    
    for file in os.listdir(hr_path):
        hr_img = cv2.imread(os.path.join(hr_path, file))
        lr_img = cv2.imread(os.path.join(lr_path, file))
        
        if hr_img is not None and lr_img is not None:
            # Convert images to YCrCb and extract the Y (luminance) channel
            hr_img_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # Y channel only
            lr_img_y = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # Y channel only
            
            # Normalize images
            hr_img_y = hr_img_y.astype(np.float32) / 255.0
            lr_img_y = lr_img_y.astype(np.float32) / 255.0
            
            # Add channel axis (since the model expects 4D input)
            hr_img_y = np.expand_dims(hr_img_y, axis=-1)
            lr_img_y = np.expand_dims(lr_img_y, axis=-1)
            
            hr_images.append(hr_img_y)
            lr_images.append(lr_img_y)
    
    return np.array(hr_images), np.array(lr_images)

# Load your data
hr_images, lr_images = load_training_data(hr_path, lr_path)







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define custom PSNR metric
def psnr1(y_true, y_pred):
    # Ensure both y_true and y_pred are cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    max_pixel = 1.0  # Assuming images are normalized in the range [0, 1]
    return 10.0 * tf.math.log((max_pixel ** 2) / (tf.reduce_mean(tf.square(y_pred - y_true)))) / tf.math.log(10.0)

# Define custom SSIM metric
def ssim1(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Define SRCNN model
def model():

    # Define model type
    SRCNN = Sequential()

    # Add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='same', use_bias=True))

    # Define optimizer
    adam = Adam(learning_rate=0.0003)

    # Compile model with MSE, PSNR, and SSIM as metrics
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error', psnr1, ssim1])

    return SRCNN

# Defining SRCNN Model
srcnn = model()
srcnn.summary()
plot_model(srcnn, to_file='H:/GPU/srgan thesis/4-SRCNN/srcnn.png', show_shapes=True, show_layer_names=True)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Train the model
history = srcnn.fit(lr_images, hr_images, batch_size=1, epochs=50, validation_split=0.2)


# Save model weights
srcnn.save_weights('H:/GPU/srgan thesis/4-SRCNN/srcnn_weights.h5')

# Save the entire model
srcnn.save('H:/GPU/srgan thesis/4-SRCNN/srcnn_model.h5')

# Plot training & validation loss values
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot PSNR values
plt.subplot(2, 1, 2)
plt.plot(history.history['psnr1'], label='Train PSNR')
plt.plot(history.history['val_psnr1'], label='Validation PSNR')
plt.title('PSNR')
plt.ylabel('PSNR (dB)')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Plot SSIM values
plt.figure(figsize=(6, 4))
plt.plot(history.history['ssim1'], label='Train SSIM')
plt.plot(history.history['val_ssim1'], label='Validation SSIM')
plt.title('SSIM')
plt.ylabel('SSIM')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Image Processing Functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img
# define main prediction function


def predict(image_path):

    # Load the SRCNN model with weights
    srcnn = model()
    srcnn.load_weights('H:/GPU/srgan thesis/4-SRCNN/srcnn_weights.h5')

    # Load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread(hr_path+'{}'.format(file))

    # Check if images are loaded correctly
    if degraded is None:
        raise FileNotFoundError(f"Degraded image not found at path: {image_path}")
    if ref is None:
        raise FileNotFoundError(f"Reference image not found at path: source/{file}")

    # Preprocess the image with modcrop
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)

    # Convert the image to YCrCb (SRCNN trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)

    # Create image slice and normalize
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

    # Perform super-resolution with SRCNN
    pre = srcnn.predict(Y, batch_size=1)

    # Post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # Resize the predicted output to match the original image dimensions
    pre_resized = cv2.resize(pre[0, :, :, 0], (temp.shape[1], temp.shape[0]))

    # Ensure that the dimensions of the resized prediction match the original Y channel
    if pre_resized.shape != temp[:, :, 0].shape:
        raise ValueError(f"Shape mismatch: Predicted output {pre_resized.shape} does not match original image shape {temp[:, :, 0].shape}")

    # Copy Y channel back to image and convert to BGR
    temp[:, :, 0] = pre_resized  # Replace Y channel with super-resolved Y
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # Ensure that the reference image and output have the same dimensions by resizing
    ref_resized = cv2.resize(ref, (output.shape[1], output.shape[0]))

    # Crop images to the smallest common size (after any processing)
    common_height = min(ref_resized.shape[0], output.shape[0])
    common_width = min(ref_resized.shape[1], output.shape[1])

    ref_cropped = ref_resized[:common_height, :common_width]
    output_cropped = output[:common_height, :common_width]

    # Ensure degraded image also has matching dimensions
    degraded_cropped = cv2.resize(degraded, (common_width, common_height))

    # Image quality calculations
    scores = []
    scores.append(compare_images(degraded_cropped, ref_cropped))  # Degraded vs Reference
    scores.append(compare_images(output_cropped, ref_cropped))    # Output vs Reference

    # Return images and scores
    return ref_cropped, degraded_cropped, output_cropped, scores

# Test the function
try:
    ref, degraded, output, scores = predict('H:/GPU/srgan thesis/4-SRCNN/lr_train/IX-11-98611_0215_0810.JPG')
except FileNotFoundError as e:
    print(e)


# print all scores for all images
print('Degraded Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
print('Reconstructed Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))



# display images as subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
axs[1].set_title('Degraded')
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Performing Super Resolution using our SRCNN model on all images (degraded)
'''
for file in os.listdir('images'):
    # perform super-resolution
    try:
      ref, degraded, output, scores = predict('images/{}'.format(file))
    except:
      continue
'''  
ref, degraded, output, scores = predict('H:/GPU/srgan thesis/4-SRCNN/lr_train/IX-11-98611_0215_1058.JPG')

# display images as subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
axs[1].set_title('Degraded')
axs[1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')
axs[2].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

print('Saving {}'.format(file))
fig.savefig('output/{}.png'.format(os.path.splitext(file)[0]))
plt.close()
















