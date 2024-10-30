import tensorflow as tf
from tensorflow.keras import layers, regularizers
import os

import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.image import psnr, ssim
import time


def conv_block(x, filter_size, size, dropout, batch_norm=True):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    #conv = layers.PReLU(shared_axes=[1, 2])(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    #conv = layers.PReLU(shared_axes=[1, 2])(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def UNet(input_tensor, dropout_rate=0.0, batch_norm=True):
    """
    U-Net architecture used for SRGAN generator.
    """
    FILTER_NUM = 64  # number of filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters

    #inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    conv_128 = conv_block(input_tensor, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Final 1x1 convolution to map to the number of channels (RGB output)
    conv_final = layers.Conv2D(64, kernel_size=(1, 1), padding="same")(up_conv_128)
    #conv_final = layers.Activation('tanh')(conv_final)

    # Model
    #model = models.Model(inputs, conv_final, name="UNet")
    #return model
    return conv_final



def build_unet_srgan_generator(input_shape=(64, 64, 3), num_filters=(64, 128, 256)):
    """
    Builds the SRGAN generator model using a U-Net architecture.
    """
    # Input Layer (Low-Resolution image)
    input_lr = layers.Input(shape=input_shape)

    # Initial Convolution + PReLU (this is standard SRGAN)
    x = layers.Conv2D(num_filters[0], (9, 9), padding='same')(input_lr)
    x = layers.PReLU(shared_axes=[1, 2])(x)  # PReLU with shared axes

    # U-Net Architecture
    skip=x
    #u_net_output = UNet(input_shape)(x)
    u_net_output = UNet(x)

    # Post-Residual Convolution + Skip Connection
    x = layers.Conv2D(64, (3, 3), padding='same')(u_net_output)
    x = layers.BatchNormalization()(x)
    #x = layers.Add()([skip, u_net_output])  # Skip connection after residual blocks
    x = layers.Concatenate()([x, skip])  # Concatenation instead of addition

    # Upsampling (Pixel Shuffle)
    x = layers.Conv2D(num_filters[2] * 4, (3, 3), padding='same')(x)  # Pixel shuffle needs 4x filters for 2x upsampling
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # Pixel shuffle (2x upsampling)

    x = layers.Conv2D(num_filters[2] * 4, (3, 3), padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # Another pixel shuffle (4x upsampling)

    # Final output layer (super-resolved image)
    output_sr = layers.Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

    # Build and return the model
    generator = tf.keras.models.Model(input_lr, output_sr)
    return generator




# Instantiate the generator model with 3 stacked autoencoders and print summary
generator = build_unet_srgan_generator(input_shape=(64, 64, 3))
generator.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def build_discriminator(input_shape=(256, 256, 3)):
    def conv_block(x, filters, size, strides):
        x = layers.Conv2D(filters, size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    # Input Layer (High-Resolution or Super-Resolved Image)
    input_hr = layers.Input(shape=input_shape)  # Define the input shape for the discriminator
    
    # Convolutional Blocks
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same')(input_hr)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = conv_block(x, 64, (3, 3), strides=2)
    x = conv_block(x, 128, (3, 3), strides=1)
    x = conv_block(x, 128, (3, 3), strides=2)
    x = conv_block(x, 256, (3, 3), strides=1)
    x = conv_block(x, 256, (3, 3), strides=2)
    x = conv_block(x, 512, (3, 3), strides=1)
    x = conv_block(x, 512, (3, 3), strides=2)
    
    # Flatten the feature maps to pass into Dense layers
    x = layers.Flatten()(x)  # This will convert the 4D tensor into a 2D tensor suitable for Dense layers
    
    # Fully Connected Layers
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Output 1 for binary classification (real or fake)

    # Build and return the model
    discriminator = tf.keras.models.Model(input_hr, x)
    return discriminator

# Instantiate the discriminator model and print summary
discriminator = build_discriminator(input_shape=(256, 256, 3))  # Set input shape for HR images
discriminator.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#In this code, we’re loading the pre-trained VGG19 model (without the top layers).
# The output of the block5_conv4 layer is often used for perceptual loss since it captures high-level features.

def build_vgg19():
    """
    Load a pre-trained VGG19 model and return a model that outputs feature maps from a specific layer.
    We will use the output of 'block5_conv4' as the feature extractor.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    vgg.trainable = False  # Keep the VGG model frozen during training
    model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    return model

# Instantiate the VGG19 feature extractor
vgg = build_vgg19()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def vgg_loss(vgg, real_img, generated_img):
    """
    Compute VGG loss (perceptual loss) between real HR image and generated SR image.
    We use the feature maps from the pre-trained VGG19 model's 'block5_conv4' layer.
    """
    real_features = vgg(real_img)
    generated_features = vgg(generated_img)
    
    # Compute the MSE loss between the feature maps of real and generated images
    return tf.reduce_mean(tf.square(real_features - generated_features))

def generator_loss(vgg, real_img, sr_img, disc_output):
    """
    Generator loss consists of:
    1. VGG Loss (perceptual loss) between generated SR image and real HR image.
    2. Adversarial Loss from the discriminator.
    """
    # VGG Perceptual Loss
    perceptual_loss = vgg_loss(vgg, real_img, sr_img)

    # Adversarial Loss: Encourage the discriminator to classify the SR image as real
    adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_output), disc_output)
    
    pixel_loss = tf.reduce_mean(tf.abs(real_img - sr_img))  # L1 content loss
    
    
    total_gen_loss = 0.006 * pixel_loss + perceptual_loss + 1e-3 * adversarial_loss

    # Combine perceptual and adversarial loss (with a weighted factor for adversarial loss)
    total_gen_loss = perceptual_loss + 1e-3 * adversarial_loss  # Adversarial loss weighted lower than perceptual loss
    return total_gen_loss

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss consists of two parts:
    1. Real Loss: Binary Cross-Entropy loss on real HR images (should classify as 1).
    2. Fake Loss: Binary Cross-Entropy loss on generated SR images (should classify as 0).
    
    Remove the sigmoid activation from the last layer of the discriminator and keep from_logits=True in the loss function.
Set from_logits=False in the loss function, because the discriminator’s output already passes through a sigmoid.
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fake_output), fake_output)
    
    # Total Discriminator Loss
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss


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
    hr_image = hr_image / 127.5 - 1
    lr_image = lr_image / 127.5 - 1

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
'''
Every 5 epochs, we generate some sample SR images and save them for visual inspection.
For each sample:
We display the low-resolution image (input to the generator).
We display the super-resolved image (output of the generator).
We display the high-resolution image (ground truth).
'''

def generate_sample_images(epoch, lr_images, hr_images, num_samples=1):
    """
    Generate and save sample SR images during training.
    Saves `num_samples` low-res, super-res, and high-res images for visual comparison.
    """
    sr_images = generator(lr_images, training=False)

    for i in range(num_samples):
        lr_image = lr_images[i]
        sr_image = sr_images[i]
        hr_image = hr_images[i]

        # Rescale images to [0, 1] for display
        lr_image = (lr_image + 1.0) / 2.0
        sr_image = (sr_image + 1.0) / 2.0
        hr_image = (hr_image + 1.0) / 2.0

        # Save images in a folder
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot LR image
        axs[0].imshow(tf.clip_by_value(lr_image, 0.0, 1.0))
        axs[0].set_title('Low-Resolution')
        axs[0].axis('off')
        
        # Plot SR image
        axs[1].imshow(tf.clip_by_value(sr_image, 0.0, 1.0))
        axs[1].set_title('Super-Resolution')
        axs[1].axis('off')

        # Plot HR image
        axs[2].imshow(tf.clip_by_value(hr_image, 0.0, 1.0))
        axs[2].set_title('High-Resolution')
        axs[2].axis('off')

        # Save the figure
        sample_dir = 'H:/GPU/srgan thesis/12- srgansimpleunet/generated_samples'
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        plt.savefig(f"{sample_dir}/epoch_{epoch}_sample_{i + 1}.png")
        plt.close()
        
        
        
        
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




from tensorflow.keras.optimizers import Adam
'''
We log the generator and discriminator losses to TensorBoard using tf.summary.scalar().
We log the low-resolution, super-resolved, and high-resolution images to TensorBoard using tf.summary.image().
'''
# Set up TensorBoard writer
log_dir = 'H:/GPU/srgan thesis/12- srgansimpleunet/logs'
summary_writer = tf.summary.create_file_writer(log_dir)



# Define the optimizers for generator and discriminator
generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)
discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)

# Define the generator and discriminator models
generator = build_unet_srgan_generator()
discriminator = build_discriminator()

# Define a checkpoint directory to save model weights during training
checkpoint_dir = 'H:/GPU/srgan thesis/12- srgansimpleunet/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
# Use a CheckpointManager to load the latest checkpoint
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# Restore the latest checkpoint
checkpoint.restore(checkpoint_manager.latest_checkpoint)

if checkpoint_manager.latest_checkpoint:
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")
else:
    print("No checkpoint found, training from scratch.")'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# PSNR metric
def psnr_metric(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1.0)  # Assuming images are in [0, 1] range

# SSIM metric
def ssim_metric(y_true, y_pred):
    return ssim(y_true, y_pred, max_val=1.0)  # Assuming images are in [0, 1] range


# Custom CSV Logger callback
class CSVLoggerCallback(Callback):
    def __init__(self, csv_path):
        super(CSVLoggerCallback, self).__init__()
        self.csv_path = csv_path
        # Initialize a DataFrame to store the logs
        if os.path.exists(self.csv_path):
            self.logs_df = pd.read_csv(self.csv_path)
        else:
            self.logs_df = pd.DataFrame(columns=[
                'epoch', 'gen_loss', 'disc_loss', 'psnr', 'ssim'
            ])

    def on_epoch_end(self, epoch, logs=None):
        # Create a new DataFrame for the current epoch
        row = pd.DataFrame({
            'epoch': [epoch + 1],
            'gen_loss': [logs.get('gen_loss')],
            'disc_loss': [logs.get('disc_loss')],
            'psnr': [logs.get('psnr')],
            'ssim': [logs.get('ssim')]
        })
        # Concatenate the new row with the existing DataFrame
        self.logs_df = pd.concat([self.logs_df, row], ignore_index=True)
        # Save the DataFrame to the CSV file
        self.logs_df.to_csv(self.csv_path, index=False)
        print(f'Epoch {epoch + 1}: logs saved to {self.csv_path}')

# Path to the CSV file
csv_path = 'H:/GPU/srgan thesis/12- srgansimpleunet/training_logs4x.csv'
csv_logger_callback = CSVLoggerCallback(csv_path)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@tf.function
def train_step(lr_images, hr_images, vgg):
    """
    Performs a single training step, updating both the generator and discriminator.
    Includes VGG perceptual loss for the generator.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate super-resolved image from low-resolution input
        sr_images = generator(lr_images, training=True)

        # Discriminator output (for real and generated images)
        real_output = discriminator(hr_images, training=True)
        fake_output = discriminator(sr_images, training=True)

        # Compute generator loss (includes VGG loss and adversarial loss)
        gen_loss = generator_loss(vgg, hr_images, sr_images, fake_output)

        # Compute discriminator loss
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradients for both generator and discriminator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply the gradients to update the models
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Calculate PSNR and SSIM
    psnr_value = psnr_metric(hr_images, sr_images)
    ssim_value = ssim_metric(hr_images, sr_images)

    return gen_loss, disc_loss, sr_images, psnr_value, ssim_value  # Return the metrics for logging

def train(dataset, epochs, vgg):
    """
    Training loop to train the SRGAN model with VGG perceptual loss on the given dataset.
    """
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        avg_psnr = 0
        avg_ssim = 0

        # Loop through the dataset for each epoch
        for lr_images, hr_images in dataset:
            # Unpack 5 values from train_step
            gen_loss, disc_loss, sr_images, psnr_value, ssim_value = train_step(lr_images, hr_images, vgg)
            avg_psnr += tf.reduce_mean(psnr_value)
            avg_ssim += tf.reduce_mean(ssim_value)

        avg_psnr /= len(dataset)
        avg_ssim /= len(dataset)

        # Print losses and metrics
        print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}, '
              f'PSNR: {avg_psnr.numpy()}, SSIM: {avg_ssim.numpy()}')
        
        # Log losses and metrics to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('Generator Loss', gen_loss, step=epoch + 1)
            tf.summary.scalar('Discriminator Loss', disc_loss, step=epoch + 1)
            tf.summary.scalar('PSNR', avg_psnr, step=epoch + 1)
            tf.summary.scalar('SSIM', avg_ssim, step=epoch + 1)

        # Log metrics to CSV using the CSVLoggerCallback
        csv_logger_callback.on_epoch_end(epoch, logs={
            'gen_loss': gen_loss.numpy(),
            'disc_loss': disc_loss.numpy(),
            'psnr': avg_psnr.numpy(),
            'ssim': avg_ssim.numpy()
        })

        # Save model checkpoints every 250 epochs
        if (epoch + 1) % 300 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
        # Optionally generate some sample images every few epochs
        if (epoch + 1) % 10 == 0:
            generate_sample_images(epoch + 1, lr_images, hr_images)
            
            
            
       
        # Log sample images to TensorBoard every 5 epochs
        if (epoch + 1) % 5 == 0:
            with summary_writer.as_default():
                tf.summary.image('Low-Resolution Image', lr_images, max_outputs=5, step=epoch + 1)
                tf.summary.image('Super-Resolution Image', sr_images, max_outputs=5, step=epoch + 1)
                tf.summary.image('High-Resolution Image', hr_images, max_outputs=5, step=epoch + 1)
start_time = time.time()
# Example usage:
train(dataset, epochs=5000, vgg=vgg)
end_time = time.time()
save_path = 'H:/GPU/srgan thesis/12- srgansimpleunet/generator_model.h5'
generator.save(save_path)
print(f"Final generator model saved to {save_path}")
discriminator.save('H:/GPU/srgan thesis/12- srgansimpleunet/discriminator_model.h5')
epoch_duration = end_time - start_time
hours, remainder = divmod(epoch_duration, 3600)
minutes, seconds = divmod(remainder, 60)
print(f'training took {int(hours)} hours, {int(minutes)} minutes')
print(f'Training took {epoch_duration:.2f} seconds')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




import numpy as np



generator = tf.keras.models.load_model('H:/GPU/srgan thesis/12- srgansimpleunet/generator_model.h5')

# Function to evaluate/generate a super-resolved (SR) image
def evaluate_sr_image(generator, lr_image):
    sr_image = generator(lr_image, training=False)  # Set training=False for inference
    sr_image = (sr_image + 1.0) / 2.0  # Rescale to [0, 1] for display
    return sr_image

from tensorflow.keras.preprocessing.image import array_to_img

# Directory to save the generated high-resolution images
output_dir = 'H:/GPU/srgan thesis/12- srgansimpleunet/generated_hr_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_sr_images(generator, dataset, output_dir):
    """
    This function generates super-resolved images using the trained generator 
    and saves them to a specified directory.
    """
    # Iterate over the dataset
    for i, (lr_images, hr_images) in enumerate(dataset):
        # Generate super-resolved images
        sr_images = generator(lr_images, training=False)
        
        # Rescale the SR images back to [0, 255] for saving as images
        sr_images = (sr_images + 1.0) / 2.0  # Rescale to [0, 1]
        sr_images = np.clip(sr_images, 0, 1)  # Ensure pixel values are within [0, 1]
        
        # Convert and save each image
        for j in range(sr_images.shape[0]):
            sr_image = sr_images[j]
            sr_image = (sr_image * 255).astype(np.uint8)  # Convert to [0, 255]
            
            # Convert the NumPy array to an image
            img = array_to_img(sr_image)
            
            # Save the image to the output directory
            img.save(os.path.join(output_dir, f'sr_image_{i}_{j}.png'))
            
        print(f"Processed and saved batch {i + 1}")

# Load your dataset (use the same dataset loading function as before)
data_dir = "H:/GPU/srgan thesis/5-Srgan tensorflow/hr_train_LR"
batch_size = 1
hr_size = (256, 256)
dataset = load_dataset(data_dir, batch_size=batch_size, hr_size=hr_size)

# Call the function to process and save SR images
save_sr_images(generator, dataset, output_dir)





























