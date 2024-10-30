import tensorflow as tf
from tensorflow.keras import layers ,regularizers
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.image import psnr, ssim
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def rrdb_block(x, filters=64, growth_channels=32, num_dense_layers=3):
    """
    Creates a Residual-in-Residual Dense Block (RRDB).
    """
    def dense_block(x, growth_channels):
        for _ in range(num_dense_layers):
            dense_layer = layers.Conv2D(growth_channels, (3, 3), padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
            dense_layer = layers.LeakyReLU(alpha=0.2)(dense_layer)
            x = layers.Concatenate()([x, dense_layer])
        return x

    # Create a dense block and add a convolution layer for feature compression
    dense_output = dense_block(x, growth_channels)
    compressed_output = layers.Conv2D(filters, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dense_output)
    scaled_output = layers.Lambda(lambda x: x * 0.2)(compressed_output)
    
    # Residual scaling and add back the skip connection
    return layers.Add()([x, scaled_output])

def build_generator_esrgan(num_rrdb_blocks=16, input_shape=(64, 64, 3)):
    input_lr = layers.Input(shape=input_shape)
    
    # Initial Convolution + PReLU
    x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_lr)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    
    # Save a copy of this feature map for the skip connection
    initial_feature_map = x

    # Add multiple RRDB blocks
    for _ in range(num_rrdb_blocks):
        x = rrdb_block(x)

    # Post-RRDB Convolution
    x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    # Skip connection: Add the initial feature map back to x
    x = layers.Add()([x, initial_feature_map])

    # Upsampling (Pixel Shuffle)
    for _ in range(2):  # Two stages of upsampling (4x scale)
        x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
    
    # Final Output Layer (SR Image)
    output_sr = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(x)
    
    generator = tf.keras.models.Model(input_lr, output_sr)
    return generator

esrgan_generator = build_generator_esrgan(input_shape=(64, 64, 3))
esrgan_generator.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def build_esrgan_discriminator(input_shape=(256, 256, 3)):
    def conv_block(x, filters, size, strides):
        x = layers.Conv2D(filters, size, strides=strides, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    input_image = layers.Input(shape=input_shape)
    x = conv_block(input_image, 64, (3, 3), strides=1)
    
    # Add more convolutional layers with increasing filters
    for filters in [64, 128, 128, 256, 256, 512, 512]:
        strides = 2 if filters % 128 == 0 else 1
        x = conv_block(x, filters, (3, 3), strides=strides)

    # Flatten and add fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1)(x)

    return tf.keras.models.Model(input_image, x)

# Build the ESRGAN discriminator model
esrgan_discriminator = build_esrgan_discriminator(input_shape=(256, 256, 3))
esrgan_discriminator.summary()



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


def build_vgg19_for_esrgan():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    vgg.trainable = False
    model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv2').output)
    return model


# Instantiate the VGG19 feature extractor
vgg = build_vgg19_for_esrgan()
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

def relativistic_discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(real_output) - tf.reduce_mean(fake_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(fake_output) - tf.reduce_mean(real_output), fake_output)
    return real_loss + fake_loss

def esrgan_generator_loss(vgg, real_img, sr_img, fake_output, perceptual_weight=1e-2):
    perceptual_loss = vgg_loss(vgg, real_img, sr_img)
    adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)
    pixel_loss = tf.reduce_mean(tf.abs(real_img - sr_img))
    total_gen_loss = perceptual_loss + adversarial_loss * 0.001 + pixel_loss * perceptual_weight
    return total_gen_loss



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
        sample_dir = 'H:/GPU/srgan thesis/10-ESRGAN/generated_samples'
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
log_dir = 'H:/GPU/srgan thesis/10-ESRGAN/logs'
summary_writer = tf.summary.create_file_writer(log_dir)


generator_optimizer = Adam(learning_rate=1e-6, beta_1=0.9)
discriminator_optimizer = Adam(learning_rate=1e-6, beta_1=0.9)
'''
# Define the optimizers for generator and discriminator
generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)
discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)
'''
# Define the generator and discriminator models
generator = build_generator_esrgan()
discriminator = build_esrgan_discriminator()

# Define a checkpoint directory to save model weights during training
checkpoint_dir = 'H:/GPU/srgan thesis/10-ESRGAN/training_checkpoints'
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
csv_path = 'H:/GPU/srgan thesis/10-ESRGAN/training_logs4x.csv'
csv_logger_callback = CSVLoggerCallback(csv_path)





def lr_schedule(epoch, lr):
    decay_rate = 0.95
    decay_step = 100
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
'''
def lr_schedule(epoch, lr):
    if epoch % 100 == 0 and epoch:
        lr = lr * 0.5  # Halve the learning rate every 100 epochs
    return lr
'''
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
train_callbacks = [lr_callback, csv_logger_callback]


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
        gen_loss = esrgan_generator_loss(vgg, hr_images, sr_images, fake_output)

        # Compute discriminator loss
        disc_loss = relativistic_discriminator_loss(real_output, fake_output)

    # Compute gradients for both generator and discriminator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_generator = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients_of_generator]
    gradients_of_discriminator = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients_of_discriminator]

    # Apply the gradients to update the models
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Calculate PSNR and SSIM
    psnr_value = psnr_metric(hr_images, sr_images)
    ssim_value = ssim_metric(hr_images, sr_images)

    return gen_loss, disc_loss, sr_images, psnr_value, ssim_value  # Return the metrics for logging

def train(dataset, epochs, vgg,callbacks=[]):
    """
    Training loop to train the SRGAN model with VGG perceptual loss on the given dataset.
    """
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        avg_psnr = 0
        avg_ssim = 0
        #new_lr = lr_schedule(epoch, generator_optimizer.learning_rate.numpy())
        #generator_optimizer.learning_rate.assign(new_lr)
        #discriminator_optimizer.learning_rate.assign(new_lr)

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

# Example usage:
#train(dataset, epochs=5000, vgg=vgg)
train(dataset, epochs=15000, vgg=vgg, callbacks=train_callbacks)


save_path = 'H:/GPU/srgan thesis/10-ESRGAN/generator_model.h5'
generator.save(save_path)
print(f"Final generator model saved to {save_path}")
discriminator.save('H:/GPU/srgan thesis/10-ESRGAN/discriminator_model.h5')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
matplotlib.use('TkAgg')  # Use TkAgg, which supports interactivity
def evaluate(generator, lr_image):
    """
    Generates a super-resolved (SR) image from a low-resolution (LR) input image using the trained generator.
    """
    sr_image = generator(lr_image, training=False)  # Set training=False for inference
    sr_image = (sr_image + 1.0) / 2.0  # Rescale to [0, 1]
    return sr_image

# Example of generating an SR image
for lr_images, hr_images in dataset.take(2):  # Assume we use the same dataset for simplicity
    sr_image = evaluate(generator, lr_images[0:1])  # Evaluate on the first LR image
    plt.imshow(tf.clip_by_value(sr_image[0], 0.0, 1.0))
    plt.title('Super-Resolved Image')
    plt.axis('off')
    plt.show()





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Save generator and discriminator models
generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')

# To load the models back for inference or further training
generator = tf.keras.models.load_model('generator_model.h5')
discriminator = tf.keras.models.load_model('discriminator_model.h5')




































