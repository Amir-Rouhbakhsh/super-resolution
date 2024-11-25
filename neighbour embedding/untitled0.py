import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import os

def load_images_from_folder(folder):
    """
    Load images from a specified folder.
    
    :param folder: Path to the folder containing images.
    :return: List of images loaded as numpy arrays.
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def extract_patches(image, patch_size, stride):
    """
    Extracts overlapping patches from the input image.
    
    :param image: Input image (2D).
    :param patch_size: Size of each patch (tuple).
    :param stride: Stride between patches.
    :return: Array of patches.
    """
    patches = []
    for y in range(0, image.shape[0] - patch_size[0] + 1, stride):
        for x in range(0, image.shape[1] - patch_size[1] + 1, stride):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch.flatten())
    return np.array(patches)

def build_hr_patch_database(hr_images, patch_size, stride):
    """
    Build a database of high-resolution patches from the HR images.
    
    :param hr_images: List of high-resolution images.
    :param patch_size: Size of each patch.
    :param stride: Stride between patches.
    :return: Array of high-resolution patches.
    """
    hr_patches = []
    for hr_image in hr_images:
        patches = extract_patches(hr_image, patch_size, stride)
        hr_patches.append(patches)
    return np.vstack(hr_patches)

def reconstruct_image(patches, image_size, patch_size, stride):
    """
    Reconstructs the image from patches.
    
    :param patches: Array of patches.
    :param image_size: Size of the output image.
    :param patch_size: Size of each patch.
    :param stride: Stride between patches.
    :return: Reconstructed image.
    """
    image = np.zeros(image_size)
    weight = np.zeros(image_size)
    patch_idx = 0
    for y in range(0, image_size[0] - patch_size[0] + 1, stride):
        for x in range(0, image_size[1] - patch_size[1] + 1, stride):
            patch = patches[patch_idx].reshape(patch_size)
            image[y:y + patch_size[0], x:x + patch_size[1]] += patch
            weight[y:y + patch_size[0], x:x + patch_size[1]] += 1
            patch_idx += 1
    return image / weight

def neighbor_embedding_sr(lr_image, hr_patches, k=5, patch_size=(5, 5), stride=1):
    """
    Perform Neighbor Embedding Super-Resolution on the given low-resolution image.
    
    :param lr_image: Low-resolution input image.
    :param hr_patches: Array of high-resolution patches from training data.
    :param k: Number of nearest neighbors.
    :param patch_size: Size of each patch.
    :param stride: Stride for patch extraction.
    :return: Super-resolved high-resolution image.
    """
    lr_patches = extract_patches(lr_image, patch_size, stride)
    
    # Find K nearest neighbors using Euclidean distance
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(hr_patches)
    distances, indices = nbrs.kneighbors(lr_patches)
    
    # Reconstruct HR patches by weighted sum of nearest neighbors
    hr_reconstructed_patches = np.zeros_like(lr_patches)
    for i in range(lr_patches.shape[0]):
        weights = np.exp(-distances[i]**2 / (2 * np.var(distances[i])))
        weights /= np.sum(weights)
        hr_reconstructed_patches[i] = np.dot(weights, hr_patches[indices[i]])
    
    # Reconstruct the high-resolution image from patches
    hr_image_size = (lr_image.shape[0] * 2, lr_image.shape[1] * 2)  # Assuming 2x upscaling
    hr_image = reconstruct_image(hr_reconstructed_patches, hr_image_size, patch_size, stride)
    
    return hr_image

# Example Usage
# Define paths to your HR and LR image folders
hr_image_folder =  'H:/GPU/neighbouring embedding/hr_train_LR'
lr_image_folder =  'H:/GPU/neighbouring embedding/lr6464'

# Load HR and LR images
hr_images = load_images_from_folder(hr_image_folder)
lr_images = load_images_from_folder(lr_image_folder)

# Parameters
patch_size = (5, 5)
stride = 1
k = 5

# Build the HR patch database from the training HR images
hr_patches = build_hr_patch_database(hr_images, patch_size, stride)

# Super-resolve each LR image
for i, lr_image in enumerate(lr_images):
    sr_image = neighbor_embedding_sr(lr_image, hr_patches, k=k, patch_size=patch_size, stride=stride)
    cv2.imwrite(f'super_resolved_image_{i}.png', sr_image)

print("Super-resolution complete. Check the output images.")