import os
from PIL import Image

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def resize_images(input_folder, output_folder, size=(64, 64)):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"The output folder '{output_folder}' does not exist.")
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        # Construct full file path
        input_path = os.path.join(input_folder, file)
        
        # Check if it is a file
        if os.path.isfile(input_path):
            try:
                # Open an image file
                with Image.open(input_path) as img:
                    # Resize image
                    img = img.resize(size, Image.ANTIALIAS)
                    
                    # Construct the output file path
                    output_path = os.path.join(output_folder, file)
                    
                    # Save the resized image to the output folder
                    img.save(output_path)
                    
                    print(f"Resized and saved {file} to {output_path}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Example usage
input_folder = 'C:/SRGAN_CustomDataset-uav/SRGAN_CustomDataset-main/custom_dataset/hr_valid_LR'
output_folder = 'C:/SRGAN_CustomDataset-uav/SRGAN_CustomDataset-main/custom_dataset/64'
resize_images(input_folder, output_folder)
