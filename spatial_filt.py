import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Directories
input_dir = 'selected_dataset/'
output_dir = 'processed_images_spat_filters/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process images for smoothing and sharpening
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Smoothing (Gaussian Blur)
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Sharpening (Using Kernel)
        sharpening_kernel = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

        # Save the processed images
        cv2.imwrite(os.path.join(output_dir, f"smoothed_{filename}"), smoothed_image)
        cv2.imwrite(os.path.join(output_dir, f"sharpened_{filename}"), sharpened_image)

        # Plot and display
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("Smoothed")
        plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 3)
        plt.title("Sharpened")
        plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
        plt.show()

print(f"Filtered images (Smoothed and Sharpened) saved in: {output_dir}")
