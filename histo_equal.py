import cv2
import os
import matplotlib.pyplot as plt

# Directories
input_dir = 'selected_dataset/'
output_dir = 'processed_images_hist_eq/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process images for histogram equalization
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Histogram Equalization
        hist_eq_image = cv2.equalizeHist(image)

        # Save the processed image
        cv2.imwrite(os.path.join(output_dir, f"hist_eq_{filename}"), hist_eq_image)

        # Plot and display
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Histogram Equalized")
        plt.imshow(hist_eq_image, cmap='gray')
        plt.show()

print(f"Histogram Equalized images saved in: {output_dir}")
