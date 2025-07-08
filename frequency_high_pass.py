import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Directories
input_dir = 'selected_dataset/'
output_dir = 'processed_images_highpass/'

os.makedirs(output_dir, exist_ok=True)

# Function to apply a high-pass filter
def high_pass_filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency spectrum

    # Create a mask that blocks low frequencies
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # Apply Fourier Transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Apply mask and inverse transform
    filtered_dft = dft_shift * mask
    filtered_dft_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.abs(np.fft.ifft2(filtered_dft_shift))

    return filtered_image

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply high-pass filter
        sharpened_image = high_pass_filter(image)

        # Save and display results
        cv2.imwrite(os.path.join(output_dir, f"highpass_{filename}"), sharpened_image)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("High-Pass Filtered")
        plt.imshow(sharpened_image, cmap='gray')
        plt.show()

print(f"High-pass filtered images saved in: {output_dir}")
