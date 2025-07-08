import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Directories
input_dir = 'selected_dataset/'
output_dir = 'processed_images_lowpass/'

os.makedirs(output_dir, exist_ok=True)

# Function to apply a low-pass filter
def low_pass_filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency spectrum

    # Create a mask with low frequencies at the center
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

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

        # Apply low-pass filter
        smoothed_image = low_pass_filter(image)

        # Save and display results
        cv2.imwrite(os.path.join(output_dir, f"lowpass_{filename}"), smoothed_image)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Low-Pass Filtered")
        plt.imshow(smoothed_image, cmap='gray')
        plt.show()

print(f"Low-pass filtered images saved in: {output_dir}")
