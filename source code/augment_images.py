import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),  # Random shifting/scaling/rotation
    A.MotionBlur(p=0.2),  # Add motion blur
    A.RandomSizedCrop(min_max_height=(400, 600), height=640, width=640, p=0.2),  # Random cropping and resizing
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize the image
])

# Augment images
def augment_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}. Skipping.")
                continue
            augmented = transform(image=image)["image"]
            # Convert the image back to uint8 format for saving
            augmented = (augmented * 255).astype(np.uint8)
            augmented_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(augmented_image_path, augmented)

# Paths for augmentation
input_folder = "datasets/train/images"  # Original training images
output_folder = "datasets/augmented/images"  # Augmented images

# Run augmentation
augment_images(input_folder, output_folder)

print("Augmentation complete! Augmented images saved to", output_folder)
