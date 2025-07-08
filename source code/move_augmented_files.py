import shutil
import os

# Paths
augmented_images_dir = "datasets/augmented/images/"
augmented_labels_dir = "datasets/augmented/labels/"
train_images_dir = "datasets/train/images/"
train_labels_dir = "datasets/train/labels/"

# Move images
for img in os.listdir(augmented_images_dir):
    shutil.move(os.path.join(augmented_images_dir, img), os.path.join(train_images_dir, img))

# Move labels
for lbl in os.listdir(augmented_labels_dir):
    shutil.move(os.path.join(augmented_labels_dir, lbl), os.path.join(train_labels_dir, lbl))

print("Augmented images and labels moved to train folder.")
