import os
import shutil

# Paths
original_labels_dir = "datasets/train/labels/"
augmented_images_dir = "datasets/augmented/images/"
augmented_labels_dir = "datasets/augmented/labels/"

# Create labels folder if not exists
os.makedirs(augmented_labels_dir, exist_ok=True)

for filename in os.listdir(augmented_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Extract original base name without '_aug'
        base_name = filename.rsplit('_aug', 1)[0]
        original_label_path = os.path.join(original_labels_dir, base_name + ".txt")
        augmented_label_path = os.path.join(augmented_labels_dir, filename.rsplit('.', 1)[0] + ".txt")
        
        # Copy the original label
        if os.path.exists(original_label_path):
            shutil.copyfile(original_label_path, augmented_label_path)
            print(f"Created: {augmented_label_path}")
        else:
            print(f"Warning: No label found for {filename}")
