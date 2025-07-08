import os

# Path to augmented images directory
augmented_images_dir = "datasets/augmented/images/"

for filename in os.listdir(augmented_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        base_name, ext = os.path.splitext(filename)
        new_name = base_name + "_aug" + ext
        os.rename(
            os.path.join(augmented_images_dir, filename),
            os.path.join(augmented_images_dir, new_name),
        )
        print(f"Renamed: {filename} -> {new_name}")
