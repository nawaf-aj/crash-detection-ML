import os

# Directory containing labels
labels_dir = "./datasets/train/labels"  # Repeat for valid/test as needed

# Clean labels to only include detection boxes
for root, _, files in os.walk(labels_dir):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Filter out segmentation labels (if any)
            clean_lines = [line for line in lines if len(line.split()) == 5]

            with open(file_path, 'w') as file:
                file.writelines(clean_lines)

print("Cleaned labels to include only detection boxes.")
