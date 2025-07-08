import os
import random

def count_class_annotations(annotation_dir):
    """
    Counts the number of annotations for each class in the dataset.

    Args:
    - annotation_dir: Path to directory containing YOLO annotation files.

    Returns:
    - Dictionary with class IDs as keys and their counts as values.
    """
    class_counts = {}
    for root, _, files in os.walk(annotation_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                with open(os.path.join(root, file_name), 'r') as file:
                    for line in file:
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return class_counts

def downsample_class(annotation_dir, target_class, target_ratio, base_class_count):
    """
    Downsamples a specific class in YOLO annotation files to a target ratio.

    Args:
    - annotation_dir: Path to directory containing YOLO annotation files.
    - target_class: The class ID to downsample.
    - target_ratio: Desired ratio of target class to base class.
    - base_class_count: Number of instances in the base class to calculate the target count.
    """
    target_count = base_class_count * target_ratio
    current_count = 0
    for root, _, files in os.walk(annotation_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                new_lines = []
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id == target_class:
                        if current_count < target_count:
                            new_lines.append(line)
                            current_count += 1
                    else:
                        new_lines.append(line)

                with open(file_path, 'w') as file:
                    file.writelines(new_lines)

                print(f"Processed {file_path}")

    print(f"Downsampled {target_class} to {current_count} instances.")

# Paths
train_labels_dir = "./datasets/train/labels"

# Step 1: Count the current distribution of classes
class_counts = count_class_annotations(train_labels_dir)
print(f"Class counts before downsampling: {class_counts}")

# Step 2: Downsample Car/Vehicles (class 1) to achieve a 3:1 ratio with Accidents (class 0)
base_class_count = class_counts.get(0, 0)  # Number of Accidents
target_ratio = 3  # Desired ratio of Car/Vehicles to Accidents
downsample_class(train_labels_dir, target_class=1, target_ratio=target_ratio, base_class_count=base_class_count)

# Step 3: Recount to verify
class_counts_after = count_class_annotations(train_labels_dir)
print(f"Class counts after downsampling: {class_counts_after}")
