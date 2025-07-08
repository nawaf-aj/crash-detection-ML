import os
from collections import Counter

def count_classes(annotation_dir):
    """
    Counts the number of annotations per class in a YOLO dataset.

    Args:
    - annotation_dir: Path to the directory containing YOLO annotation files.

    Returns:
    - Counter object with counts per class.
    """
    class_counts = Counter()
    for root, _, files in os.walk(annotation_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                with open(os.path.join(root, file_name), 'r') as file:
                    for line in file:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    return class_counts

# Directories to check
annotation_dirs = [
    "./datasets/train/labels",
    "./datasets/valid/labels",
    "./datasets/test/labels"
]

for annotation_dir in annotation_dirs:
    counts = count_classes(annotation_dir)
    print(f"Class distribution in {annotation_dir}: {dict(counts)}")
