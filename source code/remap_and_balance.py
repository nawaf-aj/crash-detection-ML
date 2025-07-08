import os
import random

# Classes to keep
keep_classes = {4, 6}  # 4 = car, 6 = car_car_accident
target_car_count = 11000  # Target count for the car class (4)

# Paths to annotation directories
annotation_dirs = [
    "./datasets/train/labels",
    "./datasets/valid/labels",
    "./datasets/test/labels"
]

# Process annotations
for annotation_dir in annotation_dirs:
    for root, _, files in os.walk(annotation_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Filter lines to keep only the required classes
                filtered_lines = []
                car_lines = []  # Store lines for class 4 (car) separately
                for line in lines:
                    components = line.strip().split()
                    class_id = int(components[0])
                    if class_id == 4:  # car
                        car_lines.append(line)
                    elif class_id in keep_classes:
                        filtered_lines.append(line)

                # Reduce the car class to the target count
                if len(car_lines) > target_car_count:
                    car_lines = random.sample(car_lines, target_car_count)

                # Combine filtered lines and reduced car lines
                filtered_lines.extend(car_lines)

                # Write the filtered lines back to the file
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)

print("Dataset cleaned and balanced!")
