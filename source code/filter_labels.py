import os
import random

# Configuration
train_labels_dir = "./datasets/train/labels"  # Training labels directory
car_class_id = 4  # Class ID for 'car'
target_car_count = 9000  # Desired number of car labels

# Step 1: Collect all car lines from all label files
total_car_lines = []
label_files = []

for root, _, files in os.walk(train_labels_dir):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            label_files.append(file_path)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Separate car lines and store with the file path
            for line in lines:
                if int(line.split()[0]) == car_class_id:
                    total_car_lines.append((file_path, line))

# Step 2: Randomly sample the car lines to reduce them to the target count
if len(total_car_lines) > target_car_count:
    sampled_car_lines = random.sample(total_car_lines, target_car_count)
else:
    sampled_car_lines = total_car_lines

# Step 3: Rewrite all label files
for file_path in label_files:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out car lines and add back the sampled car lines for this file
    filtered_lines = [line for line in lines if int(line.split()[0]) != car_class_id]
    filtered_lines += [line[1] for line in sampled_car_lines if line[0] == file_path]

    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

print("Car class reduced to 9,000 labels!")
