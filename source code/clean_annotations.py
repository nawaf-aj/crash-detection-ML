import os

# Configuration
labels_dir = "./datasets"  # Root directory containing train/valid/test label directories
mapping = {4: 0, 6: 1}  # Map class 4 to 0 and class 6 to 1

# Loop through all datasets (train, valid, test)
for subset in ["train", "valid", "test"]:
    label_path = os.path.join(labels_dir, subset, "labels")
    if not os.path.exists(label_path):
        print(f"Labels directory not found: {label_path}")
        continue

    print(f"Processing labels in {label_path}...")

    # Loop through all label files
    for root, _, files in os.walk(label_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Rewrite labels with the new mapping
                remapped_lines = []
                for line in lines:
                    components = line.strip().split()
                    class_id = int(components[0])
                    if class_id in mapping:
                        components[0] = str(mapping[class_id])  # Update class ID
                        remapped_lines.append(" ".join(components) + "\n")

                # Write the updated lines back to the file
                with open(file_path, 'w') as file:
                    file.writelines(remapped_lines)

print("All labels remapped successfully!")
