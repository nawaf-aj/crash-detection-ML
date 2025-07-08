import os

images_dir = "datasets/train/images/"
labels_dir = "datasets/train/labels/"

missing_labels = []

for image in os.listdir(images_dir):
    if image.endswith(".jpg") or image.endswith(".png"):
        label_path = os.path.join(labels_dir, image.rsplit('.', 1)[0] + ".txt")
        if not os.path.exists(label_path):
            missing_labels.append(image)

if missing_labels:
    print("Missing labels for the following images:")
    for missing in missing_labels:
        print(missing)
else:
    print("All images have corresponding labels!")
