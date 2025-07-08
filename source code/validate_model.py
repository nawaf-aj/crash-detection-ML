from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO("best.pt")

# Perform validation
print("Starting validation...")
model.val(
    data="datasets/data.yaml",  # Path to your dataset YAML file
    conf=0.25,                  # Confidence threshold for validation
    iou=0.5                     # IOU threshold for validation
)
print("Validation complete!")
