from ultralytics import YOLO

if __name__ == "__main__":
    # Load the YOLO model with your trained weights
    model = YOLO('best.pt')  # Path to your trained weights

    # Fine-tune the model
    model.train(
        data='datasets/data.yaml',        # Path to your updated data.yaml file
        epochs=50,               # Number of fine-tuning epochs (adjust as needed)
        batch=16,                # Batch size
        imgsz=640,               # Image size
        device='cuda',                # Use GPU; set to 'cpu' for CPU training   # Number of workers (reduce to 0 for Windows issues)
        optimizer='Adam',        # Use Adam optimizer for smoother fine-tuning
        lr0=0.001,               # Lower learning rate for fine-tuning
        name='fine_tunee',        # Name of the fine-tuning run
        project='runs/fine_tunee',# Directory to save fine-tuning results
        val=True,                # Perform validation after every epoch
        pretrained=True,         # Continue from pre-trained weights
        verbose=True,            # Print detailed logs
        amp=True                 # Enable Automatic Mixed Precision for faster training
    )
