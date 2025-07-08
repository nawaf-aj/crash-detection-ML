from ultralytics import YOLO

if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO('best.pt')  # Change to your model weights if needed

    # Train the model
    model.train(
        data='datasets/data.yaml',        # Path to your data.yaml file
        epochs=150,  
        patience = 50,             # Number of epochs
        batch=16,                # Batch size
        imgsz=640,               # Image size
        device='cuda',                # Use GPU; set to 'cpu' for CPU training
                   # Set workers to 0 for Windows
        optimizer='Adam',        # Optimizer
        val=True,                # Perform validation
        amp=True                 # Automatic Mixed Precision
    )