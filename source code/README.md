# Car Crash Detection Using YOLO

## 📌 Project Overview
This project implements a **Car Crash Detection System** using the **YOLO (You Only Look Once) object detection model**. The model is trained to classify and detect car crashes in real-time, aiming to improve road safety by identifying accident scenarios efficiently.

## 🚀 Features
- **Real-time crash detection** from video feeds or images.
- **YOLOv8 model** for fast and accurate object detection.
- **Custom dataset training** using labeled images of car crashes and normal road conditions.
- **Deployment-ready** for integration into smart surveillance or traffic monitoring systems.

## 📂 Dataset
The model is trained using a custom dataset that includes:
- **Crash (Class 0)**: Images containing car accidents.
- **Non-Crash (Class 1)**: Normal road conditions.
- **Filtered Class**: Removed any unnecessary class labels.

We used the **Berkeley DeepDrive (BDD100K)** dataset as a data source and converted the annotations into YOLO format.

## 🔧 Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/car-crash-detection.git
cd car-crash-detection
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
Ensure you have **CUDA** enabled for GPU training.

### **3. Download the Dataset**
- Preprocessed dataset is stored in `datasets/`.
- If needed, download BDD100K and convert annotations to YOLO format.

### **4. Train the YOLO Model**
```bash
python train.py --img 640 --batch 16 --epochs 50 --data config.yaml --weights yolov8n.pt
```
- `--img 640`: Image resolution.
- `--batch 16`: Batch size.
- `--epochs 50`: Number of training epochs.
- `--data config.yaml`: Path to dataset configuration.
- `--weights yolov8n.pt`: Using YOLOv8 pre-trained weights.

### **5. Testing & Inference**
To test on a sample image:
```bash
python detect.py --weights best.pt --source test_image.jpg
```
For real-time detection via webcam:
```bash
python detect.py --weights best.pt --source 0
```

## 📊 Model Performance
- **mAP@50:** XX% (To be updated after training)
- **Precision:** XX%
- **Recall:** XX%

## 📌 Future Improvements
- Fine-tune the model with a larger dataset.
- Deploy as an API for real-time crash detection.
- Improve accuracy using advanced feature extraction.

## 🤝 Contributing
Feel free to fork this repository and improve the project. Pull requests are welcome!

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For any inquiries, reach out via GitHub or email: your.email@example.com

---
🚀 **Let's make roads safer with AI!**

