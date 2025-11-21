# 🦅 Aerial Object Classification & Detection

A comprehensive machine learning solution for identifying and classifying aerial objects, specifically distinguishing between **birds** and **drones** in aerial imagery using both classification and object detection approaches.

## 📋 Project Overview

This project implements a dual-approach system combining:
- **Image Classification**: Binary classification using MobileNetV2 and Custom CNN
- **Object Detection**: Precise localization using YOLOv8 with bounding box annotations

The system provides accurate and reliable aerial object recognition for practical surveillance and monitoring applications.

## 🎯 Key Features

- **Multiple Model Architectures**: Custom CNN, MobileNetV2 Transfer Learning, and YOLOv8
- **Comprehensive Training Pipeline**: From scratch training and fine-tuning strategies  
- **Web Application**: Professional Streamlit interface for real-time predictions
- **Interactive Notebooks**: Detailed experimentation and analysis workflows
- **Production Ready**: Optimized models for deployment scenarios

## 📁 Project Structure

```
├── data/                          # Dataset storage
│   ├── classification_dataset/    # Image classification data (train/valid/test)
│   │   ├── train/
│   │   │   ├── bird/             # Bird training images
│   │   │   └── drone/            # Drone training images
│   │   ├── valid/                # Validation split
│   │   └── test/                 # Test split
│   └── object_detection_Dataset/  # YOLO format object detection data
│       ├── train/
│       │   ├── images/           # Training images
│       │   └── labels/           # YOLO format annotations
│       ├── valid/                # Validation data
│       ├── test/                 # Test data
│       └── data.yaml             # YOLO dataset configuration
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── dataset_loader.py     # PyTorch DataLoader utilities
│   │   └── transfomer.py         # Image preprocessing transforms
│   ├── models/                   # Model architectures
│   │   ├── custom_cnn.py         # Custom CNN implementation
│   │   ├── mobilenet_cnn.py      # MobileNetV2 model utilities
│   │   └── inference.py          # Inference pipeline
│   ├── train/                    # Training and evaluation
│   │   ├── training.py           # Training orchestration
│   │   └── eval.py               # Model evaluation utilities
│   ├── utils/                    # Utility functions
│   │   └── train_model.py        # Training loop utilities
│   └── yolo.py                   # YOLO training script
├── notebooks/                    # Jupyter notebooks
│   ├── experiments.ipynb         # Complete classification pipeline
│   └── yolo_model.ipynb          # YOLO object detection workflow
├── models/                       # Trained model weights
│   ├── best_classifier.pt        # Best performing classification model
│   ├── mobilenet_phase1.pt       # Phase 1 transfer learning checkpoint
│   └── custom_cnn.pt             # Custom CNN model weights
├── streamlit_app/               # Web application
│   └── app.py                    # Streamlit web interface
├── documentation.md              # Comprehensive project documentation
├── streamlit_app_documentation.md # Web app technical documentation
└── requirements.txt              # Python dependencies
```

## 📊 Dataset Information

### Classification Dataset
- **Training**: 2,662 images (1,414 birds, 1,248 drones)
- **Validation**: 442 images (217 birds, 225 drones)  
- **Test**: 215 images (121 birds, 94 drones)
- **Format**: Standard ImageFolder structure

### Object Detection Dataset  
- **Training**: 2,662 images with YOLO format annotations
- **Test**: 224 images
- **Classes**: 2 (Bird=0, Drone=1)
- **Format**: YOLO bounding box annotations

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
CUDA-capable GPU (optional, for faster training)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/viraj-gavade/Aerial-Object-Classification---Detection.git
   cd "Aerial Object Classification & Detection"
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install ultralytics
   pip install streamlit plotly
   pip install scikit-learn matplotlib pillow seaborn
   pip install tqdm
   ```

3. **Prepare your dataset**
   - Place your aerial images in the appropriate data directories
   - Ensure proper folder structure for classification dataset
   - Verify YOLO annotation format for object detection

## 💻 Usage

### 1. Training Models

#### Classification Models
```python
from src.train.training import train_mobilenet

# Train MobileNetV2 with transfer learning
train_mobilenet(
    train_dir="data/classification_dataset/train",
    val_dir="data/classification_dataset/valid", 
    test_dir="data/classification_dataset/test",
    epochs=10,
    batch_size=32,
    model_save_path="models/mobilenet_classifier.pt"
)
```

#### YOLO Object Detection
```python
from ultralytics import YOLO

# Train YOLOv8 for object detection
model = YOLO("yolov8n.pt")
model.train(
    data="data/object_detection_Dataset/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16
)
```

### 2. Running the Web Application

```bash
# Launch Streamlit web interface
cd streamlit_app
streamlit run app.py
```

The web application provides:
- Real-time image classification
- Interactive confidence visualization
- Professional UI with detailed results
- Support for JPG, JPEG, PNG formats

### 3. Model Evaluation

```python
from src.train.eval import evaluate_model

# Evaluate trained model performance
evaluate_model(
    model_path="models/best_classifier.pt",
    test_dir="data/classification_dataset/test"
)
```

### 4. Making Predictions

```python
from src.models.inference import load_image, predict
from src.models.mobilenet_cnn import load_mobilenet
import torch

# Load model
model = load_mobilenet(pretrained=False)
model.load_state_dict(torch.load("models/best_classifier.pt"))
model.eval()

# Predict on new image
img_tensor = load_image("path/to/aerial_image.jpg")
prediction = predict(model, img_tensor, device="cpu")
print(f"Predicted: {['bird', 'drone'][prediction]}")
```

## 🧠 Model Architectures

### 1. MobileNetV2 Transfer Learning
- **Backbone**: Pre-trained MobileNetV2 on ImageNet
- **Input Size**: 224×224×3 RGB images  
- **Architecture**: Efficient depthwise separable convolutions
- **Training**: Two-phase approach (frozen → fine-tuned)
- **Output**: 2 classes (Bird, Drone)

### 2. Custom CNN
- **Design**: Lightweight architecture for aerial objects
- **Layers**: 3 convolutional blocks (32→64→128 channels)
- **Features**: BatchNorm, ReLU activation, MaxPooling
- **Classifier**: Fully connected with dropout (0.3)
- **Purpose**: Baseline comparison and resource-constrained deployment

### 3. YOLOv8 Object Detection  
- **Model**: YOLOv8n (nano version for efficiency)
- **Input**: 640×640 pixel images
- **Output**: Bounding boxes + class predictions
- **Training**: 40 epochs with data augmentation
- **Performance**: Real-time inference capability

## 🔧 Technical Implementation

### Data Preprocessing
```python
# Training transforms with augmentation
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/test transforms  
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Training Strategy
- **Transfer Learning**: Start with ImageNet pre-trained weights
- **Phase 1**: Freeze backbone, train classifier head (5 epochs)
- **Phase 2**: Selective unfreezing, fine-tune last layers (8 epochs)
- **Optimization**: Adam optimizer with adaptive learning rates
- **Validation**: Hold-out validation for model selection

### Model Performance
- **Accuracy**: High classification accuracy on test set
- **Efficiency**: Optimized for real-time inference  
- **Robustness**: Handles various lighting and angle conditions
- **Deployment**: CPU-compatible for broad deployment

## 🌐 Web Application

The Streamlit web application features:

- **Professional Interface**: Modern gradient design with responsive layout
- **Real-time Processing**: Instant predictions on uploaded images
- **Interactive Visualizations**: Plotly charts showing confidence scores
- **Detailed Analysis**: Image metadata and prediction confidence
- **User-friendly**: Intuitive workflow from upload to results

### Application Features
- Upload support for multiple image formats
- Real-time prediction with confidence scoring
- Interactive bar charts for class probabilities  
- Professional styling with custom CSS
- Responsive design for different screen sizes

## 📊 Jupyter Notebooks

### experiments.ipynb
Comprehensive classification pipeline including:
- **Data Exploration**: Dataset analysis and visualization
- **Custom CNN Training**: From-scratch model development
- **Transfer Learning**: MobileNetV2 implementation  
- **Fine-tuning**: Advanced optimization techniques
- **Evaluation**: Detailed performance analysis with metrics

### yolo_model.ipynb  
Complete object detection workflow:
- **Data Preparation**: Dataset cleaning and validation
- **YOLO Training**: YOLOv8 model training and configuration
- **Performance Analysis**: mAP metrics and detection evaluation
- **Visualization**: Training curves and detection results

## 🛠️ Development Workflow

1. **Data Preparation**: Organize datasets in proper directory structure
2. **Exploratory Analysis**: Use notebooks for data understanding
3. **Model Development**: Train and evaluate different architectures
4. **Performance Comparison**: Compare Custom CNN, MobileNetV2, and YOLO
5. **Model Selection**: Choose best performing model for deployment
6. **Web Deployment**: Launch Streamlit application for end-users

## ⚡ Performance Optimization

- **Model Caching**: Streamlit @st.cache_resource for efficient loading
- **CPU Inference**: Optimized for CPU deployment compatibility
- **Batch Processing**: Efficient data loading with PyTorch DataLoaders
- **Memory Management**: Proper tensor cleanup and garbage collection
- **Transfer Learning**: Reduced training time with pre-trained features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](#license) section below for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

## 📧 Contact

**Project Author**: Viraj Gavade  
**Repository**: [Aerial-Object-Classification---Detection](https://github.com/viraj-gavade/Aerial-Object-Classification---Detection)

---

## 📜 License

MIT License

Copyright (c) 2024 Viraj Gavade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*This project demonstrates effective implementation of modern computer vision techniques for practical aerial surveillance applications. The comprehensive approach combines multiple model architectures, detailed experimentation, and production-ready deployment solutions.*