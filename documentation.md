# Aerial Object Classification & Detection System

## Project Overview

This project implements a comprehensive machine learning solution for identifying and classifying aerial objects, specifically focusing on distinguishing between **birds** and **drones** in aerial imagery. The system combines both image classification and object detection approaches to provide accurate and reliable aerial object recognition.

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Dataset Description](#dataset-description)
3. [Technologies Used](#technologies-used)
4. [Model Architectures](#model-architectures)
5. [Training Approaches](#training-approaches)
6. [Implementation Details](#implementation-details)
7. [Results and Evaluation](#results-and-evaluation)
8. [Deployment](#deployment)
9. [Usage Instructions](#usage-instructions)

## Project Architecture

The project follows a modular architecture with clear separation of concerns:

```
├── data/                          # Dataset storage
│   ├── classification_dataset/    # Image classification data
│   └── object_detection_dataset/  # Object detection data with bounding boxes
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures and inference
│   ├── train/                    # Training and evaluation scripts
│   └── utils/                    # Utility functions
├── notebooks/                    # Jupyter notebooks for experimentation
├── models/                       # Trained model weights
└── streamlit_app/               # Web application for deployment
```

## Dataset Description

### Classification Dataset
The classification dataset consists of aerial images organized into two categories:

**Dataset Statistics:**
- **Training Set**: 2,662 images
  - Bird images: 1,414
  - Drone images: 1,248
- **Validation Set**: 442 images
  - Bird images: 217
  - Drone images: 225
- **Test Set**: 215 images
  - Bird images: 121
  - Drone images: 94

### Object Detection Dataset
The object detection dataset contains images with bounding box annotations for precise localization:

**Dataset Statistics:**
- **Training Set**: 2,662 images with corresponding label files
- **Test Set**: 224 images
- **Classes**: 2 (Bird, Drone)
- **Format**: YOLO format annotations

*[Placeholder for dataset sample images]*

## Technologies Used

### Core Technologies
- **Python 3.x**: Primary programming language
- **PyTorch**: Deep learning framework for model development
- **torchvision**: Computer vision utilities and pre-trained models

### Key Libraries
- **Ultralytics YOLO**: State-of-the-art object detection
- **scikit-learn**: Evaluation metrics and preprocessing
- **matplotlib**: Data visualization and plotting
- **PIL (Pillow)**: Image processing and manipulation
- **Streamlit**: Web application framework for deployment
- **Plotly**: Interactive visualization for web interface

### Model Architectures
- **MobileNetV2**: Lightweight convolutional neural network
- **Custom CNN**: Purpose-built architecture for aerial object classification
- **YOLOv8**: You Only Look Once object detection model

## Model Architectures

### 1. MobileNetV2 Transfer Learning

The primary classification model uses MobileNetV2 as a backbone with transfer learning:

```python
# Key features:
- Pre-trained on ImageNet
- Fine-tuned classifier head for 2-class classification
- Efficient architecture suitable for mobile deployment
- Freeze/unfreeze training strategy for optimal performance
```

**Architecture Details:**
- **Input Size**: 224×224×3 RGB images
- **Backbone**: MobileNetV2 feature extractor
- **Classifier**: Custom fully connected layer with dropout (0.2)
- **Output**: 2 classes (Bird, Drone)

### 2. Custom CNN Architecture

A lightweight custom CNN designed specifically for aerial object classification:

```python
# Architecture components:
- 3 Convolutional blocks with BatchNorm and ReLU
- Progressive channel expansion: 32 → 64 → 128
- Max pooling for spatial dimension reduction
- Fully connected classifier with dropout (0.3)
```

**Network Structure:**
- **Conv Block 1**: 3→32 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool
- **Conv Block 2**: 32→64 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool
- **Conv Block 3**: 64→128 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool
- **Classifier**: Flatten → Linear(128×28×28, 256) → ReLU → Dropout → Linear(256, 2)

### 3. YOLO Object Detection

YOLOv8 implementation for precise object localization and classification:

```python
# Configuration:
- Model: YOLOv8n (nano version for efficiency)
- Input size: 640×640 pixels
- Classes: 2 (Bird, Drone)
- Training epochs: 40
- Batch size: 16
```

## Training Approaches

### Data Preprocessing

**Image Transformations:**
- **Training**: Resize(224×224), RandomHorizontalFlip, RandomRotation(10°), Normalize
- **Validation/Test**: Resize(224×224), Normalize
- **Normalization**: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

### Training Strategy

1. **Transfer Learning Approach**:
   - Load pre-trained MobileNetV2 weights
   - Freeze backbone parameters initially
   - Fine-tune classifier head
   - Optional backbone fine-tuning in later epochs

2. **Training Configuration**:
   - **Optimizer**: Adam with learning rate 1e-3
   - **Loss Function**: CrossEntropyLoss
   - **Batch Size**: 32 (adjustable)
   - **Epochs**: 5-10 (with early stopping)

3. **Validation Strategy**:
   - Hold-out validation set for model selection
   - Performance monitoring during training
   - Best model checkpointing based on validation accuracy

## Implementation Details

### Data Loading Pipeline

The data loading system uses PyTorch's ImageFolder and DataLoader:

```python
# Key components:
- Automatic class discovery from folder structure
- Configurable batch sizes and transforms
- Efficient data loading with multiple workers
- Separate transforms for training and validation
```

### Model Training Loop

Training implementation includes:
- **Training Phase**: Forward pass, loss calculation, backpropagation
- **Validation Phase**: Model evaluation without gradient updates
- **Metrics Tracking**: Loss and accuracy monitoring
- **Model Checkpointing**: Save best performing models

### Inference Pipeline

The inference system provides:
- **Image Preprocessing**: Consistent with training pipeline
- **Prediction**: Forward pass with softmax probabilities
- **Post-processing**: Class prediction and confidence scores

## Results and Evaluation

### Model Performance Metrics

*[Placeholder for performance metrics table]*

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| MobileNetV2 | XX.X% | XX.X% | XX.X% | XX.X% |
| Custom CNN | XX.X% | XX.X% | XX.X% |XX.X% |
| YOLOv8 (mAP) | XX.X% | - | - | - |

### Confusion Matrix

*[Placeholder for confusion matrix visualization]*

### Training Curves

*[Placeholder for training/validation loss and accuracy curves]*

### Sample Predictions

*[Placeholder for sample prediction images with confidence scores]*

## Deployment

### Streamlit Web Application

The project includes a professional web interface built with Streamlit:

**Features:**
- **Real-time Image Classification**: Upload and classify aerial images
- **Confidence Visualization**: Interactive bar charts showing prediction confidence
- **Professional UI**: Modern gradient design with intuitive navigation
- **Performance Metrics**: Display of prediction confidence and class probabilities

**Technical Implementation:**
- **Model Loading**: Cached model loading for optimal performance
- **Image Processing**: Consistent preprocessing pipeline
- **Interactive Visualization**: Plotly charts for confidence distribution
- **Responsive Design**: Mobile-friendly interface

### Application Architecture

```python
# Key components:
- Streamlit framework for web interface
- PyTorch model inference backend
- Image upload and processing pipeline
- Real-time prediction and visualization
```

## Usage Instructions

### 1. Environment Setup

```bash
# Install required dependencies
pip install torch torchvision
pip install streamlit plotly
pip install ultralytics
pip install scikit-learn matplotlib pillow
```

### 2. Training a New Model

```python
# Classification training
from src.train.training import train_mobilenet

train_mobilenet(
    train_dir="data/classification_dataset/train",
    val_dir="data/classification_dataset/valid",
    test_dir="data/classification_dataset/test",
    epochs=10,
    batch_size=32
)
```

### 3. Model Evaluation

```python
# Evaluate trained model
from src.train.eval import evaluate_model

evaluate_model(
    model_path="models/best_classifier.pt",
    test_dir="data/classification_dataset/test"
)
```

### 4. YOLO Object Detection Training

```python
# Object detection training
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data/object_detection_Dataset/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16
)
```

### 5. Running the Web Application

```bash
# Launch Streamlit application
cd streamlit_app
streamlit run app.py
```

### 6. Making Predictions

```python
# Single image inference
from src.models.inference import load_image, predict
from src.models.mobilenet_cnn import load_mobilenet

# Load model and image
model = load_mobilenet(pretrained=False)
model.load_state_dict(torch.load("models/best_classifier.pt"))
img_tensor = load_image("path/to/image.jpg")

# Predict
prediction = predict(model, img_tensor, device="cpu")
print(f"Predicted class: {['bird', 'drone'][prediction]}")
```

## File Structure Details

### Source Code Organization

- **`src/data/dataset_loader.py`**: Data loading utilities with ImageFolder integration
- **`src/data/transformer.py`**: Image preprocessing and augmentation transforms
- **`src/models/mobilenet_cnn.py`**: MobileNetV2 model implementation
- **`src/models/custom_cnn.py`**: Custom CNN architecture definition
- **`src/models/inference.py`**: Inference pipeline and prediction utilities
- **`src/train/training.py`**: Model training orchestration
- **`src/train/eval.py`**: Model evaluation and metrics calculation
- **`src/utils/train_model.py`**: Training loop utilities and validation functions

### Notebooks

- **`notebooks/experiments.ipynb`**: Comprehensive experimentation notebook with data exploration, model training, and evaluation
- **`notebooks/yolo_model.ipynb`**: YOLO model training and evaluation workflows

### Models Directory

- **`models/best_classifier.pt`**: Best performing classification model weights
- **`models/mobilenet_phase1.pt`**: MobileNet model checkpoint
- **`models/custom_cnn.pt`**: Custom CNN model weights

---

*This documentation provides a comprehensive overview of the Aerial Object Classification & Detection system. The project demonstrates effective implementation of modern computer vision techniques for practical aerial surveillance applications.*