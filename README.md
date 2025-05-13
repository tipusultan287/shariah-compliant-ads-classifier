# 🚀 Fine-Tuning ResNet50 for Binary Ad Image Classification

This project fine-tunes a pre-trained ResNet50 CNN model to classify advertisements as **Allowed** or **Not Allowed** using deep learning. It leverages transfer learning to achieve high accuracy on a custom dataset with approximately 7,000 ad images.

---

## 📌 Objective

Automatically classify advertisements based on their visual content to aid moderation processes on digital platforms.

---

## 🧠 Problem Statement

Manual ad moderation is labor-intensive and prone to inconsistency. This project builds an automated system to classify ad images into **Allowed Ads** and **Not Allowed Ads** using binary image classification with transfer learning.

---

## 🗃️ Dataset

- Total Images: **6605**
- Categories:
  - `allowed_ads`: 4767
  - `not_allowed_ads`: 1838
- Split:
  - **Training**: 5283 (Allowed: 3845, Not Allowed: 1438)
  - **Validation**: 1322 (Allowed: 922, Not Allowed: 400)

Images were resized to **224x224** and normalized using `tf.keras.applications.resnet50.preprocess_input`.

---

## ⚙️ Methodology Summary

### 1. **Model Architecture**
- **Base Model**: ResNet50 (ImageNet weights)
- **Custom Head**:
  - GlobalAveragePooling2D
  - Dense(512) → BatchNorm → Dropout(0.5)
  - Dense(256) → BatchNorm → Dropout(0.3)
  - Dense(1, activation='sigmoid')

### 2. **Training Phases**
- **Phase 1**: Train only the custom head (ResNet frozen)
  - Optimizer: Adam (lr=1e-3)
- **Phase 2**: Fine-tune entire model
  - Optimizer: Adam (lr=1e-5)

### 3. **Data Augmentation**
Used `ImageDataGenerator` for training:
- Rotation: ±30°
- Width/Height shift: 20%
- Shear: 0.2
- Zoom: 20%
- Horizontal Flip

---

## 🧪 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Confusion Matrix


---

## 🖥️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ad-classification-resnet50.git
   cd ad-classification-resnet50
