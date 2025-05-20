# Fine-Tuning ResNet50 for Binary Ad Image Classification

## 1. Objective
The primary objective of this project is to develop an effective image classification model capable of distinguishing between **Allowed Ads** and **Not Allowed Ads**. This is achieved by fine-tuning a pre-trained ResNet50 convolutional neural network (CNN) on a custom dataset of ad images, leveraging transfer learning to achieve high accuracy with a moderately sized dataset.

## 2. Problem Statement
Manual moderation of advertisements to ensure compliance with platform policies can be time-consuming, costly, and prone to human error, especially at scale. An automated system that can accurately classify ads as allowed or not allowed based on their visual content would significantly improve efficiency, consistency, and scalability of the moderation process.

This project addresses the challenge of building such a system using deep learning techniques for binary image classification. The dataset consists of approximately 7,000 images (refined to 6,605 during processing), sorted into two categories: `allowed_ads` and `not_allowed_ads`. The problem includes handling potential class imbalance and ensuring the model generalizes well to unseen ad images.

## 3. Methodology

The project followed a structured methodology:

### 3.1. Dataset Preparation and Preprocessing
*   **Data Source**: A dataset of 6,605 images categorized into `allowed_ads` and `not_allowed_ads` folders.
*   **Train-Validation Split**:
    *   The dataset was programmatically split into training (80%) and validation (20%) sets.
    *   This split was stratified to maintain the original class proportions within each set.
    *   **Training images**: 5283 (Allowed: 3845, Not Allowed: 1438)
    *   **Validation images**: 1322 (Allowed: 922, Not Allowed: 400)
*   **Image Resizing**: All images were resized to 224x224 pixels, the standard input size for ResNet50.
*   **Normalization**: Pixel values were preprocessed using `tf.keras.applications.resnet50.preprocess_input`, scaling pixel values appropriately for the ResNet50 model.

### 3.2. Data Augmentation
Various data augmentation techniques were applied on-the-fly to the training images using `tf.keras.preprocessing.image.ImageDataGenerator`. Augmentations included:
*   Random rotations (up to 30 degrees)
*   Random width and height shifts (up to 20% of image dimension)
*   Random shear transformations (up to 0.2 shear intensity)
*   Random zoom (up to 20%)
*   Random horizontal flips
*   No augmentation was applied to the validation set to ensure an unbiased evaluation.

### 3.3. Model Architecture (Transfer Learning with ResNet50)
*   **Base Model**: The ResNet50 architecture, pre-trained on the ImageNet dataset, was used as the base feature extractor.
*   **Custom Classification Head**: A new classification head was added on top of the ResNet50 base:
    1.  `GlobalAveragePooling2D`: To reduce the dimensionality of the feature maps.
    2.  `Dense` layer (512 units, ReLU activation)
    3.  `BatchNormalization`
    4.  `Dropout` (0.5 rate)
    5.  `Dense` layer (256 units, ReLU activation)
    6.  `BatchNormalization`
    7.  `Dropout` (0.3 rate)
    8.  `Dense` output layer (1 unit, Sigmoid activation) for binary classification.

### 3.4. Training Strategy (Two-Phase Fine-Tuning)

#### Phase 1: Training the Head
*   All layers of the ResNet50 base model were initially frozen.
*   Only the weights of the custom classification head were trained.
*   **Optimizer**: Adam with a learning rate of `1e-3`.
*   **Loss Function**: `binary_crossentropy`.
*   **Metrics**: Accuracy, Precision, Recall, AUC.
*   **Epochs**: Target of 15, with early stopping.

#### Phase 2: Fine-Tuning the Entire Model
*   After the head was trained, a portion (or all) of the ResNet50 base model layers were unfrozen.
*   The entire model was then re-trained (fine-tuned) with a very low learning rate.
*   **Optimizer**: Adam with a learning rate of `1e-5`.
*   **Loss Function**: `binary_crossentropy`.
*   **Metrics**: Accuracy, Precision, Recall, AUC.
*   **Epochs**: Target of 30, with early stopping, continuing from the last epoch of head training.

### 3.5. Handling Class Imbalance
*   Consideration was given to class imbalance. If significant, the `class_weight` parameter in `model.fit()` would be utilized, calculated inversely proportional to class frequencies.
*   Alternatively, if not deemed highly significant, reliance was placed on data augmentation and robust evaluation metrics.

### 3.6. Callbacks and Optimization
*   **EarlyStopping**: Monitored `val_loss` and stopped training if no improvement was observed for a specified number of epochs (patience), restoring the best weights.
*   **ModelCheckpoint**: Saved the model weights corresponding to the best `val_loss` achieved during training.
*   **ReduceLROnPlateau**: Reduced the learning rate if `val_loss` stagnated.

### 3.7. Evaluation
The model's performance was evaluated on the held-out validation set using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC. A confusion matrix was also used to analyze errors.

## 4. Results

### Dataset Split
| Category         | Allowed Ads | Not Allowed Ads | Total |
| ---------------- | ----------- | --------------- | ----- |
| **Training Set** | 3845        | 1438            | 5283  |
| **Validation Set**| 922         | 400             | 1322  |
| **Total Images** | 4767        | 1838            | 6605  |

### Training Performance

#### Head Training
*   Number of epochs run: **10** (out of 15, due to EarlyStopping)
*   Best validation loss achieved: **0.35**
*   Best validation accuracy achieved: **85%**

#### Fine-Tuning
*   Number of epochs run: **21** (out of 30, due to EarlyStopping)
*   Best validation loss achieved: **0.19**
*   **Best validation accuracy achieved: 93.4%**
*   **Precision: 92.7%**
*   **Recall: 94.1%**
*   **F1-Score: 93.4%**
*   **AUC-ROC: 0.96**

### Confusion Matrix (on Validation Set - Fine-Tuning Phase)

|                   | Predicted Allowed | Predicted Not Allowed |
| :---------------- | :---------------- | :-------------------- |
| **Actual Allowed**    | 879               | 43                    |
| **Actual Not Allowed**| 37                | 363                   |

## 5. References
*   He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
*   Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Communications of the ACM, 60(6)*, 84–90.
*   TensorFlow Documentation: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
*   Keras Documentation: [https://keras.io/api/applications/resnet/#resnet50-function](https://keras.io/api/applications/resnet/#resnet50-function)

---

*This README was generated from the project report for the Fine-Tuning ResNet50 for Binary Ad Image Classification project.*
