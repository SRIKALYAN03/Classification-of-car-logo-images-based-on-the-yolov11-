# Classification-of-car-logo-images-based-on-the-yolov11
# **YOLOv11 Classification Model Training and Deployment**

This repository contains a Python implementation for organizing datasets, training a classification model using YOLOv11, and deploying it for real-world applications. The project is designed to handle custom datasets and provides a complete pipeline from dataset preparation to model saving.

---

## **Task Description**

The primary objective of this project is to **classify car logos** into their respective car brands using a fine-tuned YOLOv11 classification model. The workflow includes:

1. **Organizing raw car logo datasets** into structured folders for training, validation, and testing.
2. **Fine-tuning a pre-trained YOLOv11 model** on the prepared dataset to classify logos into car brands.
3. **Saving the trained model** in `.pt` format for deployment and future predictions.
4. **Performing predictions** on new logo images to identify the car brand accurately.

---

## **Required Methods**

### **1. Dataset Preparation**

- **Method**: `prepare_dataset`
- **Purpose**: Prepares the dataset by applying transformations and loading it into `train`, `val`, and `test` splits.
- **Steps**:
   - Resize images to (224, 224).
   - Normalize images using mean and standard deviation.
   - Load images into PyTorch `DataLoader` for training and evaluation.

---

### **2. Model Training**

- **Method**: `train_logo_model`
- **Purpose**: Fine-tunes the YOLOv11 classification model on the prepared dataset.
- **Steps**:
   - Replace the pre-trained YOLOv11 classification head with a custom classifier.
   - Perform forward and backward passes using PyTorch.
   - Save the model checkpoint with the best validation accuracy.

---

### **3. Model Inference**

- **Method**: `predict_logo`
- **Purpose**: Predicts the class of a single image using the trained YOLOv11 model.
- **Steps**:
   - Load the saved model checkpoint.
   - Preprocess the input image (resize, normalize).
   - Perform forward pass to predict the class of the image.
   - Display the prediction with confidence score.

---

### **4. Model Evaluation**

- **Method**: `evaluate_model`
- **Purpose**: Evaluates the model's performance on the test dataset.
- **Steps**:
   - Generate predictions for the test set.
   - Calculate overall test accuracy.
   - Display the confusion matrix to analyze model performance.

---

## **How It Works**

1. **Prepare Dataset**: Organizes images into `train`, `val`, and `test` splits.
2. **Train Model**: Fine-tunes YOLOv11 with dropout for regularization.
3. **Evaluate**: Calculates accuracy and visualizes confusion matrix.
4. **Inference**: Performs predictions on new images.

---

## **Requirements**

Install the required dependencies using:
```bash
pip install torch torchvision ultralytics scikit-learn tqdm pillow matplotlib
