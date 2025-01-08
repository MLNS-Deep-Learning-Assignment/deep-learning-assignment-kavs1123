# CNN Implementation for Digit Sum Recognition: Technical Report

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) to identify the sum of digits in images. The task is formulated as a regression problem where the model predicts a continuous value that is then rounded to the nearest integer.

## 2. Implementation Details

### 2.1 Data Handling
- Data is loaded from three separate numpy files (`data0.npy`, `data1.npy`, `data2.npy`) and their corresponding labels
- The datasets are concatenated to form a single training set
- Data is split into training (80%) and validation (20%) sets using scikit-learn's `train_test_split`
- Images are reshaped to dimensions of 40x168 pixels and normalized by dividing by 255

### 2.2 Model Architecture

The implemented CNN architecture consists of:

```python
# Convolutional Layers
- Conv2D(1 → 32, kernel_size=3)
- ReLU activation
- MaxPool2D(2x2)
- Conv2D(32 → 64, kernel_size=3)
- ReLU activation
- MaxPool2D(2x2)

# Fully Connected Layers
- Linear(flattened_size → 128)
- ReLU activation
- Linear(128 → 1)
```

### 2.3 Training Configuration

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate 0.001
- Batch Size: 32
- Training/Validation Split: 80/20
- Number of Epochs: 25

## 3. Performance Metrics

The model's performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Exact Accuracy (predictions that match labels exactly)
- Within-One Accuracy (predictions that are off by at most 1)
- The metrics are as follows
- For train dataset {'mse': 0.5587150933742523, 'rmse': 0.7474724699774917, 'mae': 0.5359167, 'exact_accuracy': 51.35833333333333, 'within_one_accuracy': 95.37916666666668}

- for validation set {'mse': 12.220009432474772, 'rmse': 3.495713007738875, 'mae': 2.7468333, 'exact_accuracy': 12.0, 'within_one_accuracy': 34.46666666666667}

## 4. Suggested Improvements



1. **Deeper Architecture**:
   - Add more convolutional layers with increasing filter sizes
   - Add batch normalization after convolutional layers

2. **Regularization Techniques**:
   - Try dropout layers



## 5. Conclusion

The current implementation provides a basic foundation  or baseline for digit sum recognition. By systematically implementing the suggested improvements, particularly focusing on data augmentation and architectural enhancements, the model's accuracy and robustness can be significantly improved.