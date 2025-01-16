# CNN Implementation for Digit Sum Recognition: Technical Report

## 1. Project Overview

This project implements a
- Advanced Convolutional Neural Network (CNN) with BatchNorm and Global Average Pooling
- A regression head built on top of a Resnet 50 (pretrained)

to identify the sum of digits in images. The task is formulated as a regression problem where the model predicts a continuous value that is then rounded to the nearest integer.

## 2. Implementation Details

### 2.1 Data Handling
- Data is loaded from three separate numpy files (`data0.npy`, `data1.npy`, `data2.npy`) and their corresponding labels
- The datasets are concatenated to form a single training set
- Data is split into training (80%) and validation (20%) sets using scikit-learn's `train_test_split`
- Images are reshaped to dimensions of 40x168 pixels and normalized by dividing by 255

### 2.2 Model Architecture

The implemented advanced CNN architecture consists of:

```python
# Convolutional Layers
- Conv2D(1 → 32, kernel_size=3, padding=1)
- BatchNorm2D(32)
- ReLU activation
- MaxPool2D(2x2)
- Conv2D(32 → 64, kernel_size=3, padding=1)
- BatchNorm2D(64)
- ReLU activation
- MaxPool2D(2x2)
- Conv2D(64 → 128, kernel_size=3, padding=1)
- BatchNorm2D(128)
- ReLU activation
- Attention Layer: Sequential(
    - Conv2D(128 → 1, kernel_size=1)
    - Sigmoid activation
  )
- Element-wise multiplication with attention weights
- Global Average Pooling (AdaptiveAvgPool2D(1))

# Fully Connected Layers
- Linear(128 → 256)
- ReLU activation
- Dropout(rate=0.5)
- Linear(256 → 128)
- ReLU activation
- Dropout(rate=0.5)
- Linear(128 → 1)

```


On the other hand the Resnet based model has the following architecture

```python
# Convolutional Layers
- Conv2D(1 → 64, kernel_size=7, stride=2, padding=3)
- ResNet-50 pre-trained layers (with modified first convolution for 1 input channel)
  - Multiple convolutional, batch normalization, and ReLU layers as per ResNet architecture

# Fully Connected Layers
- Linear(2048 → 1)  # Output from ResNet's final fully connected layer, modified to predict a single value
```

### 2.3 Training Configuration

#### 2.3.1 Advanced CNN

- Loss Function: Focal L1 Loss ( Custom implementation)
- Optimizer: Adam with learning rate 0.001
- Batch Size: 32
- Training/Validation Split: 80/20
- Number of Epochs: 50

#### 2.3.2 Resnet based Model

- Loss Function: Mse Loss ( No freezing of parameters)
- Optimizer: Adam with learning rate 0.001
- Batch Size: 32
- Training/Validation Split: 80/20
- Number of Epochs: 50

## 3. Performance Metrics

The model's performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Exact Accuracy (predictions that match labels exactly)
- Within-One Accuracy (predictions that are off by at most 1)
- The metrics are as follows

#### Advanced CNN

- For train dataset {'mse': 1.4782657799720764, 'rmse': 1.2158395370985746, 'mae': 0.9092917, 'exact_accuracy': 34.57083333333333, 'within_one_accuracy': 80.37083333333334}

- for validation set {'mse': 2.3246416521072386, 'rmse': 1.5246775567664261, 'mae': 1.1206666, 'exact_accuracy': 30.133333333333333, 'within_one_accuracy': 72.6}

- I observed that during the last assignemnet due to heavy overfitting the baseline model was giving very poor results on the validation sets. Hence this time we saved the model with the best validation loss . Therefore training accuracy did lower but this improved val accuracy significantly.

#### ResNet based model

- For train dataset {'mse': 0.032732275544355316, 'rmse': 0.18092063327424907, 'mae': 0.016041666, 'exact_accuracy': 98.70833333333333, 'within_one_accuracy': 99.775}

- for validation set {'mse': 0.26381338940560817, 'rmse': 0.513627675856362, 'mae': 0.09266666, 'exact_accuracy': 94.91666666666667, 'within_one_accuracy': 98.05}

- As you can see due to the pretraining of the resnet on a very large image dataset like ImageNet , it gives amazing results even without the individual digit labels.



## 5. Conclusion

The Resnet implementation is very powerful and using these pretrained model on huge image datasets like ImageNet, our problem of not having digit labels can be easily solved