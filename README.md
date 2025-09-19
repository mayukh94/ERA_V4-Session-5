# CNN Model Training for MNIST Classification

This repository contains a Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch. The model demonstrates various deep learning techniques including batch normalization, dropout, and global average pooling.

## Model Architecture

### Network Structure
The CNN model consists of three main convolutional blocks:

1. **Conv1 Block**: 
   - Input: 1×28×28 (grayscale MNIST images)
   - Convolution: 1→8 channels, 3×3 kernel, padding=1
   - ReLU activation
   - Batch Normalization
   - Dropout (0.05)
   - Convolution: 8→16 channels, 3×3 kernel
   - ReLU activation
   - Batch Normalization
   - Dropout (0.05)
   - MaxPool2d (2×2)

2. **Conv2 Block**:
   - Convolution: 16→8 channels, 1×1 kernel, padding=1
   - ReLU + BatchNorm + Dropout
   - Convolution: 8→16 channels, 3×3 kernel
   - ReLU + BatchNorm + Dropout
   - Convolution: 16→32 channels, 3×3 kernel
   - ReLU + BatchNorm + Dropout
   - Convolution: 32→32 channels, 3×3 kernel
   - ReLU + BatchNorm + Dropout

3. **Conv3 Block**:
   - Convolution: 32→10 channels, 3×3 kernel
   - ReLU + BatchNorm + Dropout
   - **Global Average Pooling**: AvgPool2d(5×5)
   - Output: 10 classes (digits 0-9)

## Model Analysis

### 1. Total Parameter Count Test
The model uses `torchsummary` to display the total parameter count:
```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```

**Parameter Count**: The model is designed to be lightweight with a relatively small number of parameters due to:
- Use of 1×1 convolutions for channel reduction
- Global Average Pooling instead of fully connected layers
- Strategic use of batch normalization and dropout

### 2. Use of Batch Normalization
**Implementation**: ✅ **Extensively Used**

Batch Normalization is applied after every convolutional layer:
- `nn.BatchNorm2d(8)` after first conv layer
- `nn.BatchNorm2d(16)` after second conv layer  
- `nn.BatchNorm2d(32)` after third and fourth conv layers
- `nn.BatchNorm2d(10)` after final conv layer

**Benefits**:
- Stabilizes training by normalizing inputs to each layer
- Reduces internal covariate shift
- Allows for higher learning rates
- Acts as a regularizer

### 3. Use of Dropout
**Implementation**: ✅ **Consistently Applied**

Dropout is applied throughout the network with a rate of 0.05:
- After every BatchNorm layer
- Applied to all convolutional blocks
- Helps prevent overfitting
- Provides regularization during training

**Configuration**:
```python
nn.Dropout(0.05)  # 5% dropout rate
```

### 4. Use of Fully Connected Layer or GAP
**Implementation**: ✅ **Global Average Pooling (GAP) Used**

The model uses **Global Average Pooling** instead of fully connected layers:
```python
nn.AvgPool2d(5,5)  # Global Average Pooling
```

**Advantages of GAP over FC layers**:
- **Reduced Parameters**: Eliminates the need for large fully connected layers
- **Better Generalization**: Reduces overfitting risk
- **Translation Invariance**: More robust to spatial translations
- **Computational Efficiency**: Faster inference and training

**Architecture Flow**:
```
Conv3 → ReLU → BatchNorm → Dropo
```

## Training & Test Logs

Below is the test accuracy recorded after each epoch for a 20‑epoch run:

```
Epoch 0 :  Test Accuracy 98.24%
Epoch 1 :  Test Accuracy 99.04%
Epoch 2 :  Test Accuracy 99.31%
Epoch 3 :  Test Accuracy 99.33%
Epoch 4 :  Test Accuracy 99.41%
Epoch 5 :  Test Accuracy 99.30%
Epoch 6 :  Test Accuracy 99.43%
Epoch 7 :  Test Accuracy 99.41%
Epoch 8 :  Test Accuracy 99.37%
Epoch 9 :  Test Accuracy 99.51%
Epoch 10:  Test Accuracy 99.46%
Epoch 11:  Test Accuracy 99.54%
Epoch 12:  Test Accuracy 99.46%
Epoch 13:  Test Accuracy 99.53%
Epoch 14:  Test Accuracy 99.55%   <-- Highest
Epoch 15:  Test Accuracy 99.53%
Epoch 16:  Test Accuracy 99.54%
Epoch 17:  Test Accuracy 99.53%
Epoch 18:  Test Accuracy 99.54%
Epoch 19:  Test Accuracy 99.54%
```

<details>
<summary><strong>Full raw logs (loss & accuracy)</strong></summary>

```
 Epoch: 0
Test set: Average loss: 0.1335, Accuracy: 9824/10000 (98.24%)

 Epoch: 1
Test set: Average loss: 0.0958, Accuracy: 9904/10000 (99.04%)

 Epoch: 2
Test set: Average loss: 0.0938, Accuracy: 9931/10000 (99.31%)

 Epoch: 3
Test set: Average loss: 0.0859, Accuracy: 9933/10000 (99.33%)

 Epoch: 4
Test set: Average loss: 0.0813, Accuracy: 9941/10000 (99.41%)

 Epoch: 5
Test set: Average loss: 0.0957, Accuracy: 9930/10000 (99.30%)

 Epoch: 6
Test set: Average loss: 0.0888, Accuracy: 9943/10000 (99.43%)

 Epoch: 7
Test set: Average loss: 0.0919, Accuracy: 9941/10000 (99.41%)

 Epoch: 8
Test set: Average loss: 0.0896, Accuracy: 9937/10000 (99.37%)

 Epoch: 9
Test set: Average loss: 0.0925, Accuracy: 9951/10000 (99.51%)

 Epoch: 10
Test set: Average loss: 0.0926, Accuracy: 9946/10000 (99.46%)

 Epoch: 11
Test set: Average loss: 0.0899, Accuracy: 9954/10000 (99.54%)

 Epoch: 12
Test set: Average loss: 0.0891, Accuracy: 9946/10000 (99.46%)

 Epoch: 13
Test set: Average loss: 0.0888, Accuracy: 9953/10000 (99.53%)

 Epoch: 14
Test set: Average loss: 0.0895, Accuracy: 9955/10000 (99.55%)

 Epoch: 15
Test set: Average loss: 0.0874, Accuracy: 9953/10000 (99.53%)

 Epoch: 16
Test set: Average loss: 0.0907, Accuracy: 9954/10000 (99.54%)

 Epoch: 17
Test set: Average loss: 0.0905, Accuracy: 9953/10000 (99.53%)

 Epoch: 18
Test set: Average loss: 0.0877, Accuracy: 9954/10000 (99.54%)

 Epoch: 19
Test set: Average loss: 0.0902, Accuracy: 9954/10000 (99.54%)
```
</details>
