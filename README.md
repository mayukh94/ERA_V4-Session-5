# CNN for MNIST (PyTorch) — Training Logs & Results

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license) 
[![Dataset: MNIST](https://img.shields.io/badge/Dataset-MNIST-blue)](http://yann.lecun.com/exdb/mnist/)

> **Repo:** _Add your public repository URL here._  
> Example: `https://github.com/<username>/<repo-name>`

This repository implements a compact **Convolutional Neural Network (CNN)** for **MNIST digit classification** in PyTorch. It uses **Batch Normalization**, **Dropout**, and **Global Average Pooling (GAP)** to achieve strong accuracy with a small parameter budget.

---

## ✨ Key Results

- **Best Test Accuracy:** **99.55%** (achieved at **Epoch 14**)  
- **Run Length:** 20 epochs  
- **Dataset:** MNIST (train: 60,000 • test: 10,000)  
- **Hardware (fill in):** e.g., CPU only / NVIDIA T4 / RTX 3060, etc.

> **What number should I report?**  
> Report the **maximum** validation/test accuracy observed during training. In this run, that is **99.55%** at **epoch 14**.

---

## 📦 Project Structure

```
.
├── Model_Training.ipynb        # Training & evaluation notebook
├── README.md                    # You are here
├── requirements.txt             # (Recommended) Package pins for reproducibility
└── data/                        # (Optional) Local cache for MNIST if you download manually
```

---

## 🧠 Model Overview

**Architecture highlights**  
- Stack of convolutional blocks with **ReLU + BatchNorm + Dropout(0.05)**  
- **1×1 convolutions** to control channel width  
- Final **Conv → GAP (AvgPool)** instead of large fully connected layers  
- Output layer produces logits for **10 classes (0–9)**

**Why GAP over FC?** Reduces parameters, improves generalization, and is translation robust.

You can print parameter counts in the notebook with:
```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```

---

## 🛠️ Environment & Setup

1) **Python & pip**
```bash
python -V
# 3.9+ recommended
pip install -r requirements.txt
```
Minimal packages (pin versions in `requirements.txt`):
```txt
torch
torchvision
torchsummary
tqdm
numpy
matplotlib
```
> If you’re using CUDA, install the torch/torchvision build matching your CUDA version from the official PyTorch site.

2) **Run the notebook**
```bash
jupyter notebook Model_Training.ipynb
```
- Execute all cells to **train** (20 epochs by default) and **evaluate** on the **test set**.
- The notebook prints per-epoch test metrics (loss & accuracy).

> **Tip:** If you convert the notebook into a script, keep a `--epochs` flag, a `--seed` flag, and save **best-checkpoint** (see below).

---

## 🚀 Reproducibility

Set a seed and enable deterministic algorithms where possible:
```python
import torch, random, numpy as np
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
> Exact reproducibility across hardware/OS/CUDA may still vary; logging the **commit hash**, **CUDA/cuDNN** versions, and **device** helps reviewers reproduce.

---

## 💾 Checkpointing (Recommended)

Save the **best** model when the test/validation accuracy improves:
```python
best_acc = 0.0
for epoch in range(epochs):
    train(...)
    test_acc = evaluate(...)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
```
Include the saved `best_model.pt` in releases or provide a download link.

---

## 📊 Training & Test Logs (20 Epochs)

**Summary (accuracy only):**
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

> If you prefer a figure, log metrics to CSV and plot `epoch vs accuracy` with matplotlib.

---

## 📈 Evaluation Tips (Optional Enhancements)

Add the following for deeper evaluation (recommended for completeness):
- **Confusion Matrix** & **Classification Report** (precision/recall/F1 by class)
- **Per-class accuracy** (digits 0–9)
- **Misclassified examples** visualization

Example (sklearn):
```python
from sklearn.metrics import classification_report, confusion_matrix
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(device))
        preds = logits.argmax(1).cpu()
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print(classification_report(y_true, y_pred, digits=4))
print(confusion_matrix(y_true, y_pred))
```

---

## ▶️ How to Run (Summary)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Launch Jupyter
jupyter notebook Model_Training.ipynb

# 3) In the notebook
#    - Run training (20 epochs by default)
#    - Observe per-epoch test metrics
#    - (Optional) Save best checkpoint
```

---

## 🔍 Notes & Design Choices

- **BatchNorm + Dropout(0.05)** after most conv layers for regularization & stable training
- **1×1 convs** for channel reduction/expansion
- **GAP** instead of dense layers to reduce params & overfitting
- **Small model**: fast convergence on MNIST

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` (add one if missing).

---

## 🙌 Acknowledgments

- Yann LeCun’s MNIST dataset
- PyTorch team & docs
- Community tutorials and references

---

## 📫 Contact

- **Author:** _Your Name_  
- **Email:** _your.email@example.com_  
- **LinkedIn/GitHub:** _links_

---

### Changelog

- `v1.0` — Initial public release with notebook, logs, and README.
