# 🧠 Neural Network Architectures: Comparing Single-Layer and Deep MLPs

This project explores how neural network depth and architecture impact classification performance on two widely-used datasets: **CelebA** and **MNIST**. We compare a **shallow neural network** (one hidden layer) with a **deep neural network (DNN)** comprising multiple hidden layers, using both traditional and modern deep learning techniques.

---

## 🚀 Project Objectives

- Evaluate the impact of depth on model accuracy and training time.
- Compare traditional optimization (Conjugate Gradient) vs. modern optimizers (Adam).
- Examine regularization, hidden units, and layer counts on performance.
- Build modular, extensible neural network pipelines for both datasets.

---

## 📁 Repository Contents

```
├── face_nn.py               # Single-layer neural net logic for CelebA
├── dnn.py                   # Deep neural net logic for CelebA (PyTorch)
├── nn_script.py             # Single-layer neural net for MNIST
├── ML - PROJECT 1.ipynb     # Unified notebook for training, evaluation, plots
├── face_all.pickle          # Preprocessed CelebA dataset
├── mnist_all.mat            # MNIST dataset in .mat format
├── *.pkl                    # Serialized model results and parameters
├── *.png                    # Visual outputs (accuracy/loss graphs)
└── README.md
```

---

## 📊 Datasets Used

- **CelebA (Binary Classification)**  
  Task: Predict whether the person is wearing glasses  
  Features: Pre-extracted facial features from CelebA dataset

- **MNIST (Multi-Class Classification)**  
  Task: Digit classification (0–9)  
  Format: Loaded from `.mat` file (MATLAB format)

---

## 🧪 Model Architectures

### 1️⃣ Single-Layer Neural Network
- Implemented from scratch using **NumPy & SciPy**
- Optimized with **Conjugate Gradient Descent**
- Tuned for:
  - Number of hidden units
  - L2 Regularization strength (λ)

### 🔁 Deep Neural Network (DNN)
- Implemented using **PyTorch**
- Configurable architecture: 4, 12, or 20 layers
- Components:
  - **ReLU** activation
  - **Batch Normalization**
  - **Dropout**
- Optimized with **Adam Optimizer**
- Tuned for:
  - Number of layers
  - Learning rate

---

## 📈 Results Summary

| Model      | Dataset | Best Accuracy | Training Time | Optimal Settings               |
|------------|---------|---------------|----------------|---------------------------------|
| face_nn    | CelebA  | 85.29%        | 44.06 sec      | 1 hidden unit, λ = 12          |
| dnn        | CelebA  | 86.41%        | 82.74 sec      | 12 layers, LR = 0.01           |
| nn_script  | MNIST   | 93.57%        | 69.94 sec      | 20 hidden units, λ = 1         |

---

## 📊 Visualizations (Included in Notebook)

- Accuracy vs. Regularization (λ)
- Accuracy vs. Hidden Units
- Accuracy vs. Number of Layers (DNN)
- Training Time vs. Layer Depth (DNN)
- Sample predictions and error analysis

---
