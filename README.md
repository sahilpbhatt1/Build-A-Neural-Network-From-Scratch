# Neural Network Function Approximation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project demonstrates the implementation of a **feedforward neural network from scratch** using only NumPy, designed to approximate an unknown nonlinear function. Key technical components include:

- **Model Architecture Design**: Custom 2-layer neural network with ReLU activation
- **Hyperparameter Optimization**: Systematic grid search over learning rates and training epochs
- **Backpropagation Algorithm**: Manual implementation of gradient descent weight updates
- **Model Evaluation**: Statistical validation across multiple runs for robust performance metrics
- **Forecasting & Prediction**: Generalization to unseen test data with 3D visualization

## Key Technical Highlights

### 🔬 Machine Learning Fundamentals
- Built neural network **without high-level frameworks** to demonstrate deep understanding of underlying mathematics
- Implemented **stochastic gradient descent (SGD)** with manual weight updates
- Applied **ReLU activation functions** for nonlinear transformation

### 📊 Hyperparameter Optimization & Grid Search
- Conducted systematic search over **14 learning rates × 6 epoch configurations**
- Averaged results over **50 independent runs** per configuration for statistical robustness
- Identified optimal parameters: `α = 0.03`, `epochs = 6`

### 📈 Experimental Methodology
- **Train/Test Split**: 2,000 training samples, 2,000 test samples
- **Cross-validation**: Multiple random data splits to ensure generalization
- **Performance Metric**: Mean Absolute Error (MAE) on held-out test set

## Project Structure

```
neural-network-function-approximation/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── neural_network_function_approximation.ipynb  # Main Jupyter notebook
│
├── data/
│   ├── training_data.npy              # Training dataset (4,000 samples)
│   └── test_data.npy                  # Final evaluation dataset (2,000 samples)
```

## Network Architecture

```
Input Layer (2 features: x₁, x₂)
        │
        ▼
┌─────────────────────────────────────┐
│   Hidden Layer (4 neurons, ReLU)    │
│   y_j = ReLU(Σ w_ij * x_i + b_j)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Output Layer (1 neuron, Linear)   │
│   ŷ = Σ w_j * h_j + b               │
└─────────────────────────────────────┘
        │
        ▼
    Predicted Output (ŷ)
```

## Methodology

### 1. Hyperparameter Grid Search
The first phase systematically explores the hyperparameter space:

| Parameter | Values Tested |
|-----------|---------------|
| Learning Rate (α) | 0.001, 0.002, ..., 0.009, 0.01, 0.02, ..., 0.05 |
| Epochs | 1, 2, 3, 4, 5, 6 |

Each configuration is evaluated **50 times** with different random initializations to compute average error metrics.

### 2. Training with Optimal Parameters
Using the best-found hyperparameters:
- **Learning Rate**: 0.03
- **Epochs**: 6
- **Hidden Neurons**: 4

### 3. Backpropagation Update Rules
Weight updates follow the gradient descent rule:

```
Output Layer:  Δw_j = α * error * h_j
Hidden Layer:  Δw_ij = α * error * w_j * ReLU'(z_j) * x_i
```

## Results

The trained model successfully approximates the unknown target function with:
- **Robust convergence** across multiple random initializations
- **Consistent generalization** to unseen test data
- **Smooth 3D surface** visualization of learned function

## Technologies & Skills Demonstrated

| Category | Technologies |
|----------|-------------|
| **Programming** | Python, NumPy, Pandas |
| **ML Fundamentals** | Neural Networks, Backpropagation, SGD |
| **Optimization** | Grid Search, Hyperparameter Tuning |
| **Data Analysis** | Statistical Averaging, Cross-validation |
| **Visualization** | Matplotlib, 3D Surface Plots |
 
## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook neural_network_function_approximation.ipynb
```

## Future Enhancements

- [ ] Extend to deeper architectures with multiple hidden layers
- [ ] Implement additional activation functions (Sigmoid, Tanh, LeakyReLU)
- [ ] Add learning rate scheduling and momentum
- [ ] Port implementation to PyTorch for GPU acceleration
- [ ] Apply to time-series forecasting problems

## Author

**Sahil Bhatt**
 
