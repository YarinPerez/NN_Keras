# Product Requirements Document: Neural Networks for Logic Gates

## 1. Overview

This project implements neural networks using Keras to solve classical logic gate problems (AND, XOR). The primary goal is **educational** - to demonstrate fundamental concepts in neural network design, training, and evaluation while exploring the differences between linearly separable (AND) and non-linearly separable (XOR) problems.

### 1.1 Educational Objectives

1. **Network Architecture Design**: Understanding how to construct minimal neural networks capable of solving specific problems
2. **Dataset Creation**: Learning to generate synthetic datasets for classification tasks
3. **Loss Functions**: Implementing and understanding Mean Squared Error (MSE) for binary classification
4. **Optimization**: Exploring gradient descent through Keras optimizers
5. **Robustness Testing**: Evaluating model performance under noisy conditions

## 2. Problem Background

### 2.1 Logic Gates

**AND Gate Truth Table:**
```
Input A | Input B | Output
   0    |    0    |   0
   0    |    1    |   0
   1    |    0    |   0
   1    |    1    |   1
```

**XOR Gate Truth Table:**
```
Input A | Input B | Output
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

### 2.2 Linear Separability

**AND Gate**: Linearly separable - can be solved with a single perceptron (no hidden layer required)

**XOR Gate**: NOT linearly separable - requires at least one hidden layer with 2+ neurons. This is a classical problem that demonstrates the necessity of multi-layer perceptrons.

**Historical Context**: The XOR problem was famously identified by Marvin Minsky and Seymour Papert in 1969 as a limitation of single-layer perceptrons, leading to the "AI winter" until backpropagation made multi-layer networks practical.

## 3. Project Structure

### Part A: Clean Data Implementation
Build minimal neural networks to solve AND and XOR gates with the standard 4-sample dataset.

### Part B: Noisy Data Implementation
Test model robustness by:
- Adding Gaussian noise to inputs: ±0.2 around binary values (0 or 1)
- Clipping: Inputs clipped at 0 (no negative values for zero inputs)
- Expanding dataset: 100 samples per feature combination instead of 4
- Outputs remain clean (0 or 1) - only inputs are noisy

## 4. Technical Requirements

### 4.1 Framework
- **Primary Framework**: Keras (TensorFlow backend)
- **Environment**: uv virtual environment
- **Performance**: Code should be optimized for runtime performance where applicable

### 4.2 Network Architecture Requirements

#### For AND Gate:
- **Minimum requirement**: Single perceptron (1 layer)
- **Input layer**: 2 neurons (for 2 inputs)
- **Output layer**: 1 neuron (binary output)
- **Activation function**: Sigmoid (for binary classification)

#### For XOR Gate:
- **Minimum requirement**: Multi-layer perceptron
  - Input layer: 2 neurons
  - Hidden layer: Minimum 2 neurons (proven minimum for XOR)
  - Output layer: 1 neuron
- **Activation functions**:
  - Hidden layer: ReLU, tanh, or sigmoid
  - Output layer: Sigmoid

### 4.3 Dataset Specifications

#### Part A - Clean Data:
```python
# 4 samples (exhaustive truth table)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]
y_xor = [0, 1, 1, 0]
```

#### Part B - Noisy Data:
- **Sample size**: 100 samples per truth table row (400 total samples)
- **Noise specification**: Gaussian noise added to INPUTS only (not outputs)
  - For input value 0: Add Gaussian noise with mean=0, std=0.2, then clip at 0 (no negatives)
  - For input value 1: Add Gaussian noise with mean=0, std=0.2
  - Example: Input [0, 1] becomes [0.15, 1.08] or [0.00, 0.92], etc.
- **Output values**: Remain clean binary [0, 1] - no noise added to outputs

### 4.4 Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
  - Formula: MSE = (1/n) Σ(y_true - y_pred)²
  - Rationale: Simple, differentiable, suitable for demonstrating basic optimization

- **Optimizer**: Train with BOTH and compare:
  - **SGD** (Stochastic Gradient Descent) - educational value, shows basic gradient descent mechanics
  - **Adam** - modern, adaptive learning rate, typically faster convergence
  - **Comparison Required**: Document runtime performance and convergence behavior
  - **Educational Analysis**: Explain why SGD is valuable for learning despite Adam being more efficient

- **Metrics**:
  - Accuracy (primary)
  - MSE loss (for tracking convergence)

- **Training parameters**:
  - Epochs: Sufficient for convergence (suggest 500-2000)
  - Batch size: For Part A (4 samples), use batch training
  - Learning rate: To be tuned, suggest starting at 0.01-0.1

## 5. Performance Metrics & Analysis

### 5.1 Required Metrics

1. **Accuracy**: Percentage of correct predictions
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Visual representation of predictions vs actual

### 5.2 Expected Performance

#### Part A (Clean Data):
- **AND Gate**: 100% accuracy expected (linearly separable)
- **XOR Gate**: 100% accuracy expected with proper architecture

#### Part B (Noisy Data):
- Performance will degrade based on noise level
- Expect 70-95% accuracy depending on noise magnitude
- Model should demonstrate generalization capability

## 6. Visualization Requirements

### 6.1 Training Visualizations

1. **Loss Curves**:
   - Plot training loss vs epochs
   - Show convergence behavior
   - Compare AND vs XOR convergence patterns

2. **Accuracy Curves**:
   - Plot training accuracy vs epochs
   - Demonstrate learning progress

### 6.2 Performance Visualizations

1. **Confusion Matrices**:
   - Heatmap visualization
   - Separate plots for AND and XOR
   - Separate plots for Part A (clean) and Part B (noisy)

2. **Decision Boundaries** (Advanced):
   - 2D visualization of learned decision boundaries
   - Show how network separates classes in input space
   - Particularly educational for XOR to show non-linear boundary

3. **Performance Bar Charts**:
   - Compare metrics (Accuracy, Precision, Recall, F1) across:
     - AND vs XOR
     - Clean vs Noisy data

### 6.3 Architecture Diagrams

- Visual representation of network architectures
- Show layers, neuron counts, and connections
- Label activation functions

## 7. Results & Output Requirements

### 7.1 Saved Artifacts

All results must be saved to files for later inspection:

1. **Model Checkpoints**:
   - `models/and_gate_clean.keras`
   - `models/xor_gate_clean.keras`
   - `models/and_gate_noisy.keras`
   - `models/xor_gate_noisy.keras`

2. **Training History**:
   - `results/and_gate_clean_history.json`
   - `results/xor_gate_clean_history.json`
   - `results/and_gate_noisy_history.json`
   - `results/xor_gate_noisy_history.json`

3. **Performance Metrics**:
   - `results/metrics_summary.json`
   - `results/confusion_matrices.pkl`

4. **Visualizations**:
   - `results/plots/loss_curves.png`
   - `results/plots/accuracy_curves.png`
   - `results/plots/confusion_matrix_and_clean.png`
   - `results/plots/confusion_matrix_xor_clean.png`
   - `results/plots/confusion_matrix_and_noisy.png`
   - `results/plots/confusion_matrix_xor_noisy.png`
   - `results/plots/decision_boundaries.png`
   - `results/plots/metrics_comparison.png`

### 7.2 README Documentation

The README.md must include:

1. **Project description** and learning objectives
2. **Theoretical background** on AND/XOR and linear separability
3. **Network architectures** with justification
4. **Results summary** with embedded visualizations
5. **Performance analysis**:
   - Clean vs noisy data comparison
   - AND vs XOR learning difficulty
   - Convergence analysis
6. **Key findings** and educational insights
7. **Usage instructions**

## 8. Code Organization

### 8.1 File Structure
```
NN_Keras/
├── docs/
│   ├── PRD.md
│   ├── TASKS.md
│   └── PLANNING.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean_data.py      (≤150 lines)
│   │   └── noisy_data.py      (≤150 lines)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── and_gate.py        (≤150 lines)
│   │   └── xor_gate.py        (≤150 lines)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         (≤150 lines)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         (≤150 lines)
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py           (≤150 lines)
│       └── decision_boundary.py (≤150 lines)
├── results/
│   └── plots/
├── models/
├── main.py                     (≤150 lines)
├── README.md
├── requirements.txt
└── pyproject.toml
```

### 8.2 Code Quality Standards

- **Maximum 150 lines per file**
- **Comprehensive docstrings** for educational purposes
- **Type hints** for clarity
- **Comments** explaining key concepts
- **Modular design** with clear separation of concerns

## 9. Dependencies

### 9.1 Required Packages
```
tensorflow >= 2.10.0
keras >= 2.10.0
numpy >= 1.23.0
matplotlib >= 3.5.0
seaborn >= 0.12.0
scikit-learn >= 1.1.0
pandas >= 1.5.0
```

### 9.2 Environment Setup
- Use `uv` for virtual environment management
- All dependencies specified in `pyproject.toml`

## 10. Success Criteria

### 10.1 Functional Requirements
- ✓ AND gate achieves 100% accuracy on clean data
- ✓ XOR gate achieves 100% accuracy on clean data
- ✓ Models maintain reasonable accuracy (>70%) on noisy data
- ✓ All visualizations generated and saved
- ✓ Training completes within reasonable time (<5 minutes total)

### 10.2 Educational Requirements
- ✓ Code is well-documented and easy to understand
- ✓ README provides clear explanations of concepts
- ✓ Visualizations effectively illustrate learning process
- ✓ Results demonstrate key ML concepts (overfitting, generalization, etc.)

### 10.3 Technical Requirements
- ✓ Code runs in uv virtual environment
- ✓ All files ≤150 lines
- ✓ Results saved for later inspection
- ✓ No errors or warnings during execution

## 11. Experimental Design

### 11.1 Noise Analysis (Part B)

**Hypothesis**: Neural networks trained on noisy data will demonstrate:
1. Longer convergence time
2. Lower final accuracy
3. More robust generalization (paradoxically, noise can act as regularization)

**Noise levels to test**:
- Low noise: ±0.1
- Medium noise: ±0.3
- High noise: ±0.5

Document how different noise levels affect:
- Training convergence
- Final accuracy
- Variance in predictions

## 12. Timeline & Milestones

**Note**: As per project guidelines, no time estimates provided. Implementation proceeds after approval.

**Milestones**:
1. PRD Approval
2. TASKS.md and PLANNING.md Approval
3. Data generation modules complete
4. Model architecture modules complete
5. Training pipeline complete
6. Evaluation and visualization complete
7. Documentation complete
8. Final review and submission

## 13. Future Extensions (Out of Scope)

Potential enhancements for learning purposes (not required for current implementation):
- Additional logic gates (OR, NAND, NOR)
- Comparison of different activation functions
- Hyperparameter tuning experiments
- Cross-validation on noisy data
- Regularization techniques (L1, L2, Dropout)
- Different loss functions (Binary Cross-Entropy vs MSE)

## 14. Clarifications (RESOLVED)

Implementation requirements confirmed:

1. **Noise specification**: ✓ Add Gaussian noise to INPUTS only (not outputs)
2. **Noise distribution**: ✓ Gaussian (mean=0, std=0.2)
3. **Noise magnitude**: ✓ ±0.2 around binary values, clip at 0 for zero inputs
4. **Optimizer choice**: ✓ Train with BOTH SGD and Adam, compare performance and explain educational value of SGD

## 15. Approval Checklist

Before proceeding to TASKS.md and PLANNING.md:

- [ ] Educational objectives clearly defined
- [ ] Technical requirements specified
- [ ] Architecture requirements detailed
- [ ] Performance metrics identified
- [ ] Visualization requirements listed
- [ ] Code organization structure defined
- [ ] Success criteria established
- [ ] User has reviewed and approved this PRD

---

**Document Version**: 1.0
**Created**: 2025-12-29
**Status**: Pending Approval
