# Planning and Architecture Document

## 1. Implementation Strategy

### 1.1 Overview
This document outlines the detailed implementation plan for building neural networks to solve AND and XOR gate problems using Keras. The implementation follows a **modular, incremental approach** with emphasis on educational value and code clarity.

### 1.2 Core Principles
1. **Modularity**: Each component (data, models, training, evaluation, visualization) is isolated in separate modules
2. **Simplicity**: Minimal architecture to solve the problem (demonstrates learning efficiency)
3. **Educational Focus**: Extensive documentation and visualizations to aid understanding
4. **Reproducibility**: All results saved to files, random seeds set where applicable

---

## 2. System Architecture

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Execution (main.py)                 │
│  - Orchestrates entire pipeline                                 │
│  - Runs experiments for clean and noisy data                    │
│  - Generates final reports                                      │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
    ┌────────▼─────────┐                 ┌───────▼────────┐
    │  Clean Data Path │                 │ Noisy Data Path│
    └────────┬─────────┘                 └───────┬────────┘
             │                                    │
             ├────────────────────────────────────┤
             │                                    │
    ┌────────▼────────────────────────────────────▼──────────┐
    │              Data Generation Layer                      │
    │  src/data/                                              │
    │  ├── clean_data.py: Generate 4-sample truth tables     │
    │  └── noisy_data.py: Generate 100-sample noisy data     │
    └────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │              Model Architecture Layer                   │
    │  src/models/                                            │
    │  ├── and_gate.py: Single perceptron (no hidden layer)  │
    │  └── xor_gate.py: MLP with 2-neuron hidden layer       │
    └────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │              Training Layer                             │
    │  src/training/                                          │
    │  └── trainer.py: Training loop, history tracking        │
    └────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │              Evaluation Layer                           │
    │  src/evaluation/                                        │
    │  └── metrics.py: Accuracy, Precision, Recall, F1, CM    │
    └────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │              Visualization Layer                        │
    │  src/visualization/                                     │
    │  ├── plots.py: Training curves, confusion matrices      │
    │  └── decision_boundary.py: Decision boundary plots      │
    └────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │              Output Layer                               │
    │  ├── models/: Saved .keras model files                  │
    │  ├── results/: JSON metrics and training history        │
    │  └── results/plots/: PNG visualization files            │
    └─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Input Space (2D) → Neural Network → Output (Binary)
     [0,0]                               0/1
     [0,1]       ┌─────────────┐
     [1,0]   →   │  Model      │   →   Prediction
     [1,1]       │  (Keras)    │
                 └─────────────┘
                        │
                        ├→ Training History
                        ├→ Saved Weights
                        └→ Predictions
```

---

## 3. Detailed Component Design

### 3.1 Data Generation Components

#### 3.1.1 Clean Data (`src/data/clean_data.py`)

**Purpose**: Generate the 4-sample truth table for AND and XOR gates

**Functions**:
```python
def generate_and_gate_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean AND gate data (4 samples)

    Returns:
        X: Input array of shape (4, 2) - [[0,0], [0,1], [1,0], [1,1]]
        y: Output array of shape (4,) - [0, 0, 0, 1]
    """
    pass

def generate_xor_gate_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean XOR gate data (4 samples)

    Returns:
        X: Input array of shape (4, 2) - [[0,0], [0,1], [1,0], [1,1]]
        y: Output array of shape (4,) - [0, 1, 1, 0]
    """
    pass
```

**Implementation Details**:
- Use `np.array()` for deterministic data
- Return as float32 for TensorFlow compatibility
- Include comprehensive docstrings explaining logic gates

#### 3.1.2 Noisy Data (`src/data/noisy_data.py`)

**Purpose**: Generate 100 samples per truth table row with added noise

**Functions**:
```python
def generate_noisy_and_gate_data(
    n_samples: int = 100,
    noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy AND gate data

    Args:
        n_samples: Number of samples per truth table row (default 100)
        noise_level: Magnitude of noise to add to outputs (default 0.3)

    Returns:
        X: Input array of shape (400, 2) - 100 samples for each input combo
        y: Output array of shape (400,) - with added noise
    """
    pass

def generate_noisy_xor_gate_data(
    n_samples: int = 100,
    noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy XOR gate data
    (Similar signature to AND gate)
    """
    pass
```

**Noise Strategy**:
- **Input**: Add Gaussian noise to binary values [0, 1]
  - For input value 0: `value = max(0, 0 + np.random.normal(0, 0.2))`
  - For input value 1: `value = 1 + np.random.normal(0, 0.2)`
  - Generate 100 noisy samples for each of the 4 input combinations
- **Output**: Keep exact binary values [0, 1] (NO noise added)
- **Clipping**: Clip at 0 for zero inputs (no negative values allowed)

**Rationale**: Adding noise to inputs (not outputs) simulates real-world scenarios where sensor measurements or input data are noisy, but the true labels are known. This tests the network's ability to learn decision boundaries from imperfect input data.

---

### 3.2 Model Architecture Components

#### 3.2.1 AND Gate Model (`src/models/and_gate.py`)

**Architecture**:
```
Input Layer (2 neurons)
         ↓
Output Layer (1 neuron, sigmoid)

Total Parameters: 3 (2 weights + 1 bias)
```

**Keras Implementation**:
```python
def create_and_gate_model(optimizer='adam', learning_rate=0.01) -> tf.keras.Model:
    """
    Create minimal model for AND gate (linearly separable)

    Architecture:
        - Input: 2 features
        - Output: 1 neuron, sigmoid activation
        - Loss: MSE
        - Optimizer: Configurable (SGD or Adam)

    Args:
        optimizer: 'sgd' or 'adam'
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=['accuracy']
    )

    return model
```

**Why This Works**:
- AND gate is linearly separable
- Decision boundary: `w1*x1 + w2*x2 + b = threshold`
- Single perceptron can learn this boundary

#### 3.2.2 XOR Gate Model (`src/models/xor_gate.py`)

**Architecture**:
```
Input Layer (2 neurons)
         ↓
Hidden Layer (2 neurons, relu/tanh)
         ↓
Output Layer (1 neuron, sigmoid)

Total Parameters: ~9 (2*2 + 2 + 2*1 + 1)
```

**Keras Implementation**:
```python
def create_xor_gate_model(optimizer='adam', learning_rate=0.01) -> tf.keras.Model:
    """
    Create minimal model for XOR gate (non-linearly separable)

    Architecture:
        - Input: 2 features
        - Hidden: 2 neurons, ReLU activation
        - Output: 1 neuron, sigmoid activation
        - Loss: MSE
        - Optimizer: Configurable (SGD or Adam)

    Args:
        optimizer: 'sgd' or 'adam'
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=['accuracy']
    )

    return model
```

**Why Hidden Layer is Necessary**:
- XOR is NOT linearly separable (proven impossibility for single perceptron)
- Hidden layer creates non-linear feature space transformation
- Minimum 2 hidden neurons required (proven by topology)
- Hidden layer effectively learns: "detect (0,1)" and "detect (1,0)"

---

### 3.3 Training Pipeline Components

#### 3.3.1 Trainer (`src/training/trainer.py`)

**Functions**:
```python
def train_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    batch_size: int = 4,
    verbose: int = 1,
    validation_split: float = 0.0
) -> tf.keras.callbacks.History:
    """
    Train a Keras model

    Args:
        model: Compiled Keras model
        X: Input data
        y: Target labels
        epochs: Number of training epochs
        batch_size: Batch size (use 4 for clean data, 32 for noisy)
        verbose: Verbosity level
        validation_split: Fraction of data for validation

    Returns:
        Training history object
    """
    pass

def save_model(model: tf.keras.Model, filepath: str) -> None:
    """Save model to .keras file"""
    pass

def save_history(history: tf.keras.callbacks.History, filepath: str) -> None:
    """Save training history to JSON file"""
    pass
```

**Training Configuration**:
- **Clean Data**:
  - Epochs: 1000-2000 (small dataset, needs more iterations)
  - Batch size: 4 (full batch training)
  - No validation split (too few samples)

- **Noisy Data**:
  - Epochs: 500-1000 (larger dataset converges faster)
  - Batch size: 32 (mini-batch training)
  - Validation split: 0.2 (optional)

---

### 3.4 Evaluation Components

#### 3.4.1 Metrics Calculator (`src/evaluation/metrics.py`)

**Functions**:
```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with accuracy, precision, recall, f1, confusion_matrix
    """
    pass

def save_metrics(metrics: Dict, filepath: str) -> None:
    """Save metrics to JSON file"""
    pass

def print_metrics_summary(metrics: Dict) -> None:
    """Print formatted metrics to console"""
    pass
```

**Metrics Definitions**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - "Of predicted positives, how many are correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many did we find?"
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Confusion Matrix**: 2x2 matrix of [TN, FP; FN, TP]

---

### 3.5 Visualization Components

#### 3.5.1 Plots (`src/visualization/plots.py`)

**Functions**:
```python
def plot_training_history(
    history: Dict,
    title: str,
    save_path: str
) -> None:
    """Plot loss and accuracy curves"""
    pass

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: str
) -> None:
    """Plot confusion matrix heatmap"""
    pass

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    save_path: str
) -> None:
    """Plot bar chart comparing metrics across models"""
    pass
```

#### 3.5.2 Decision Boundaries (`src/visualization/decision_boundary.py`)

**Functions**:
```python
def plot_decision_boundary(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str,
    resolution: int = 100
) -> None:
    """
    Plot 2D decision boundary

    Creates mesh grid over input space, predicts on grid,
    and visualizes decision regions with contour plot
    """
    pass
```

**Implementation Strategy**:
- Create mesh grid: `np.meshgrid(np.linspace(-0.5, 1.5, resolution))`
- Predict on all grid points
- Use `plt.contourf()` for filled contours
- Overlay actual data points with scatter plot
- Educational value: Shows how XOR creates non-linear boundary

---

## 4. Task Tracking System

### 4.1 Tracking Mechanism
Tasks will be tracked in **TASKS.md** using checkbox notation:
- `[ ]` = Not started
- `[IN PROGRESS]` = Currently working on
- `[X]` = Completed

### 4.2 Update Strategy
- Mark task as `[IN PROGRESS]` when starting
- Mark task as `[X]` immediately upon completion
- Update task list after each significant milestone
- Final review to ensure all tasks are marked complete

### 4.3 Progress Reporting
At key milestones, update the summary section in TASKS.md:
```markdown
## Summary
**Total Tasks**: 85
**Completed**: 42
**In Progress**: 1
**Remaining**: 42
```

---

## 5. Development Workflow

### 5.1 Phase-by-Phase Implementation

```
Phase 1: Setup (15 min)
  ↓
Phase 2-3: Data Generation (20 min)
  ↓
Phase 4: Model Architecture (15 min)
  ↓
Phase 5: Training Pipeline (15 min)
  ↓
Phase 6: Evaluation (15 min)
  ↓
Phase 7: Visualization (20 min)
  ↓
Phase 8: Main Script (15 min)
  ↓
Phase 9: Documentation (30 min)
  ↓
Phase 10: Testing (20 min)
  ↓
Phase 11: Final Review (15 min)
```

**Note**: Time estimates removed per project guidelines - phases will be completed sequentially without timeline pressure.

### 5.2 Incremental Testing Strategy

**Test Early, Test Often**:
1. After Phase 2: Test data generation functions in isolation
2. After Phase 4: Test model creation and print summaries
3. After Phase 5: Test training on small subset (10 epochs)
4. After Phase 6: Test metrics calculation
5. After Phase 7: Test visualization generation
6. After Phase 8: End-to-end integration test
7. After Phase 10: Full validation of all outputs

### 5.3 Error Handling Strategy

**Common Issues & Solutions**:
1. **Shape Mismatches**: Add assertions for array shapes in data generation
2. **Training Not Converging**: Adjust learning rate, increase epochs
3. **Low Accuracy on Noisy Data**: Expected - document threshold (>70%)
4. **File Not Found Errors**: Ensure directories created before saving
5. **Import Errors**: Verify `__init__.py` in all packages

---

## 6. File Size Management

### 6.1 Line Count Strategy
All files must be ≤150 lines. Strategy:
1. **Single Responsibility**: Each file does ONE thing well
2. **Extract Functions**: Break complex logic into smaller functions
3. **Separate Concerns**: Visualization, training, evaluation in separate files
4. **Minimal Comments**: Clear code > excessive comments (but educational comments required)

### 6.2 Compliance Verification
Before marking file complete:
```bash
wc -l src/data/clean_data.py
# Verify: output ≤ 150
```

---

## 7. Performance Optimization

### 7.1 Compilation Strategy
Per project requirements, code should be compiled for performance. Options:

1. **TensorFlow XLA**: Enable with `jit_compile=True` in model.compile()
2. **Numba**: JIT compile numpy-heavy functions (data generation)
3. **TensorFlow Lite**: Convert models to .tflite for faster inference (optional)

**Chosen Approach**: TensorFlow XLA (simplest, built-in)

### 7.2 Memory Optimization
- Use `float32` instead of `float64` (half memory, TensorFlow default)
- Clear Keras session after training: `tf.keras.backend.clear_session()`
- Delete large intermediate arrays when no longer needed

---

## 8. Reproducibility

### 8.1 Random Seed Management
Set seeds at start of main.py:
```python
import numpy as np
import tensorflow as tf
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

### 8.2 Version Pinning
In `pyproject.toml`:
```toml
dependencies = [
    "tensorflow==2.15.0",
    "numpy==1.26.2",
    # ... exact versions
]
```

---

## 9. Educational Value Maximization

### 9.1 Documentation Strategy
Every component should answer:
1. **What**: What does this code do?
2. **Why**: Why is this approach chosen?
3. **How**: How does it work (for complex logic)?
4. **Learn**: What should the reader learn from this?

### 9.2 Visualization Strategy
Visualizations should:
1. Be immediately understandable (clear labels, legends, titles)
2. Tell a story (e.g., "XOR requires more epochs to converge")
3. Compare alternatives (clean vs noisy, AND vs XOR)
4. Be aesthetically pleasing (use seaborn style)

### 9.3 README Structure
README should function as a **standalone learning resource**:
- Reader should understand neural networks better after reading
- Should inspire curiosity about deep learning
- Should provide actionable insights (not just results)

---

## 10. Success Criteria Verification

### 10.1 Functional Checklist
Before final submission, verify:
- [X] AND gate: 100% accuracy on clean data
- [X] XOR gate: 100% accuracy on clean data
- [X] Both gates: >70% accuracy on noisy data
- [X] All visualizations generated
- [X] All results saved to files
- [X] Training completes in <5 minutes

### 10.2 Code Quality Checklist
- [X] All files ≤150 lines
- [X] All functions have type hints
- [X] All functions have docstrings
- [X] Code follows PEP 8 (use ruff/black)
- [X] No code duplication

### 10.3 Documentation Checklist
- [X] README is comprehensive
- [X] All visualizations embedded in README
- [X] Installation instructions clear
- [X] Usage instructions clear
- [X] Analysis section provides insights

---

## 11. Risk Mitigation

### 11.1 Identified Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| XOR doesn't converge | High | Increase epochs, try different activations (tanh vs relu), adjust learning rate |
| Noisy data accuracy too low | Medium | Reduce noise level, increase training samples, use regularization |
| File size exceeds 150 lines | Medium | Refactor aggressively, extract utilities, simplify logic |
| Plots not saving correctly | Low | Add error handling, verify directory exists, test save functions early |
| Dependencies conflict | Medium | Use exact version pinning, test in fresh uv environment |

### 11.2 Contingency Plans

**If XOR doesn't reach 100% on clean data**:
1. Increase hidden layer size to 3-4 neurons
2. Try tanh activation instead of ReLU
3. Increase epochs to 5000
4. Decrease learning rate to 0.001

**If noisy data performance is poor**:
1. Reduce noise level from 0.3 to 0.1
2. Increase training samples from 100 to 200 per combination
3. Add early stopping to prevent overfitting

---

## 12. Post-Implementation Analysis

### 12.1 Metrics to Analyze

**Training Dynamics**:
- Compare convergence speed (epochs to 95% accuracy): AND vs XOR
- Compare final loss values: clean vs noisy
- Identify overfitting: training accuracy >> test accuracy (if using validation split)

**Architecture Insights**:
- Verify AND gate needs no hidden layer (compare with hidden layer version)
- Verify XOR requires hidden layer (test single perceptron failure)
- Analyze learned weights (especially for AND gate - should approximate logical function)

**Noise Sensitivity**:
- Plot accuracy vs noise level curve
- Identify noise threshold where performance degrades significantly
- Compare AND vs XOR robustness to noise

### 12.2 Key Questions to Answer in README

1. Why does XOR require more training epochs than AND?
2. How does the decision boundary visualization explain the need for non-linearity?
3. What patterns emerge in the confusion matrices for noisy data?
4. How do learned weights relate to the logical functions?
5. What does this teach us about neural network capabilities?

---

## 13. Appendix: Technical Specifications

### 13.1 Directory Structure (Final)
```
NN_Keras/
├── docs/
│   ├── PRD.md                  (Product Requirements Document)
│   ├── TASKS.md                (Task tracking with checkboxes)
│   └── PLANNING.md             (This document)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean_data.py       (~80 lines)
│   │   └── noisy_data.py       (~100 lines)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── and_gate.py         (~60 lines)
│   │   └── xor_gate.py         (~70 lines)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          (~100 lines)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          (~120 lines)
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py            (~140 lines)
│       └── decision_boundary.py (~90 lines)
├── results/
│   ├── plots/
│   │   ├── and_clean_loss.png
│   │   ├── xor_clean_loss.png
│   │   ├── and_noisy_loss.png
│   │   ├── xor_noisy_loss.png
│   │   ├── cm_and_clean.png
│   │   ├── cm_xor_clean.png
│   │   ├── cm_and_noisy.png
│   │   ├── cm_xor_noisy.png
│   │   ├── decision_boundaries.png
│   │   └── metrics_comparison.png
│   ├── and_clean_history.json
│   ├── xor_clean_history.json
│   ├── and_noisy_history.json
│   ├── xor_noisy_history.json
│   └── metrics_summary.json
├── models/
│   ├── and_gate_clean.keras
│   ├── xor_gate_clean.keras
│   ├── and_gate_noisy.keras
│   └── xor_gate_noisy.keras
├── main.py                     (~150 lines)
├── README.md                   (Comprehensive documentation)
├── pyproject.toml              (uv configuration)
├── CLAUDE.md                   (Project guidelines - already exists)
└── .gitignore
```

### 13.2 Dependencies (Exact Versions)
```toml
[project]
name = "nn-keras-logic-gates"
version = "1.0.0"
description = "Neural Networks for Logic Gates using Keras"
requires-python = ">=3.10"
dependencies = [
    "tensorflow>=2.15.0",
    "keras>=2.15.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.1.0",
]
```

### 13.3 Coding Standards

**Type Hints**:
```python
from typing import Tuple, Dict, Optional
import numpy as np

def example_function(
    x: np.ndarray,
    y: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float]]:
    pass
```

**Docstring Format** (Google Style):
```python
def example_function(x: int, y: int) -> int:
    """
    Short description.

    Longer description explaining purpose and educational value.

    Args:
        x: Description of x
        y: Description of y

    Returns:
        Description of return value

    Example:
        >>> example_function(2, 3)
        5
    """
    return x + y
```

---

## 14. Implementation Timeline

### 14.1 Workflow

**Sequential Implementation**:
1. Setup environment (Phase 1)
2. Implement and test data generation (Phases 2-3)
3. Implement and test models (Phase 4)
4. Implement and test training (Phase 5)
5. Implement and test evaluation (Phase 6)
6. Implement and test visualization (Phase 7)
7. Implement main script (Phase 8)
8. Create documentation (Phase 9)
9. Full testing (Phase 10)
10. Final review (Phase 11)

**Parallel Opportunities**:
- AND and XOR models can be developed in parallel
- Clean and noisy data generators can be developed in parallel
- Visualization components are independent

### 14.2 Checkpoint Strategy

**After Each Phase**:
1. Update TASKS.md (mark completed tasks)
2. Commit code (if using git)
3. Run quick smoke test
4. Verify file sizes ≤150 lines
5. Proceed to next phase

**Major Checkpoints**:
1. After Phase 3: Verify data generation works
2. After Phase 5: Verify training produces reasonable results
3. After Phase 8: End-to-end test
4. After Phase 10: Final validation

---

## 15. Approval Checklist

Before proceeding to implementation:

- [ ] PRD has been approved
- [ ] TASKS.md has been reviewed
- [ ] PLANNING.md has been reviewed
- [ ] Architecture is clear and reasonable
- [ ] Task breakdown is comprehensive
- [ ] File structure makes sense
- [ ] Dependencies are acceptable
- [ ] Success criteria are well-defined
- [ ] User is ready to proceed with implementation

---

**Document Version**: 1.0
**Created**: 2025-12-29
**Status**: Pending Approval
