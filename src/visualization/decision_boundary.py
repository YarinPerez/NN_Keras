"""
Decision boundary visualization for 2D classification problems.

This module provides functions to visualize how neural networks partition
the input space for binary classification tasks like AND and XOR gates.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_decision_boundary(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str,
    resolution: int = 200
) -> None:
    """
    Plot the decision boundary learned by a 2D classification model.

    This visualization shows how the neural network partitions the 2D input
    space into regions for different classes. This is particularly educational
    for understanding:
        - Linear boundaries for AND gate (linearly separable)
        - Non-linear boundaries for XOR gate (not linearly separable)

    How It Works:
        1. Create a dense mesh grid covering the input space
        2. Predict class probabilities for every point on the grid
        3. Color-code regions based on predicted class
        4. Overlay actual training data points

    Args:
        model: Trained Keras model for 2D binary classification
        X: Training input data of shape (n_samples, 2)
        y: Training labels of shape (n_samples,)
        title: Title for the plot
        save_path: Path where plot will be saved
        resolution: Number of points per axis for mesh grid (default: 200)

    Example:
        >>> plot_decision_boundary(model, X, y, "AND Gate", "results/plots/db_and.png")
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create mesh grid
    margin = 0.3
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points, verbose=0)
    Z = Z.reshape(xx.shape)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot decision boundary as contour
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(contour, label='Predicted Probability')

    # Plot decision boundary line at 0.5 threshold
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='--')

    # Plot training data points
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap='RdYlBu_r',
        s=200,
        edgecolors='black',
        linewidths=2,
        alpha=1.0
    )

    # Add labels to points
    for i, (x_val, y_val) in enumerate(X):
        plt.annotate(
            f'({int(x_val)},{int(y_val)})',
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.xlabel('Input 1', fontsize=12, fontweight='bold')
    plt.ylabel('Input 2', fontsize=12, fontweight='bold')
    plt.title(f'Decision Boundary - {title}', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
