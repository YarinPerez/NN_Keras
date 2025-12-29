"""
Visualization utilities for training results and model analysis.

This module provides functions to create various plots for visualizing
neural network training progress and performance metrics.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for consistent, professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_history(
    history_dict: Dict,
    title: str,
    save_path: str
) -> None:
    """
    Plot training loss and accuracy curves over epochs.

    Args:
        history_dict: Dictionary with 'loss' and optionally 'accuracy' keys
        title: Title for the plot
        save_path: Path where plot will be saved

    Example:
        >>> plot_training_history(history, "AND Gate Training", "results/plots/and_loss.png")
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history_dict['loss'], linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'{title} - Loss Curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy if available
    if 'accuracy' in history_dict:
        axes[1].plot(history_dict['accuracy'], linewidth=2, color='#06A77D')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'{title} - Accuracy Curve', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)
    else:
        fig.delaxes(axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: str
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix as 2x2 numpy array [[TN, FP], [FN, TP]]
        title: Title for the plot
        save_path: Path where plot will be saved

    Example:
        >>> cm = np.array([[10, 2], [1, 8]])
        >>> plot_confusion_matrix(cm, "AND Gate", "results/plots/cm_and.png")
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1'],
        linewidths=2,
        linecolor='white'
    )

    plt.title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_optimizer_comparison(
    histories: Dict[str, Dict],
    metric: str,
    title: str,
    save_path: str
) -> None:
    """
    Compare training curves for different optimizers.

    Args:
        histories: Dict mapping optimizer names to history dictionaries
        metric: Metric to plot ('loss' or 'accuracy')
        title: Title for the plot
        save_path: Path where plot will be saved

    Example:
        >>> histories = {'SGD': sgd_history, 'Adam': adam_history}
        >>> plot_optimizer_comparison(histories, 'loss', "XOR Training", "results/plots/opt_comp.png")
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    colors = {'sgd': '#D62828', 'adam': '#2E86AB'}

    for name, history in histories.items():
        if metric in history:
            plt.plot(
                history[metric],
                label=name.upper(),
                linewidth=2,
                color=colors.get(name.lower(), '#000000')
            )

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'{title} - {metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
