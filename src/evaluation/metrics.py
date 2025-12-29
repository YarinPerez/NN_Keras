"""
Model evaluation and metrics calculation.

This module provides functions to calculate comprehensive performance metrics
for binary classification models, including accuracy, precision, recall, F1-score,
and confusion matrix.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Metrics Explained:
        - Accuracy: (TP + TN) / Total
          Overall correctness of predictions

        - Precision: TP / (TP + FP)
          "Of predicted positives, how many are correct?"
          High precision = few false positives

        - Recall: TP / (TP + FN)
          "Of actual positives, how many did we find?"
          High recall = few false negatives

        - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
          Harmonic mean of precision and recall
          Balances both metrics

        - Confusion Matrix: [[TN, FP], [FN, TP]]
          Visual representation of prediction vs actual

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities (0.0 to 1.0)
        threshold: Classification threshold (default: 0.5)
                   Predictions >= threshold become class 1

    Returns:
        Dictionary containing all metrics and confusion matrix

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred_binary)),
        'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred_binary, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary).tolist(),
        'threshold': float(threshold)
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary containing metrics
        filepath: Path where metrics will be saved (should end with .json)

    Example:
        >>> save_metrics(metrics, 'results/and_clean_metrics.json')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics_summary(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    """
    Print a formatted summary of metrics to console.

    Args:
        metrics: Dictionary containing metrics
        model_name: Name of the model for display purposes

    Example:
        >>> print_metrics_summary(metrics, "AND Gate - Clean Data")
        === AND Gate - Clean Data ===
        Accuracy:  1.0000
        Precision: 1.0000
        Recall:    1.0000
        F1-Score:  1.0000
        ...
    """
    print(f"\n{'=' * 50}")
    print(f"{model_name:^50}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'=' * 50}\n")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.

    Args:
        filepath: Path to metrics JSON file

    Returns:
        Dictionary containing metrics

    Example:
        >>> metrics = load_metrics('results/and_clean_metrics.json')
    """
    with open(filepath, 'r') as f:
        return json.load(f)
