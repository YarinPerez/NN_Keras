"""
Training pipeline and utilities for neural network models.

This module provides functions for training Keras models, saving models and
training history, and managing the training process.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import tensorflow as tf
import numpy as np


def train_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    batch_size: int = 4,
    verbose: int = 0,
    validation_split: float = 0.0
) -> tuple[tf.keras.callbacks.History, float]:
    """
    Train a Keras model and measure training time.

    Args:
        model: Compiled Keras model to train
        X: Input data of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        epochs: Number of training epochs (default: 1000)
        batch_size: Batch size for training
                   - Use 4 for clean data (full batch)
                   - Use 32 for noisy data (mini-batch)
        verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        validation_split: Fraction of data to use for validation (default: 0.0)

    Returns:
        Tuple of (training history, training time in seconds)

    Example:
        >>> model = create_and_gate_model()
        >>> history, time = train_model(model, X, y, epochs=1000)
        >>> print(f"Training took {time:.2f} seconds")
    """
    start_time = time.time()

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=validation_split
    )

    training_time = time.time() - start_time

    return history, training_time


def save_model(model: tf.keras.Model, filepath: str) -> None:
    """
    Save a Keras model to disk in native Keras format.

    Args:
        model: Keras model to save
        filepath: Path where model will be saved (should end with .keras)

    Example:
        >>> save_model(model, 'models/and_gate_clean.keras')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)


def save_history(
    history: tf.keras.callbacks.History,
    filepath: str,
    training_time: float = None
) -> None:
    """
    Save training history to a JSON file.

    Args:
        history: Keras training history object
        filepath: Path where history will be saved (should end with .json)
        training_time: Optional training time in seconds to include

    Example:
        >>> save_history(history, 'results/and_clean_history.json', 12.5)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert history to dictionary and handle numpy types
    history_dict = {}
    for key, values in history.history.items():
        # Convert numpy arrays to lists for JSON serialization
        history_dict[key] = [float(v) for v in values]

    # Add metadata
    if training_time is not None:
        history_dict['training_time_seconds'] = float(training_time)

    history_dict['total_epochs'] = len(history_dict.get('loss', []))

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)


def load_history(filepath: str) -> Dict[str, Any]:
    """
    Load training history from a JSON file.

    Args:
        filepath: Path to history JSON file

    Returns:
        Dictionary containing training history

    Example:
        >>> history = load_history('results/and_clean_history.json')
        >>> print(history['loss'][-1])  # Final loss value
    """
    with open(filepath, 'r') as f:
        return json.load(f)
