"""
Noisy data generation for AND and XOR logic gates.

This module generates expanded datasets with Gaussian noise added to inputs.
This simulates real-world scenarios where sensor measurements or input data
are imperfect, testing the neural network's robustness and generalization.
"""

from typing import Tuple
import numpy as np


def generate_noisy_and_gate_data(
    n_samples: int = 100,
    noise_std: float = 0.2,
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy AND gate data with Gaussian noise on inputs.

    Creates n_samples variations of each truth table row by adding Gaussian
    noise to the input values. Outputs remain clean (0 or 1). This tests
    the network's ability to learn from noisy input data.

    Noise Strategy:
        - For input value 0: Add N(0, noise_std), clip at 0 (no negatives)
        - For input value 1: Add N(0, noise_std), no upper clip
        - Example: [0, 1] might become [0.15, 1.08] or [0.00, 0.92]

    Args:
        n_samples: Number of samples to generate per truth table row
                   (default: 100, total output will be 400 samples)
        noise_std: Standard deviation of Gaussian noise (default: 0.2)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple containing:
        - X (np.ndarray): Noisy input array of shape (n_samples*4, 2)
        - y (np.ndarray): Clean output array of shape (n_samples*4,)
                         with AND gate outputs

    Example:
        >>> X, y = generate_noisy_and_gate_data(n_samples=100, noise_std=0.2)
        >>> X.shape
        (400, 2)
        >>> y.shape
        (400,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Base truth table
    base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    base_outputs = np.array([0, 0, 0, 1], dtype=np.float32)

    X_list = []
    y_list = []

    # Generate n_samples for each truth table row
    for i, (inp, out) in enumerate(zip(base_inputs, base_outputs)):
        for _ in range(n_samples):
            # Add Gaussian noise to inputs
            noisy_input = inp + np.random.normal(0, noise_std, size=2)

            # Clip at 0 for inputs that were originally 0
            noisy_input = np.maximum(noisy_input, 0.0)

            X_list.append(noisy_input)
            y_list.append(out)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def generate_noisy_xor_gate_data(
    n_samples: int = 100,
    noise_std: float = 0.2,
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy XOR gate data with Gaussian noise on inputs.

    Creates n_samples variations of each truth table row by adding Gaussian
    noise to the input values. Outputs remain clean (0 or 1). This tests
    the network's ability to learn non-linear decision boundaries from
    noisy input data.

    Noise Strategy:
        - For input value 0: Add N(0, noise_std), clip at 0 (no negatives)
        - For input value 1: Add N(0, noise_std), no upper clip
        - Example: [0, 1] might become [0.12, 0.95] or [0.00, 1.11]

    Args:
        n_samples: Number of samples to generate per truth table row
                   (default: 100, total output will be 400 samples)
        noise_std: Standard deviation of Gaussian noise (default: 0.2)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple containing:
        - X (np.ndarray): Noisy input array of shape (n_samples*4, 2)
        - y (np.ndarray): Clean output array of shape (n_samples*4,)
                         with XOR gate outputs

    Example:
        >>> X, y = generate_noisy_xor_gate_data(n_samples=100, noise_std=0.2)
        >>> X.shape
        (400, 2)
        >>> y.shape
        (400,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Base truth table
    base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    base_outputs = np.array([0, 1, 1, 0], dtype=np.float32)

    X_list = []
    y_list = []

    # Generate n_samples for each truth table row
    for i, (inp, out) in enumerate(zip(base_inputs, base_outputs)):
        for _ in range(n_samples):
            # Add Gaussian noise to inputs
            noisy_input = inp + np.random.normal(0, noise_std, size=2)

            # Clip at 0 for inputs that were originally 0
            noisy_input = np.maximum(noisy_input, 0.0)

            X_list.append(noisy_input)
            y_list.append(out)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y
