"""
Clean data generation for AND and XOR logic gates.

This module provides functions to generate the standard truth table data
for AND and XOR gates. These are the fundamental datasets used to demonstrate
neural network learning on linearly separable (AND) and non-linearly separable
(XOR) problems.
"""

from typing import Tuple
import numpy as np


def generate_and_gate_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean AND gate truth table data.

    The AND gate is a fundamental logic gate that outputs 1 (True) only when
    BOTH inputs are 1. This is a linearly separable problem, meaning a single
    perceptron can learn to solve it.

    Truth Table:
        Input A | Input B | Output
           0    |    0    |   0
           0    |    1    |   0
           1    |    0    |   0
           1    |    1    |   1

    Returns:
        Tuple containing:
        - X (np.ndarray): Input array of shape (4, 2) with all possible
          binary input combinations [[0,0], [0,1], [1,0], [1,1]]
        - y (np.ndarray): Output array of shape (4,) with AND gate
          outputs [0, 0, 0, 1]

    Example:
        >>> X, y = generate_and_gate_data()
        >>> X.shape
        (4, 2)
        >>> y.shape
        (4,)
    """
    # Define all possible binary input combinations
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    # AND gate: output is 1 only when both inputs are 1
    y = np.array([0, 0, 0, 1], dtype=np.float32)

    return X, y


def generate_xor_gate_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean XOR gate truth table data.

    The XOR (exclusive OR) gate outputs 1 (True) when the inputs are DIFFERENT.
    This is NOT linearly separable, requiring a multi-layer perceptron with
    at least one hidden layer to solve.

    Historical Note:
        The XOR problem was famously identified by Minsky and Papert (1969)
        as impossible for single-layer perceptrons, leading to the "AI winter"
        until backpropagation made multi-layer networks practical.

    Truth Table:
        Input A | Input B | Output
           0    |    0    |   0
           0    |    1    |   1
           1    |    0    |   1
           1    |    1    |   0

    Returns:
        Tuple containing:
        - X (np.ndarray): Input array of shape (4, 2) with all possible
          binary input combinations [[0,0], [0,1], [1,0], [1,1]]
        - y (np.ndarray): Output array of shape (4,) with XOR gate
          outputs [0, 1, 1, 0]

    Example:
        >>> X, y = generate_xor_gate_data()
        >>> X.shape
        (4, 2)
        >>> y.shape
        (4,)
    """
    # Define all possible binary input combinations
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    # XOR gate: output is 1 when inputs are different
    y = np.array([0, 1, 1, 0], dtype=np.float32)

    return X, y
