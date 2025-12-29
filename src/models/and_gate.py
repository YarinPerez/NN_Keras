"""
AND gate neural network model.

This module implements a minimal neural network for solving the AND gate problem.
The AND gate is linearly separable, meaning it can be solved by a single perceptron
(no hidden layer required).
"""

import tensorflow as tf
from typing import Literal


def create_and_gate_model(
    optimizer: Literal['sgd', 'adam'] = 'adam',
    learning_rate: float = 0.01
) -> tf.keras.Model:
    """
    Create a minimal neural network model for the AND gate problem.

    Architecture:
        Input Layer: 2 neurons (for 2 binary inputs)
        Output Layer: 1 neuron with sigmoid activation
        Total Parameters: 3 (2 weights + 1 bias)

    Why This Works:
        The AND gate is linearly separable. This means there exists a straight
        line (or hyperplane in higher dimensions) that can perfectly separate
        the positive cases from negative cases in the input space.

        Mathematical Representation:
            output = sigmoid(w1*x1 + w2*x2 + b)

        For AND gate, the network learns weights approximately like:
            w1 ≈ 1, w2 ≈ 1, b ≈ -1.5
        This creates a decision boundary where output > 0.5 only when
        both x1 and x2 are close to 1.

    Args:
        optimizer: Choice of optimizer - 'sgd' or 'adam'
                   - SGD: Stochastic Gradient Descent (educational, simpler)
                   - Adam: Adaptive Moment Estimation (faster convergence)
        learning_rate: Learning rate for the optimizer (default: 0.01)

    Returns:
        Compiled Keras Sequential model ready for training

    Example:
        >>> model = create_and_gate_model(optimizer='adam', learning_rate=0.01)
        >>> model.summary()
        Model: "sequential"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #
        =================================================================
         dense (Dense)               (None, 1)                 3
        =================================================================
        Total params: 3
    """
    # Create sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,), name='input'),
        tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='output'
        )
    ], name='and_gate_model')

    # Select optimizer based on parameter
    if optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Use 'sgd' or 'adam'.")

    # Compile model with MSE loss and accuracy metric
    model.compile(
        optimizer=opt,
        loss='mse',  # Mean Squared Error
        metrics=['accuracy']
    )

    return model
