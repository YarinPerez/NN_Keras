"""
XOR gate neural network model.

This module implements a minimal multi-layer perceptron for solving the XOR gate
problem. The XOR gate is NOT linearly separable, requiring at least one hidden
layer with 2+ neurons.
"""

import tensorflow as tf
from typing import Literal


def create_xor_gate_model(
    optimizer: Literal['sgd', 'adam'] = 'adam',
    learning_rate: float = 0.01,
    hidden_neurons: int = 2
) -> tf.keras.Model:
    """
    Create a multi-layer perceptron model for the XOR gate problem.

    Architecture:
        Input Layer: 2 neurons (for 2 binary inputs)
        Hidden Layer: 2 neurons with ReLU activation
        Output Layer: 1 neuron with sigmoid activation
        Total Parameters: ~9 (2*2 + 2 + 2*1 + 1)

    Why Hidden Layer is Necessary:
        The XOR gate is NOT linearly separable. No single straight line can
        separate the positive cases (0,1) and (1,0) from the negative cases
        (0,0) and (1,1) in 2D space.

        Historical Context:
            Minsky & Papert (1969) proved that single-layer perceptrons cannot
            solve XOR, leading to the first "AI winter". Multi-layer networks
            with backpropagation (popularized in 1986) solved this limitation.

        How It Works:
            The hidden layer transforms the input space into a higher-dimensional
            space where the problem becomes linearly separable. With 2 hidden
            neurons, the network effectively learns:
                - Neuron 1: Detect "one input is high" (OR-like)
                - Neuron 2: Detect "both inputs are high" (AND-like)
                - Output: Combine these to compute XOR = OR AND (NOT AND)

    Args:
        optimizer: Choice of optimizer - 'sgd' or 'adam'
                   - SGD: Stochastic Gradient Descent (educational, simpler)
                   - Adam: Adaptive Moment Estimation (faster convergence)
        learning_rate: Learning rate for the optimizer (default: 0.01)
        hidden_neurons: Number of neurons in hidden layer (default: 2, minimum)

    Returns:
        Compiled Keras Sequential model ready for training

    Example:
        >>> model = create_xor_gate_model(optimizer='adam', learning_rate=0.01)
        >>> model.summary()
        Model: "sequential"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #
        =================================================================
         hidden (Dense)              (None, 2)                 6
         output (Dense)              (None, 1)                 3
        =================================================================
        Total params: 9
    """
    # Create sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,), name='input'),
        tf.keras.layers.Dense(
            units=hidden_neurons,
            activation='relu',
            name='hidden'
        ),
        tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='output'
        )
    ], name='xor_gate_model')

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
