#!/usr/bin/env python3
"""
Main execution script for Neural Networks on Logic Gates.

This script orchestrates the complete training and evaluation pipeline for
AND and XOR gate neural networks, comparing SGD vs Adam optimizers.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
from src.data.clean_data import generate_and_gate_data, generate_xor_gate_data
from src.data.noisy_data import generate_noisy_and_gate_data, generate_noisy_xor_gate_data
from src.models.and_gate import create_and_gate_model
from src.models.xor_gate import create_xor_gate_model
from src.training.trainer import train_model, save_model, save_history
from src.evaluation.metrics import calculate_metrics, save_metrics, print_metrics_summary
from src.visualization.plots import plot_training_history, plot_confusion_matrix, plot_optimizer_comparison
from src.visualization.decision_boundary import plot_decision_boundary

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def run_experiment(gate_type, data_type, optimizer, epochs=2000):
    """Run a single experiment with specified configuration."""
    print(f"\n{'='*60}")
    print(f"Training {gate_type.upper()} Gate - {data_type} Data - {optimizer.upper()} Optimizer")
    print(f"{'='*60}")

    # Generate data
    if data_type == "clean":
        if gate_type == "and":
            X, y = generate_and_gate_data()
        else:
            X, y = generate_xor_gate_data()
        batch_size = 4
    else:  # noisy
        if gate_type == "and":
            X, y = generate_noisy_and_gate_data(n_samples=100, noise_std=0.2, random_seed=SEED)
        else:
            X, y = generate_noisy_xor_gate_data(n_samples=100, noise_std=0.2, random_seed=SEED)
        batch_size = 32

    # Create model
    if gate_type == "and":
        model = create_and_gate_model(optimizer=optimizer, learning_rate=0.01)
    else:
        model = create_xor_gate_model(optimizer=optimizer, learning_rate=0.01)

    # Train model
    history, train_time = train_model(model, X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate
    predictions = model.predict(X, verbose=0).flatten()
    metrics = calculate_metrics(y, predictions)

    # Save results
    base_name = f"{gate_type}_{data_type}_{optimizer}"
    save_model(model, f"models/{base_name}.keras")
    save_history(history, f"results/{base_name}_history.json", train_time)
    save_metrics(metrics, f"results/{base_name}_metrics.json")

    # Print summary
    print(f"Training Time: {train_time:.2f}s | Accuracy: {metrics['accuracy']:.4f}")

    return model, history, metrics, train_time, X, y


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Neural Networks for Logic Gates - Educational Implementation")
    print("="*60)
    results = {}

    # Part A: Clean Data Experiments
    print("\n\n### PART A: CLEAN DATA EXPERIMENTS ###\n")
    for gate in ["and", "xor"]:
        for opt in ["sgd", "adam"]:
            model, hist, metrics, time, X, y = run_experiment(gate, "clean", opt, epochs=2000)
            results[f"{gate}_clean_{opt}"] = {'model': model, 'history': hist.history,
                                                'metrics': metrics, 'time': time, 'X': X, 'y': y}

    # Part B: Noisy Data Experiments
    print("\n\n### PART B: NOISY DATA EXPERIMENTS ###\n")
    for gate in ["and", "xor"]:
        for opt in ["sgd", "adam"]:
            model, hist, metrics, time, X, y = run_experiment(gate, "noisy", opt, epochs=1000)
            results[f"{gate}_noisy_{opt}"] = {'model': model, 'history': hist.history,
                                                'metrics': metrics, 'time': time, 'X': X, 'y': y}

    # Generate visualizations
    print("\n\n### GENERATING VISUALIZATIONS ###\n")
    for key, data in results.items():
        gate, dtype, opt = key.split('_')
        title = f"{gate.upper()} Gate - {dtype.capitalize()} - {opt.upper()}"
        plot_training_history(data['history'], title, f"results/plots/{key}_history.png")
        plot_confusion_matrix(np.array(data['metrics']['confusion_matrix']), title,
                              f"results/plots/{key}_cm.png")

    # Decision boundaries (clean data only)
    for gate in ["and", "xor"]:
        key = f"{gate}_clean_adam"
        plot_decision_boundary(results[key]['model'], results[key]['X'], results[key]['y'],
                               f"{gate.upper()} Gate - ADAM", f"results/plots/{key}_boundary.png")

    # Optimizer comparisons
    for gate in ["and", "xor"]:
        for dtype in ["clean", "noisy"]:
            histories = {'sgd': results[f"{gate}_{dtype}_sgd"]['history'],
                        'adam': results[f"{gate}_{dtype}_adam"]['history']}
            plot_optimizer_comparison(histories, 'loss', f"{gate.upper()} Gate - {dtype.capitalize()}",
                                     f"results/plots/{gate}_{dtype}_optimizer_comparison.png")

    # Final summary
    print("\n\n### FINAL SUMMARY ###\n")
    print(f"{'Gate':<8} {'Data':<8} {'Opt':<6} {'Accuracy':<10} {'Time (s)':<10}")
    print("="*50)
    for key in sorted(results.keys()):
        gate, dtype, opt = key.split('_')
        print(f"{gate.upper():<8} {dtype:<8} {opt.upper():<6} "
              f"{results[key]['metrics']['accuracy']:<10.4f} {results[key]['time']:<10.2f}")

    print("\n✓ All experiments completed successfully!")
    print("✓ Results saved to models/ and results/ directories")
    print("✓ Visualizations saved to results/plots/ directory\n")


if __name__ == "__main__":
    main()
