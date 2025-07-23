# -*- coding: utf-8 -*-
"""
Utility Module for the Multi-Step Mining Bayesian Optimization Project.
"""
import os
import json
import datetime
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence as skopt_plot_convergence

def setup_results_directory(target_case_name):
    """
    Creates a unique directory for storing optimization results for a specific case.
    """
    sanitized_name = target_case_name.replace('.', '_').replace(' ', '_')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{sanitized_name}_{timestamp}"
    results_dir = os.path.join("optimization_results", run_name)
    # The curves directory is kept for potential future use or for the convergence plot
    curves_dir = os.path.join(results_dir, "plots")
    os.makedirs(curves_dir, exist_ok=True)
    return results_dir, curves_dir

def save_best_parameters(params_dict, filepath):
    """Saves the best found parameters to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=4)
    print(f"Best parameters saved to: '{filepath}'")

def plot_convergence(result, filepath):
    """Plots and saves the optimization convergence plot."""
    try:
        skopt_plot_convergence(result)
        plt.title("Convergence Plot")
        plt.xlabel("Function Call")
        plt.ylabel("Minimum Loss Found")
        plt.grid(True)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Convergence plot saved to: '{filepath}'")
    except Exception as e:
        print(f"Could not generate convergence plot: {e}")

# Note: A per-iteration comparison plot is non-trivial for multi-step 3D data.
# It's recommended to write a separate post-processing script to visualize
# the results from the best parameter set found.
