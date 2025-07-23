# -*- coding: utf-8 -*-
"""
loss_function.py

This module provides loss functions for the PFC mining optimization project.
This version is robust to missing target data files, allowing for comparison
of incomplete experimental data sets.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def _process_displacement_data(data_source, x_shift=0, y_shift=0, x_scale=1, y_scale=1):
    """
    Internal helper to read, transform, and normalize displacement data.
    """
    try:
        data_matrix = pd.read_csv(data_source, index_col=0)
        data_melted = data_matrix.melt(var_name='X_m', value_name='Displacement', ignore_index=False).reset_index()
        data_melted['X_m'] = data_melted['X_m'].astype(float)
        data_melted.rename(columns={data_melted.columns[0]: 'Y_m'}, inplace=True)

        points = data_melted[['X_m', 'Y_m']].values
        values = data_melted['Displacement'].values

        if x_scale != 1 or y_scale != 1:
            points[:, 0] *= x_scale
            points[:, 1] *= y_scale
        if x_shift != 0 or y_shift != 0:
            points[:, 0] += x_shift
            points[:, 1] += y_shift

        max_abs_val = np.max(np.abs(values))
        normalized_values = values / max_abs_val if max_abs_val > 1e-9 else values
        return points, normalized_values
    except Exception as e:
        print(f"  [Loss Helper ERROR] Failed to process data source: {e}")
        return None, None

def _calculate_single_step_loss(target_field_path, sim_field_csv_string, target_transform, sim_transform):
    """
    Calculates the RMSE loss for a single step of the simulation.
    """
    target_points, target_norm_values = _process_displacement_data(
        target_field_path, **target_transform
    )
    if target_points is None:
        raise ValueError(f"Failed to process target file: {target_field_path}")

    sim_data_buffer = io.StringIO(sim_field_csv_string)
    sim_points, sim_norm_values = _process_displacement_data(
        sim_data_buffer, **sim_transform
    )
    if sim_points is None:
        raise ValueError("Failed to process simulation data string.")

    sim_norm_values_aligned = griddata(
        points=sim_points,
        values=sim_norm_values,
        xi=target_points,
        method='linear',
        fill_value=0
    )

    return np.sqrt(np.mean((target_norm_values - sim_norm_values_aligned)**2))

def calculate_multi_step_loss(target_data_dir, sim_steps_json_string,
                              target_transform=None, sim_transform=None, step_weights=None):
    """
    Calculates a total, weighted loss across multiple excavation steps,
    robustly handling missing target data files.
    """
    try:
        sim_steps_data = json.loads(sim_steps_json_string)
        sim_steps_keys = sorted(sim_steps_data.keys())

        total_loss = 0.0
        total_weight = 0.0 # Keep track of the sum of weights for averaging
        target_params = target_transform if target_transform else {}
        sim_params = sim_transform if sim_transform else {}
        
        print(f"  [Loss] Comparing {len(sim_steps_keys)} simulation steps...")

        for step_key in sim_steps_keys:
            # Construct the expected target filename based on the simulation step key
            # Example: sim_key 'step_3' -> target_filename 'step_3.csv'
            target_filename = f"{step_key}.csv"
            target_file_path = os.path.join(target_data_dir, target_filename)

            # --- KEY IMPROVEMENT: Check if the target file exists ---
            if os.path.exists(target_file_path):
                print(f"    -> Found target '{target_filename}'. Calculating loss for {step_key}...")
                
                sim_csv_string = sim_steps_data[step_key]
                if not sim_csv_string:
                    print(f"    -> [WARN] Simulation data for {step_key} is empty. Skipping.")
                    continue

                step_loss = _calculate_single_step_loss(
                    target_file_path, sim_csv_string, target_params, sim_params
                )

                if np.isnan(step_loss):
                    print(f"    -> [WARN] Loss for {step_key} is NaN. Skipping.")
                    continue

                weight = step_weights.get(step_key, 1.0) if step_weights else 1.0
                total_loss += step_loss * weight
                total_weight += weight
                print(f"    -> Step Loss: {step_loss:.6f}, Weight: {weight:.2f}, Weighted Loss Added: {step_loss * weight:.6f}")
            else:
                # If the target file doesn't exist, just print a message and skip it
                print(f"    -> Target '{target_filename}' not found. Skipping step {step_key}.")

        # If no steps were compared at all, return a large penalty
        if total_weight == 0:
            print("  [Loss WARN] No matching target files were found for any simulation step. Returning penalty.")
            return 1e10
        
        # Return the average weighted loss
        return total_loss / total_weight

    except Exception as e:
        print(f"  [Loss ERROR] An unexpected error occurred during multi-step loss calculation: {e}")
        return 1e10
