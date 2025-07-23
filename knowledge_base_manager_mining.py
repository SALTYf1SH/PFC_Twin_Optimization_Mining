# -*- coding: utf-8 -*-
"""
Knowledge Base Manager for the Multi-Step Mining Optimization.
This version stores and retrieves the full JSON object containing all step data.

REVISION 2: Fixed a TypeError by converting string parameters from the
            knowledge base back to floats before feeding them to the optimizer.
"""
import os
import json
import hashlib
from loss_function import calculate_multi_step_loss

KNOWLEDGE_BASE_DIR = "knowledge_base_mining"

def ensure_kb_directory():
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

def get_params_hash(params_dict):
    # This function expects native python types or string-formatted numbers.
    # The conversion is handled in the main client script.
    sorted_params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.sha256(sorted_params_str.encode('utf-8')).hexdigest()

def save_to_knowledge_base(params_dict, sim_steps_json_string):
    """Saves the parameters (with string-formatted numbers) and the resulting multi-step JSON data."""
    param_hash = get_params_hash(params_dict)
    filepath = os.path.join(KNOWLEDGE_BASE_DIR, f"{param_hash}.json")
    
    data_to_save = {
        'parameters': params_dict,
        'simulation_data': json.loads(sim_steps_json_string)
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"  -> Result saved to knowledge base: {param_hash[:10]}...")

def load_from_knowledge_base(params_dict):
    """
    Tries to load a result. Returns the multi-step JSON string if found, else None.
    """
    param_hash = get_params_hash(params_dict)
    filepath = os.path.join(KNOWLEDGE_BASE_DIR, f"{param_hash}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  -> Cache hit! Loaded result from knowledge base: {param_hash[:10]}...")
        return json.dumps(data.get('simulation_data'))
    return None

def warm_start_optimizer(parameter_space, target_data_dir, target_transform, sim_transform, step_weights):
    """
    Loads all existing knowledge, calculates loss against the current target,
    and prepares prior points (x, y) for the optimizer.
    """
    print("\n--- Initializing Optimizer with Prior Knowledge ---")
    ensure_kb_directory()
    x0_prior, y0_prior = [], []
    param_names = [p.name for p in parameter_space]
    kb_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".json")]

    if not kb_files:
        print("No prior knowledge found. Starting with random exploration.")
        return [], []

    for i, filename in enumerate(kb_files):
        print(f"  Processing prior point {i+1}/{len(kb_files)}...", end='\r')
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            params_dict = data['parameters']
            
            # --- FIX: Convert all parameter values back to float for skopt ---
            # This prevents the TypeError during the optimizer.tell() call.
            params_list = [float(params_dict.get(name)) for name in param_names]
            # --- END FIX ---
            
            sim_steps_json_string = json.dumps(data.get('simulation_data'))
            
            if sim_steps_json_string and all(p is not None for p in params_list):
                loss = calculate_multi_step_loss(
                    target_data_dir, sim_steps_json_string, target_transform, sim_transform, step_weights
                )
                if loss < 1e9:
                    x0_prior.append(params_list)
                    y0_prior.append(loss)
        except Exception:
            continue

    print(f"\nLoaded and processed {len(x0_prior)} valid prior data points.")
    return x0_prior, y0_prior
