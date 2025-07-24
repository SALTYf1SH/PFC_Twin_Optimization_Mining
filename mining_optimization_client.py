# -*- coding: utf-8 -*-
"""
==============================================================================
  Parallel Bayesian Optimization Client for Multi-Step Mining Process
==============================================================================

This client is designed to work with the multi-step PFC server. It optimizes
parameters by comparing the entire excavation process (multiple steps) against
a corresponding series of target data files.

REVISION 5: Fixed JSONDecodeError by making the response parsing more robust
            to handle potential extra data in the socket stream.
"""
import os
import json
import socket
import numpy as np # Import numpy to check for its types
from skopt import Optimizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import custom modules for this project ---
from loss_function import calculate_multi_step_loss
from utilities_mining import (setup_results_directory, save_best_parameters, plot_convergence)
from knowledge_base_manager_mining import (warm_start_optimizer, load_from_knowledge_base, save_to_knowledge_base)

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
# PFC Server Connection Settings
SERVER_LIST = [
    ('127.0.0.1', 50002),
    ('127.0.0.1', 50001),
]
CONNECTION_TIMEOUT = 20000

# Target Data Directory
TARGET_DATA_ROOT_DIR = "target_data"

# Bayesian Optimization Settings
N_CALLS = 50
N_INITIAL_POINTS = 5

# Transformation and Weighting for the Loss Function
SIM_TRANSFORM = {'x_shift': 125, 'y_shift': 80}
TARGET_TRANSFORM = {'x_scale': 100, 'y_scale': 100}
STEP_WEIGHTS = None

# Parameter Space
from skopt.space import Real, Integer

PARAMETER_SPACE = [
    Real(20e9, 60e9, name='key_emod000'),
    Real(1.5, 3.0, name='key_kratio'),
    Real(2.0e6, 8.0e6, name='key_ten_'),
    Real(2.0e6, 8.0e6, name='key_coh_'),
    Real(0.2, 0.6, name='key_fric'),
]

# =============================================================================
# --- 2. WORKER FUNCTION FOR PARALLEL EXECUTION ---
# =============================================================================

def run_simulation_worker(params_list, server, target_case_dir, job_id):
    """
    This function is executed by each thread. It manages one full simulation run.
    """
    params_dict = {p.name: v for p, v in zip(PARAMETER_SPACE, params_list)}

    # Step A: Convert numpy types to native Python types
    for key, value in params_dict.items():
        if isinstance(value, np.integer):
            params_dict[key] = int(value)
        elif isinstance(value, np.floating):
            params_dict[key] = float(value)

    # Step B: Format large numbers into scientific notation strings for PFC
    params_dict_for_kb = params_dict.copy()
    for key, value in params_dict_for_kb.items():
        if isinstance(value, (int, float)) and abs(value) >= 1e6:
            params_dict_for_kb[key] = f"{value:.6e}"

    print(f"[Job {job_id}] Testing parameters on {server[0]}:{server[1]}")

    # 1. Check knowledge base (cache) first
    sim_steps_json_string = load_from_knowledge_base(params_dict_for_kb)

    # 2. If not in cache, run the simulation on the PFC server
    if sim_steps_json_string is None:
        print(f"[Job {job_id}] Cache miss. Connecting to PFC server...")
        params_json_to_send = json.dumps(params_dict_for_kb)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(CONNECTION_TIMEOUT)
                s.connect(server)
                s.sendall(params_json_to_send.encode('utf-8'))

                fragments = []
                while True:
                    chunk = s.recv(1048576)
                    if not chunk: break
                    fragments.append(chunk)
                
                raw_response = b"".join(fragments).decode('utf-8').strip()

                # --- FIX: Robustly parse the JSON response ---
                # Find the first '{' and the last '}' to extract the main JSON object,
                # ignoring any potential leading/trailing garbage data.
                try:
                    start = raw_response.find('{')
                    end = raw_response.rfind('}')
                    if start != -1 and end != -1:
                        sim_steps_json_string = raw_response[start:end+1]
                        response_data = json.loads(sim_steps_json_string)
                        if "error" in response_data:
                            print(f"[Job {job_id}] FAILED: Server returned an error: {response_data['error']}")
                            sim_steps_json_string = None
                        else:
                            save_to_knowledge_base(params_dict_for_kb, sim_steps_json_string)
                    else:
                        raise json.JSONDecodeError("No valid JSON object found in response", raw_response, 0)
                except json.JSONDecodeError as e:
                    print(f"[Job {job_id}] FAILED: Could not parse JSON response from server. Error: {e}")
                    sim_steps_json_string = None
                # --- END FIX ---

        except Exception as e:
            print(f"[Job {job_id}] FAILED on server {server[0]}:{server[1]} with a network error: {e}")
            sim_steps_json_string = None

    # 3. Calculate loss using the multi-step loss function
    if sim_steps_json_string is None:
        print(f"[Job {job_id}] FAILED: No valid simulation data produced.")
        loss = 1e10
    else:
        loss = calculate_multi_step_loss(
            target_data_dir=target_case_dir,
            sim_steps_json_string=sim_steps_json_string,
            target_transform=TARGET_TRANSFORM,
            sim_transform=SIM_TRANSFORM,
            step_weights=STEP_WEIGHTS
        )
        print(f"[Job {job_id}] SUCCEEDED with Total Weighted Loss = {loss:.6f}")

    return params_list, loss, sim_steps_json_string, job_id

# =============================================================================
# --- 3. MAIN EXECUTION BLOCK ---
# =============================================================================

if __name__ == '__main__':
    if not os.path.isdir(TARGET_DATA_ROOT_DIR):
        print(f"[FATAL ERROR] Target data root directory not found: '{TARGET_DATA_ROOT_DIR}'")
        exit()

    target_cases = [d for d in os.listdir(TARGET_DATA_ROOT_DIR) if os.path.isdir(os.path.join(TARGET_DATA_ROOT_DIR, d))]
    if not target_cases:
        print(f"[FATAL ERROR] No target case subdirectories found in '{TARGET_DATA_ROOT_DIR}'")
        exit()

    print(f"Found {len(target_cases)} target case(s). Starting optimization for each: {target_cases}")

    for case_name in target_cases:
        target_case_dir = os.path.join(TARGET_DATA_ROOT_DIR, case_name)
        print(f"\n\n{'='*60}")
        print(f"===      STARTING OPTIMIZATION FOR CASE: {case_name}      ===")
        print(f"{'='*60}\n")

        results_dir, curves_dir = setup_results_directory(case_name)
        print(f"Results for this run will be saved in: '{results_dir}'")

        optimizer = Optimizer(dimensions=PARAMETER_SPACE, random_state=123)

        x_prior, y_prior = warm_start_optimizer(
            parameter_space=PARAMETER_SPACE,
            target_data_dir=target_case_dir,
            target_transform=TARGET_TRANSFORM,
            sim_transform=SIM_TRANSFORM,
            step_weights=STEP_WEIGHTS
        )
        if x_prior:
            optimizer.tell(x_prior, y_prior)
            print(f"Optimizer warm-started with {len(x_prior)} prior points.")

        num_workers = len(SERVER_LIST)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            
            n_initial = N_INITIAL_POINTS if not x_prior else num_workers
            initial_points = optimizer.ask(n_points=n_initial)
            
            futures = {
                executor.submit(run_simulation_worker, point, SERVER_LIST[i % num_workers], target_case_dir, i+1): point
                for i, point in enumerate(initial_points)
            }
            total_jobs_dispatched = len(futures)
            print(f"\n--- Dispatched initial batch of {len(futures)} jobs ---\n")

            while total_jobs_dispatched < N_CALLS:
                for future in as_completed(futures):
                    x_result, y_result, _, completed_job_id = future.result()
                    del futures[future]
                    
                    optimizer.tell([x_result], [y_result])
                    print(f"--- Optimizer updated with Job {completed_job_id}. Best loss so far: {min(optimizer.yi):.6f} ---")
                    
                    if total_jobs_dispatched < N_CALLS:
                        next_point = optimizer.ask()
                        server_for_next_job = SERVER_LIST[total_jobs_dispatched % num_workers]
                        total_jobs_dispatched += 1
                        
                        new_future = executor.submit(run_simulation_worker, next_point, server_for_next_job, target_case_dir, total_jobs_dispatched)
                        futures[new_future] = next_point
                        print(f"--- Dispatched new Job {total_jobs_dispatched} to {server_for_next_job[0]}:{server_for_next_job[1]} ---\n")
                    
                    break 

        result = optimizer.get_result()
        print("\n" + "-"*60)
        print(f"---           OPTIMIZATION COMPLETE: {case_name}           ---")
        print("-" * 60 + "\n")

        best_params_dict = {p.name: v for p, v in zip(PARAMETER_SPACE, result.x)}
        print(f"Minimum loss achieved: {result.fun:.6f}")
        print("Best parameter set found:")
        print(json.dumps(best_params_dict, indent=4))
        
        param_filepath = os.path.join(results_dir, "best_parameters.json")
        save_best_parameters(best_params_dict, param_filepath)
        
        convergence_filepath = os.path.join(results_dir, "convergence_plot.png")
        plot_convergence(result, convergence_filepath)

    print(f"\n\n{'='*60}")
    print("===            ALL TARGET CASES HAVE BEEN OPTIMIZED            ===")
    print("=" * 60)
