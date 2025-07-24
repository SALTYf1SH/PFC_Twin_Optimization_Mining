# -*- coding: utf-8 -*-
"""
==============================================================================
  Robust Parallel Bayesian Optimization Client for Multi-Step Mining Process
==============================================================================

REVISION 5: Implemented a robust server pool management system. This version
            can handle server failures and varied job completion times (e.g.,
            cache hits) without stalling or crashing, ensuring true parallel
            utilization of all available resources.
"""
import os
import json
import socket
import queue
import threading
import numpy as np
from skopt import Optimizer
from concurrent.futures import ThreadPoolExecutor, as_completed

from loss_function import calculate_multi_step_loss
from utilities_mining import (setup_results_directory, save_best_parameters, plot_convergence)
from knowledge_base_manager_mining import (warm_start_optimizer, load_from_knowledge_base, save_to_knowledge_base)

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
SERVER_LIST = [
    ('127.0.0.1', 50001),
    ('127.0.0.1', 50002)
]
CONNECTION_TIMEOUT = 20000
TARGET_DATA_ROOT_DIR = "target_data"
N_CALLS = 20
N_INITIAL_POINTS = 5
SIM_TRANSFORM = {'x_shift': 125, 'y_shift': 80}
TARGET_TRANSFORM = {'x_scale': 100, 'y_scale': 100}
STEP_WEIGHTS = None

from skopt.space import Real
PARAMETER_SPACE = [
    Real(20e9, 60e9, name='key_emod000'),
    Real(1.5, 3.0, name='key_kratio'),
    Real(2.0e6, 8.0e6, name='key_ten_'),
    Real(2.0e6, 8.0e6, name='key_coh_'),
    Real(0.2, 0.6, name='key_fric'),
]

# =============================================================================
# --- 2. ROBUST SERVER POOL MANAGER ---
# =============================================================================
class ServerPool:
    """A thread-safe pool to manage available and dead PFC servers."""
    def __init__(self, server_list):
        self.server_queue = queue.Queue()
        self.lock = threading.Lock()
        self.all_servers = list(server_list)
        self.dead_servers = set()
        for server in server_list:
            self.server_queue.put(server)

    def get_server(self):
        return self.server_queue.get()

    def return_server(self, server):
        self.server_queue.put(server)

    def mark_as_dead(self, server):
        with self.lock:
            if server not in self.dead_servers:
                print(f"!!! [Server Pool] Server {server} marked as DEAD and removed from rotation. !!!")
                self.dead_servers.add(server)

    def all_dead(self):
        with self.lock:
            return len(self.dead_servers) == len(self.all_servers)

# =============================================================================
# --- 3. WORKER FUNCTION ---
# =============================================================================
def run_simulation_worker(params_list, server_pool, target_case_dir, job_id):
    """
    Worker function that gets a server from the pool, runs a simulation,
    and returns the server to the pool.
    """
    params_dict = {p.name: v for p, v in zip(PARAMETER_SPACE, params_list)}
    for key, value in params_dict.items():
        if isinstance(value, np.number):
            params_dict[key] = value.item()

    params_dict_for_kb = params_dict.copy()
    for key, value in params_dict_for_kb.items():
        if isinstance(value, (int, float)) and abs(value) >= 1e6:
            params_dict_for_kb[key] = f"{value:.6e}"

    sim_steps_json_string = load_from_knowledge_base(params_dict_for_kb)
    
    # If it's a cache hit, we don't need to use a server at all.
    if sim_steps_json_string is not None:
        print(f"[Job {job_id}] Cache hit. Processing locally.")
    else:
        server = None
        try:
            server = server_pool.get_server()
            print(f"[Job {job_id}] Cache miss. Using server {server} for parameters...")
            
            params_json_to_send = json.dumps(params_dict_for_kb)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(CONNECTION_TIMEOUT)
                s.connect(server)
                s.sendall(params_json_to_send.encode('utf-8'))
                fragments = [s.recv(1048576) for _ in range(100) if s.recv(1048576)] # Simplified recv loop
                sim_steps_json_string = b"".join(fragments).decode('utf-8')

                response_data = json.loads(sim_steps_json_string)
                if "error" in response_data:
                    sim_steps_json_string = None
                else:
                    save_to_knowledge_base(params_dict_for_kb, sim_steps_json_string)
        except (socket.timeout, ConnectionRefusedError, ConnectionResetError) as e:
            print(f"[Job {job_id}] FAILED on server {server} with critical network error: {e}")
            if server:
                server_pool.mark_as_dead(server)
            server = None # Mark server as None so it's not returned to the pool
            sim_steps_json_string = None
        finally:
            if server:
                server_pool.return_server(server)

    if sim_steps_json_string is None:
        loss = 1e10
    else:
        loss = calculate_multi_step_loss(
            target_data_dir=target_case_dir,
            sim_steps_json_string=sim_steps_json_string,
            target_transform=TARGET_TRANSFORM,
            sim_transform=SIM_TRANSFORM,
            step_weights=STEP_WEIGHTS
        )
    print(f"[Job {job_id}] Completed. Loss = {loss:.6f}")
    return params_list, loss, job_id

# =============================================================================
# --- 4. MAIN EXECUTION BLOCK ---
# =============================================================================
if __name__ == '__main__':
    target_cases = [d for d in os.listdir(TARGET_DATA_ROOT_DIR) if os.path.isdir(os.path.join(TARGET_DATA_ROOT_DIR, d))]
    
    for case_name in target_cases:
        target_case_dir = os.path.join(TARGET_DATA_ROOT_DIR, case_name)
        print(f"\n\n{'='*60}\n=== STARTING OPTIMIZATION FOR CASE: {case_name} ===\n{'='*60}\n")
        results_dir, _ = setup_results_directory(case_name)

        optimizer = Optimizer(dimensions=PARAMETER_SPACE, random_state=123)
        server_pool = ServerPool(SERVER_LIST)

        x_prior, y_prior = warm_start_optimizer(
            PARAMETER_SPACE, target_case_dir, TARGET_TRANSFORM, SIM_TRANSFORM, STEP_WEIGHTS
        )
        if x_prior:
            optimizer.tell(x_prior, y_prior)

        with ThreadPoolExecutor(max_workers=len(SERVER_LIST)) as executor:
            n_initial = N_INITIAL_POINTS if not x_prior else len(SERVER_LIST)
            initial_points = optimizer.ask(n_points=n_initial)
            
            futures = {
                executor.submit(run_simulation_worker, point, server_pool, target_case_dir, i+1): point
                for i, point in enumerate(initial_points)
            }
            jobs_in_progress = len(futures)
            total_jobs_submitted = jobs_in_progress

            for future in as_completed(futures):
                x_result, y_result, completed_job_id = future.result()
                
                optimizer.tell([x_result], [y_result])
                print(f"--- Optimizer updated with Job {completed_job_id}. Best loss so far: {min(optimizer.yi):.6f} ---")

                if server_pool.all_dead():
                    print("[FATAL] All PFC servers are offline. Aborting optimization.")
                    break

                if total_jobs_submitted < N_CALLS:
                    next_point = optimizer.ask()
                    total_jobs_submitted += 1
                    print(f"--- Submitting new Job {total_jobs_submitted} ---")
                    executor.submit(run_simulation_worker, next_point, server_pool, target_case_dir, total_jobs_submitted)

        result = optimizer.get_result()
        print(f"\n{'-'*60}\n--- OPTIMIZATION COMPLETE: {case_name} ---\n{'-'*60}\n")
        best_params_dict = {p.name: v for p, v in zip(PARAMETER_SPACE, result.x)}
        print(f"Minimum loss achieved: {result.fun:.6f}")
        print("Best parameter set found:", json.dumps(best_params_dict, indent=4))
        
        save_best_parameters(best_params_dict, os.path.join(results_dir, "best_parameters.json"))
        plot_convergence(result, os.path.join(results_dir, "convergence_plot.png"))

    print(f"\n\n{'='*60}\n=== ALL TARGET CASES HAVE BEEN OPTIMIZED ===\n{'='*60}")
s