# -*- coding: utf-8 -*-
"""
==============================================================================
  Integrated PFC Server for Staged Excavation Parameter Optimization
==============================================================================

This script combines a robust, crash-proof socket server with a detailed,
multi-stage simulation logic.

REVISION 7: Added functionality to save the parameters used for each run
            into a 'parameters.txt' file within the corresponding
            archived experiment folder.
"""

# ==============================================================================
#  * * * USER CONFIGURATION AREA (Please modify here) * * *
# ==============================================================================
import sys
import os

# !!! IMPORTANT !!!
# Please change the path below to the absolute path of your project folder.
# Use forward slashes "/"
PROJECT_DIRECTORY = "F:\PFCprj\PFC_Twin_Optimization\PFC_model"

# ==============================================================================
#  * * * PATH VALIDATION AND SETUP * * *
# ==============================================================================
if "222" in PROJECT_DIRECTORY:
    print("\n" + "="*80)
    print("[FATAL ERROR] Project path has not been set in the script.")
    print("Please open this script and modify the 'PROJECT_DIRECTORY' variable.")
    print("="*80)
    sys.exit(1)
if not os.path.isdir(PROJECT_DIRECTORY):
    print(f"\n[FATAL ERROR] The specified project path does not exist: '{PROJECT_DIRECTORY}'")
    sys.exit(1)
if PROJECT_DIRECTORY not in sys.path:
    sys.path.insert(0, PROJECT_DIRECTORY)
print(f"[INFO] Project directory successfully set to: '{PROJECT_DIRECTORY}'")
# ==============================================================================

import json
import socket
import csv
import hashlib
import shutil
import time
import itasca
from itasca import ball, wall

# --- Dependency Check ---
try:
    from utils import (run_dat_file, delete_balls_outside_area, fenceng,
                       plot_y_displacement_heatmap)
    print("[INFO] Successfully imported functions from 'utils.py'.")
except ImportError as e:
    print(f"\n[FATAL ERROR] Could not import 'utils.py' from the project path.")
    print(f"Please ensure 'utils.py' is located in: '{PROJECT_DIRECTORY}'")
    print(f"Error details: {e}")
    sys.exit(1)

# ==============================================================================
# 0. GLOBAL DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_CONFIG = {
    "DETERMINISTIC_MODE": True, "MODEL_WIDTH": 250.0,
    "ROCK_LAYER_THICKNESSES": [
        20.50, 3.00, 7.50, 30.00, 8.00, 7.50, 6.50, 6.00, 4.50, 7.00,
        9.00, 6.50, 6.50, 7.50, 2.00, 9.00, 6.00, 3.00, 3.50, 5.50
    ], "LEFT_PILLAR_WIDTH": 45.0, "RIGHT_PILLAR_WIDTH": 45.0,
    "EXCAVATION_STEP_WIDTH": 10.0, "EXCAVATION_LAYER_GROUP": '11',
    "EQUILIBRIUM_PARAMS_LIST": [
        ('pb_modules', 1e9), ('emod000', 15e9), ('ten_', 0.75e6), ('coh_', 0.75e6),
        ('fric', 0.1), ('kratio', 2.0), ('key_pb_modules', 3e9), ('key_emod000', 45e9),
        ('key_ten_', 4.5e6), ('key_coh_', 4.5e6), ('key_fric', 0.3), ('key_kratio', 2.0),
        ('pb_modules_1', 1e8), ('emod111', 1e8), ('ten1_', 1e5), ('coh1_', 1e5),
        ('fric1', 0.1), ('kratio', 2.0), ('dpnr', 0.5), ('dpsr', 0.0),
    ], "SOLVE_CYCLES_PER_STEP": 8000, "SOLVE_RATIO_TARGET": 1e-5
}

# ==============================================================================
# 1. SIMULATION WORKFLOW FUNCTIONS
# ==============================================================================
def setup_temporary_environment(base_path, run_hash):
    run_path = os.path.join(base_path, run_hash)
    paths = {
        "root": run_path,
        "sav": os.path.join(run_path, "sav"),
        "mat": os.path.join(run_path, "mat"),
        "img": os.path.join(run_path, "img"),
        "csv": os.path.join(run_path, "csv")
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def calculate_geology(config):
    thicknesses = config["ROCK_LAYER_THICKNESSES"].copy()
    model_height = sum(thicknesses)
    thicknesses.reverse()
    cumulative_heights = [round(sum(thicknesses[:i+1]), 4) for i in range(len(thicknesses))]
    return cumulative_heights, model_height

def run_stage_one_generation(config, paths):
    save_file = os.path.join(paths["root"], "yuya.sav")
    itasca.command("model new")
    itasca.set_deterministic(config["DETERMINISTIC_MODE"])
    dat_path = os.path.join(PROJECT_DIRECTORY, "yuya-new.dat")
    run_dat_file(dat_path)
    delete_balls_outside_area(
        x_min=wall.find('boxWallLeft4').pos_x(), x_max=wall.find('boxWallRight2').pos_x(),
        y_min=wall.find('boxWallBottom1').pos_y(), y_max=wall.find('boxWallTop3').pos_y()
    )
    itasca.command(f"model save '{save_file}'")

def run_stage_two_equilibrium(config, layer_array, paths):
    save_file = os.path.join(paths["root"], "jiaojie.sav")
    initial_save_file = os.path.join(paths["root"], "yuya.sav")
    itasca.command(f"model restore '{initial_save_file}'")
    fenceng(layer_array=layer_array)
    fenceng_temp_file = os.path.join(paths["root"], "fenceng_temp.sav")
    itasca.command(f"model save '{fenceng_temp_file}'")
    itasca.command(f"model restore '{fenceng_temp_file}'")
    for name, value in config["EQUILIBRIUM_PARAMS_LIST"]:
        itasca.fish.set(name, value)
    dat_path = os.path.join(PROJECT_DIRECTORY, "jiaojie.dat")
    run_dat_file(dat_path)
    itasca.command(f"model save '{save_file}'")

def run_excavation_and_collect_data(config, paths):
    print("  [Sim] Running multi-step excavation and data collection...")
    start_x = config["LEFT_PILLAR_WIDTH"] - (config["MODEL_WIDTH"] / 2.0)
    end_x = (config["MODEL_WIDTH"] / 2.0) - config["RIGHT_PILLAR_WIDTH"]
    step_width = config["EXCAVATION_STEP_WIDTH"]
    num_steps = int((end_x - start_x) / step_width)
    all_steps_data = {}

    for i in range(num_steps):
        step_key = f"step_{i}"
        print(f"    -> Executing {step_key}...")
        excavation_pos = start_x + i * step_width
        excavation_end = excavation_pos + step_width
        cmd = f"ball delete range group '{config['EXCAVATION_LAYER_GROUP']}' pos-x {excavation_pos} {excavation_end}"
        itasca.command(cmd)
        itasca.command(f"model solve cycle {config['SOLVE_CYCLES_PER_STEP']} or ratio-average {config['SOLVE_RATIO_TARGET']}")
        
        step_name = f"excavation_face_{excavation_end:.2f}"
        plot_y_displacement_heatmap(
            window_size=itasca.fish.get('rdmax') * 2,
            model_width=config["MODEL_WIDTH"],
            model_height=160,
            name=step_name,
            interpolate='nearest',
            resu_path=paths["root"]
        )
        
        data_filename = f"resampled_displacement_{step_name}.csv"
        data_filepath = os.path.join(paths["csv"], data_filename)

        if os.path.exists(data_filepath):
            with open(data_filepath, 'r') as f:
                csv_content = f.read()
            all_steps_data[step_key] = csv_content
            print(f"    -> Collected data for {step_key}.")
        else:
            print(f"    -> [WARN] Data file not found for {step_key} at {data_filepath}")
            all_steps_data[step_key] = ""

    print("  [Sim] Multi-step excavation and data collection complete.")
    return all_steps_data

# ==============================================================================
# 2. CORE SIMULATION DRIVER
# ==============================================================================
def _run_single_optimization_cycle(client_params):
    itasca.command("python-reset-state true")

    param_string = str(sorted(client_params.items()))
    run_hash = hashlib.md5(param_string.encode('utf-8')).hexdigest()[:12]
    
    run_config = json.loads(json.dumps(DEFAULT_CONFIG))
    current_params = dict(run_config["EQUILIBRIUM_PARAMS_LIST"])
    for key, value in client_params.items():
        # The value from client might be a string (e.g., "1.23e+10"),
        # so we convert it to a float for PFC.
        try:
            current_params[key] = float(value)
        except (ValueError, TypeError):
            current_params[key] = value # Keep as is if conversion fails
    run_config["EQUILIBRIUM_PARAMS_LIST"] = list(current_params.items())

    temp_base_path = os.path.join(PROJECT_DIRECTORY, "_temp_server_runs")
    paths = setup_temporary_environment(temp_base_path, run_hash)
    
    # --- NEW: Save parameters to a text file inside the temp folder ---
    try:
        param_file_path = os.path.join(paths["root"], "parameters.txt")
        with open(param_file_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Simulation Parameters for Run: {run_hash} ---\n\n")
            f.write("Parameters received from client:\n")
            # Use the original client_params for this section for clarity
            for key, value in client_params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n\n--- Full FISH Configuration (after type conversion) ---\n")
            # The run_config has the full set of parameters ready for FISH
            for key, value in run_config["EQUILIBRIUM_PARAMS_LIST"]:
                 f.write(f"  {key}: {value}\n")
        print(f"  [Info] Parameters for this run saved to {param_file_path}")
    except Exception as e:
        print(f"  [WARN] Could not save parameters.txt file. Error: {e}")
    # --- END NEW ---

    try:
        itasca.command("model new")
        layer_array, _ = calculate_geology(run_config)
        run_stage_one_generation(run_config, paths)
        run_stage_two_equilibrium(run_config, layer_array, paths)
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")
        
        all_steps_data = run_excavation_and_collect_data(run_config, paths)

        if all_steps_data:
            return json.dumps(all_steps_data)
        else:
            return json.dumps({"error": "No data was collected during simulation."})

    except Exception as e:
        import traceback
        print(f"\n[FATAL SIMULATION ERROR] Run failed: {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Simulation failed with exception: {e}"})
    finally:
        itasca.command("model new")
        time.sleep(0.5)
        
        # Archive the results instead of deleting
        try:
            experiments_dir = os.path.join(PROJECT_DIRECTORY, "experiments")
            os.makedirs(experiments_dir, exist_ok=True)
            destination_path = os.path.join(experiments_dir, run_hash)
            
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
                
            shutil.move(paths["root"], destination_path)
            print(f"  [Archive] Successfully moved results to: {destination_path}")
        except Exception as e:
            print(f"  [Archive WARN] Could not archive results folder. Deleting it instead. Error: {e}")
            shutil.rmtree(paths["root"], ignore_errors=True)

# ==============================================================================
# 3. SOCKET SERVER
# ==============================================================================
def start_server(host='127.0.0.1', port=50002):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen()
        print("\n" + "="*70)
        print(f" PFC Multi-Step Optimization Server is RUNNING on {host}:{port}")
        print(" -> Archiving of experiment results is ENABLED.")
        print("="*70)
        
        while True:
            print("\n[Server] Waiting for a client connection...")
            conn, addr = server_socket.accept()
            print(f"[Server] Accepted connection from {addr}")
            with conn:
                data = conn.recv(8192)
                if not data:
                    print("[Server] Connection closed by client without data.")
                    continue

                client_params = json.loads(data.decode('utf-8'))
                print(f"[Server] Received parameters: {client_params}")
                
                results_json_string = _run_single_optimization_cycle(client_params)
                
                print("[Server] Simulation cycle finished. Sending multi-step JSON results...")
                conn.sendall(results_json_string.encode('utf-8'))
                print("[Server] Results sent. Closing connection.")

    except Exception as e:
        print(f"[Server FATAL ERROR] An error in the server loop forced shutdown: {e}")
    finally:
        print("\n--- Server is shutting down. ---")
        server_socket.close()

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    HOST_IP = '127.0.0.1'
    SERVER_PORT = 50002
    start_server(host=HOST_IP, port=SERVER_PORT)
