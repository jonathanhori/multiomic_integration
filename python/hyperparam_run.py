import os
import sys


import itertools

# import torch
import pyro
# from pyro.optim import Adam, ClippedAdam
# from pyro.infer import Trace_ELBO

from pathlib import Path
# import time

from python.run_methods import evaluate_fitted_model, load_data, process_data, train_model, train_model_locally

# sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")

from data_utils import load_and_process_rds_data_for_condition
# from model import SupMultiviewDecomp
# from handler import ModelHandler
# from constants import Sites, Params


train_sim_data_path = "~/multiomic_integration/sim/data/train/"
test_sim_data_path = "~/multiomic_integration/sim/data/test/"

# Absolute paths to directories containing dataset replicates
dirs = [os.path.join(train_sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(train_sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(train_sim_data_path, dir)))]
test_dirs = [os.path.join(test_sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(test_sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(test_sim_data_path, dir)))]

filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in dirs}
test_filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in test_dirs}

file_dir_base = "n{}p{}_snr{}.{}_sparse{}" #/sim_data_ywithview_rep{}.rds"
file_name_base = "sim_data_ywithview_rep{}"

model_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/models")
model_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

metric_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/metrics")
metric_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

# Create output paths if don't exist
Path(model_out_path).mkdir(parents=True, exist_ok=True)
Path(metric_out_path).mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123
TRAINING_SPLIT = False
TRAINING_SIZE = 0.8

n_array = (50, 100, 500) #, 1000)
p_array = (50, 100, 1000)
snr_x_array = [2] #(2, 1, 0.5)
snr_y_array = [2] #(2, 1, 0.5)
reps = 10
loading_sparsity = [0, 0.25, 0.5]
k_deltas = [0, 10]

sim_grid = itertools.product(
    n_array,
    p_array,
    snr_x_array,
    snr_y_array,
    # range(reps),
    loading_sparsity,
    k_deltas
    # [loading_sparsity]
)

N_POSTERIOR_SAMPLES = 500
# TOL = 1
# VARIATIONAL_TOL = 0.3
# MINIBATCH_SIZE = 32
# MINIBATCH_SIZE_LOW = 16
# MIN_EPOCHS = 100
# MIN_EPOCHS_HIGH = 500
# MIN_EPOCHS_LOCAL = 100
# NUM_EPOCHS = 1000
# initial_lr = 0.005

# gamma = 0.1  # final learning rate will be gamma * initial_lr
# lrd = gamma ** (1 / (NUM_EPOCHS * MINIBATCH_SIZE))

train_config = {
    'initial_lr': 0.1,
    'betas': (0.0, 0.999),
    'num_particles': 1,
    'lr_step_size': 20,
    'lr_decay_factor': 0.5,
    'minibatch_size': 32,
    'min_epochs': 20,
    'max_epochs': 500
}

metric_list = []

if __name__ == "__main__":
    for n, p, snr_x, snr_y, sparsity, k_delta in sim_grid:
        pyro.clear_param_store()
        
        print("Loading:")
        print(file_dir_base.format(n, p, snr_x, snr_y, sparsity))
        
        # Determine if all reps are already done - skip training
        check_model_out_filename = os.path.join(
            model_out_path, 
            model_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, reps))
        if os.path.isfile(check_model_out_filename + ".pth"):
            print('Model already trained, moving on')
            continue
        
        # Load data
        # file_name_base.format(n, p, snr_x, snr_y, sparsity) #, rep)
        files_for_condition = load_and_process_rds_data_for_condition(
            file_dir_base.format(n, p, snr_x, snr_y, sparsity), # "n100p100_snr2.2_sparse0", 
            train_sim_data_path, # loaded above
            filename_dict, # loaded above
            reps) # how many reps to import
        if not TRAINING_SPLIT:
            test_files_for_condition = load_and_process_rds_data_for_condition(
                file_dir_base.format(n, p, snr_x, snr_y, sparsity), # "n100p100_snr2.2_sparse0", 
                test_sim_data_path, # loaded above
                test_filename_dict, # loaded above
                reps) # how many reps to import
        
        # print("Files for sim configuration:")
        # print(files_for_condition.keys())

        for rep in range(reps):
            rep += 1 # sim reps are created indexed by 1
            
            rep_config = {
                'n': n,
                'p_l': p,
                'snr_x': snr_x,
                'snr_y': snr_y,
                'sparsity': sparsity,
                'k_delta': k_delta,
                'rep': rep
            }
            
            # Determine if model is already saved - skip training
            model_out_filename = os.path.join(
                model_out_path, 
                model_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep))
            if os.path.isfile(model_out_filename + ".pth"):
                print('Model already trained, moving on')
                continue
            
            ##################
            # Setup replicate data
            
            sim_data, sim_data_test = load_data(file_name_base)
            train_subset, test_subset, data_package = process_data(sim_data, 
                                                                   sim_data_test,
                                                                TRAINING_SPLIT,
                                                                TRAINING_SIZE,
                                                                RANDOM_SEED)
            
                
            ##################
            # Perform model inference
            model_config = {
                'k': data_package['k'] + k_delta,
                'k_l_list': [k_l + k_delta for k_l in data_package['k_l_list']],
                'include_view_factors': True,
                'dense_model': False
            }
            local_config = train_config
            
            factor_model, train_handler, global_train_state = train_model(model_config,
                                                     train_config,
                                                     train_subset,
                                                     model_out_filename)
            
            factor_model, test_handler, local_train_state = train_model_locally(factor_model,
                                                            local_config,
                                                            test_subset)
            
            try:             
                metric_out_filename = os.path.join(
                    metric_out_path, 
                    metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep)
                    )
                eval_metric_table = evaluate_fitted_model(rep_config,
                                                          data_package,
                                                          train_handler,
                                                          test_handler,
                                                          global_train_state,
                                                          local_train_state,
                                                          N_POSTERIOR_SAMPLES,
                                                          metric_out_filename)
                
                metric_list.append(eval_metric_table)
            except Exception as e:
                print(e)
                
                
