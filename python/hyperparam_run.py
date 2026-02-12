import os
import sys


import itertools

# import torch
import pyro
# from pyro.optim import Adam, ClippedAdam
# from pyro.infer import Trace_ELBO

from pathlib import Path
# import time

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from run_methods import evaluate_fitted_model, load_data, process_data, train_model, train_model_locally
from data_utils import load_and_process_rds_data_for_condition

# from model import SupMultiviewDecomp
# from handler import ModelHandler
# from constants import Sites, Params


train_sim_data_path = "~/multiomic_integration/sim/data_ortho/train/"
test_sim_data_path = "~/multiomic_integration/sim/data_ortho/test/"

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

model_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/hyperparam_gridsearch/models")
model_out_filename_base = "run_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

metric_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/hyperparam_gridsearch/metrics")
metric_out_filename_base = "run_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

# Create output paths if don't exist
Path(model_out_path).mkdir(parents=True, exist_ok=True)
Path(metric_out_path).mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123
TRAINING_SPLIT = True
TRAINING_SIZE = 0.8

# TESTING
n_array = [500] #, 1000)
p_array = [100]
snr_x_array = [2] #(2, 1, 0.5)
snr_y_array = [2] #(2, 1, 0.5)
reps = 1
loading_sparsity = [0.25]
k_deltas = [10]

# n_array = (50, 100, 500) #, 1000)
# p_array = (50, 100, 1000)
# snr_x_array = [2] #(2, 1, 0.5)
# snr_y_array = [2] #(2, 1, 0.5)
# reps = 10
# loading_sparsity = [0, 0.25, 0.5]
# k_deltas = [0, 10]

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

lr_array = [0.1, 0.05, 0.01]
beta_array = [(0.9, 0.999), (0.8, 0.999), (0.9, 0.9)]
particle_array = [1]
lr_step_array = [5, 10, 20, 30]
lr_decay_array = [0.9, 0.5, 0.1]
minibatch_array = [16, 32, 64]
min_epoch_array = [20]
max_epoch_array = [300]

train_grid = itertools.product(
    lr_array,
    beta_array,
    particle_array,
    lr_step_array,
    lr_decay_array,
    minibatch_array,
    min_epoch_array,
    max_epoch_array
)
train_grid_names = ('initial_lr', 'betas', 'num_particles', 'lr_step_size', \
    'lr_decay_factor', 'minibatch_size', 'min_epochs', 'max_epochs')

metric_list = []

if __name__ == "__main__":
    for n, p, snr_x, snr_y, sparsity, k_delta in sim_grid:
        for params in train_grid:
            train_config = dict(zip(train_grid_names, params))
            
            pyro.clear_param_store()
            
            print("Loading:")
            print(file_dir_base.format(n, p, snr_x, snr_y, sparsity))
            
            # Determine if all reps are already done - skip training
            check_metric_out_filename = os.path.join(
                metric_out_path, 
                metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, reps))
            if os.path.isfile(check_metric_out_filename + ".pth"):
                print('Metrics already evaluated for model, moving on')
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
            else:
                test_files_for_condition = None
            
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
                
                sim_data, sim_data_test = load_data(file_name_base,
                                                    rep,
                                                    files_for_condition,
                                                    test_files_for_condition)
                
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
                                                                test_subset,
                                                                model_out_filename)
                
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
                                                            metric_out_filename
                                                            ).\
                                        assign(
                                            initial_lr = train_config['initial_lr'],
                                            betas = str(train_config['betas']),
                                            num_particles = train_config['num_particles'],
                                            lr_step_size = train_config['lr_step_size'],
                                            lr_decay_factor = train_config['lr_decay_factor'],
                                            minibatch_size = train_config['minibatch_size'],
                                            min_epochs = train_config['min_epochs'],
                                            max_epochs = train_config['max_epochs']
                                        )
                    print("Exporting metrics:")
                    print(metric_out_filename)

                    eval_metric_table.to_csv(metric_out_filename + ".csv")
                    
                    metric_list.append(eval_metric_table)
                except Exception as e:
                    print(e)