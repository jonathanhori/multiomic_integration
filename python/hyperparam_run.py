import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects import r, vectors
from rpy2 import rinterface

# import pandas as pd
# import seaborn as sns
import itertools

# from rpy2.robjects import pandas2ri, default_converter, conversion
import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import Trace_ELBO

from sklearn.model_selection import train_test_split

# import importlib

# from torchvision import transforms

from pathlib import Path
import time

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")

from data_utils import load_and_process_rds_data_for_condition, normalize_tensor_by_col, \
    zero_variance_col_filter, obtain_posterior_pred_samples, \
        calc_all_structures_with_rescaling, extract_sim_decomp, eval_performance
from model import SupMultiviewDecomp
from handler import ModelHandler
from constants import Sites, Params


train_sim_data_path = "~/multiomic_integration/sim/data/train/"
test_sim_data_path = "~/multiomic_integration/sim/data/test/"

# Absolute paths to directories containing dataset replicates
dirs = [os.path.join(train_sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(train_sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(train_sim_data_path, dir)))]
test_dirs = [os.path.join(test_sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(test_sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(test_sim_data_path, dir)))]

# dirs[0:5]

filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in dirs}
test_filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in test_dirs}

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
TOL = 1
VARIATIONAL_TOL = 0.3
MINIBATCH_SIZE = 32
MINIBATCH_SIZE_LOW = 16
MIN_EPOCHS = 100
MIN_EPOCHS_HIGH = 500
MIN_EPOCHS_LOCAL = 100
NUM_EPOCHS = 1000
initial_lr = 0.005

gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / (NUM_EPOCHS * MINIBATCH_SIZE))

file_dir_base = "n{}p{}_snr{}.{}_sparse{}" #/sim_data_ywithview_rep{}.rds"
file_name_base = "sim_data_ywithview_rep{}"

model_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/models")
model_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

metric_out_path = os.path.expanduser("~/multiomic_integration/sim/results/integration/metrics")
metric_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

# Create output paths if don't exist
Path(model_out_path).mkdir(parents=True, exist_ok=True)
Path(metric_out_path).mkdir(parents=True, exist_ok=True)

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
            
            # Determine if model is already saved - skip training
            model_out_filename = os.path.join(
                model_out_path, 
                model_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep))
            if os.path.isfile(model_out_filename + ".pth"):
                print('Model already trained, moving on')
                continue
            
            ##################
            # Setup replicate data
            
            data_inputs = load_data(file_name_base)
            train_subset, test_subset = process_data(data_inputs,
                                                     TRAINING_SPLIT,
                                                     TRAINING_SIZE,
                                                     RANDOM_SEED)
            
                
            ##################
            # Perform model inference
            model_config = {
                'k': data_inputs['k']
                'k_l_list': data_inputs['k_l_list'],
                'include_view_factors': True
                'dense_model': False
            }
            train_config = {
                'initial_lr':
                'betas':
                'num_particles':
                'lr_step_size':
                'lr_decay_factor':
                'minibatch_size':
                'min_epochs':
                'max_epochs':
            }
            factor_model = train_model(model_config,
                                       train_config)
            try:
                # if TRAINING_SPLIT:
                #     print()
                # else:
                #     print()
                
                
                #################
                # Inference for predictive model
                print('local training')
                LOCAL_OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
                LOCAL_LOSS = Trace_ELBO(num_particles = 1)
                
                test_handler = ModelHandler("predict",
                                            factor_model,
                                            LOCAL_OPT, 
                                            LOCAL_LOSS)
                
                t0 = time.time()
                svi = test_handler.do_inference(
                    test_subset,
                    variational_tol = VARIATIONAL_TOL,
                    min_epochs = MIN_EPOCHS_LOCAL,
                    epochs = NUM_EPOCHS,
                    minibatch_flag = True,
                    minibatch_size = MINIBATCH_SIZE,
                    verbose = False
                    )
                t1 = time.time()
                test_run_minibatch = t1 - t0
                
                torch.save({
                    "inference_time": run_minibatch,
                    "epochs": factor_model.local_epochs,
                    # "model_param_store": pyro.get_param_store(),
                    "model_state_dict": pyro.get_param_store().get_state(),
                    "optimizer_state": test_handler.opt.get_state(),
                    "loss_history": factor_model.local_loss_history,
                    "param_convergence_history": factor_model.local_var_param_convergence_history                
                }, model_out_filename + "_local.pth")

                pyro.get_param_store().save(model_out_filename + "_local_paramstore.pth")
                    
                    
                    
                #######
                
                eval_metric_table = evaluate_fitted_model(factor_model)
                
                metric_list.append(eval_metric_table)
            except Exception as e:
                print(e)
                
                
