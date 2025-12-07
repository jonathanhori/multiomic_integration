import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects import r, vectors
from rpy2 import rinterface

# import pandas as pd
# import seaborn as sns
import itertools

# from rpy2.robjects import pandas2ri, default_converter, conversion

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

sys.path.insert(1, "/Users/jonathanhori/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/python")

from data_utils import load_and_process_rds_data_for_condition, normalize_tensor_by_col, \
    zero_variance_col_filter, obtain_posterior_pred_samples, \
        calc_all_structures_with_rescaling, extract_sim_decomp, eval_performance
from model import SupMultiviewDecomp, do_inference
from handler import ModelHandler


sim_data_path = "~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/data/"

# Absolute paths to directories containing dataset replicates
dirs = [os.path.join(sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(sim_data_path, dir)))]

# dirs[0:5]

filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in dirs}

RANDOM_SEED = 123
TRAINING_SPLIT = True
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

N_POSTERIOR_SAMPLES = 1000
TOL = 1
VARIATIONAL_TOL = 0.3
MINIBATCH_SIZE = 32
MIN_EPOCHS = 20
NUM_EPOCHS = 1000
initial_lr = 0.005

gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / (NUM_EPOCHS * MINIBATCH_SIZE))

file_dir_base = "n{}p{}_snr{}.{}_sparse{}" #/sim_data_ywithview_rep{}.rds"
file_name_base = "sim_data_ywithview_rep{}"

model_out_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/results/integration/models")
model_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"

metric_out_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/results/integration/metrics")
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
            sim_data_path, # loaded above
            filename_dict, # loaded above
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
            print(file_name_base.format(rep))
            sim_data = files_for_condition.get(file_name_base.format(rep))
            
            L = int(sim_data.get("L"))
            X_l_list = sim_data.get("X_l")
            y = sim_data.get("y")
            k = int(sim_data.get("K") + k_delta)
            k_l_list = [int(k_l + k_delta) for k_l in sim_data.get("K_l").int().tolist()]
            n = X_l_list[0].shape[0]
            p_l = int(sim_data.get("p_l")[0])
            
            # Perform train/test split if necessary
            if TRAINING_SPLIT:
                train_idx, test_idx = train_test_split(
                    torch.arange(X_l_list[0].shape[0]),
                    train_size = TRAINING_SIZE,
                    random_state = RANDOM_SEED,
                    shuffle = True
                )
                train_X_l_list = [X[train_idx, :] for X in X_l_list]
                test_X_l_list = [X[test_idx, :] for X in X_l_list]
                
                train_y = y[train_idx]
                test_y = y[test_idx]
            else:
                train_idx = torch.arange(n)
                test_idx = torch.arange(n)
                
                train_X_l_list = X_l_list
                test_X_l_list = train_X_l_list
                
                train_y = y
                test_y = train_y

            # Clean: remove 0 variance columns and standardize. Using Training data            
            X_l_list_column_filters = [zero_variance_col_filter(X) for X in train_X_l_list]
            
            train_X_l_list = [X[:, col_filter] for X, col_filter in zip(train_X_l_list, X_l_list_column_filters)]
            test_X_l_list = [X[:, col_filter] for X, col_filter in zip(test_X_l_list, X_l_list_column_filters)]

            clean_train_X_dict_list = [normalize_tensor_by_col(X) for X in train_X_l_list]
            X_l_mean_list = [clean["means"] for clean in clean_train_X_dict_list]
            X_l_sd_list = [clean["sds"] for clean in clean_train_X_dict_list]
            
            clean_test_X_dict_list = [normalize_tensor_by_col(X, means, stds) 
                                      for X, means, stds in 
                                      zip(test_X_l_list, X_l_mean_list, X_l_sd_list)]
            
            train_X_l_list_clean = [clean["data_clean"] for clean in clean_train_X_dict_list]
            test_X_l_list_clean = [clean["data_clean"] for clean in clean_test_X_dict_list]

            y_mean = torch.mean(train_y)
            y_std = torch.std(train_y)
            
            train_y_clean = (train_y - y_mean) / y_std
            test_y_clean = (test_y - y_mean) / y_std
            
            
            # Package data into dataset
            train_subset = TensorDataset(torch.arange(train_X_l_list_clean[0].shape[0]), 
                                        *train_X_l_list_clean, 
                                        train_y_clean)
            test_subset = TensorDataset(torch.arange(test_X_l_list_clean[0].shape[0]), 
                                        *test_X_l_list_clean, 
                                        test_y_clean)
                
            ##################
            # Perform model inference
            factor_model = SupMultiviewDecomp(
                k,
                k_l_list,
                include_view_factors = True,
                dense = False
            )

            # def subsample_create_plates(X, batch_idx, y = None):
            #     return pyro.plate("obs", factor_model.n, subsample = batch_idx)
            # guide = pyro.infer.autoguide.AutoNormal(
            #     factor_model, 
            #     create_plates = subsample_create_plates)

            # OPT = Adam({"lr": initial_lr})
            OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
            LOSS = Trace_ELBO(num_particles = 1)
            
            train_handler = ModelHandler("train",
                                         factor_model,
                                         OPT, 
                                         LOSS)
            
            try:
                t0 = time.time()
                svi, opt = train_handler.do_inference(
                    # X_l_list_clean,
                    # y_clean,
                    train_subset,
                    # model = factor_model,
                    # guide = guide,
                    # opt = OPT,
                    # elbo = LOSS,
                    # tol = TOL, # tolerance is on epoch loss per datum,
                    variational_tol = VARIATIONAL_TOL,
                    min_epochs = MIN_EPOCHS,
                    epochs = NUM_EPOCHS,
                    minibatch_flag = True,
                    minibatch_size = MINIBATCH_SIZE,
                    verbose = False
                    )
                t1 = time.time()
                run_minibatch = t1 - t0
                
                #################
                # Inference finished, save and evaluate
                # Save model parameters

                print("Exporting:")
                print(model_out_filename)

                torch.save({
                    "inference_time": run_minibatch,
                    "epochs": factor_model.total_epochs,
                    # "model_param_store": pyro.get_param_store(),
                    "model_state_dict": pyro.get_param_store().get_state(),
                    "optimizer_state": opt.get_state(),
                    "loss_history": factor_model.loss_history,
                    "param_convergence_history": factor_model.var_param_convergence_history                
                }, model_out_filename + ".pth")

                pyro.get_param_store().save(model_out_filename + "_paramstore.pth") #os.path.join(model_out_path, model_out_filename + "_paramstore.pth"))
                
                # if TRAINING_SPLIT:
                #     print()
                # else:
                #     print()
                
                
                #################
                # Inference for predictive model
                LOCAL_OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
                LOCAL_LOSS = Trace_ELBO(num_particles = 1)
                
                test_handler = ModelHandler("test",
                                            factor_model,
                                            LOCAL_OPT, 
                                            LOCAL_LOSS)
                
                t0 = time.time()
                svi, opt = test_handler.do_inference(
                    test_subset,
                    variational_tol = VARIATIONAL_TOL,
                    min_epochs = MIN_EPOCHS,
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
                    "optimizer_state": opt.get_state(),
                    "loss_history": factor_model.local_loss_history,
                    "param_convergence_history": factor_model.local_var_param_convergence_history                
                }, model_out_filename + "_local.pth")

                pyro.get_param_store().save(model_out_filename + "_local_paramstore.pth")
                    
                #################
                # Sample from posterior
                print('Sampling from posterior')
                sites = [f'joint_structure_l{l}' for l in range(L)] + [f'view_structure_l{l}' for l in range(L)]
                outcome_sites = ["y"]
                post_samples = train_handler.predict(train_X_l_list_clean, N_POSTERIOR_SAMPLES, sites)
                post_samples_y = test_handler.predict(test_X_l_list_clean, N_POSTERIOR_SAMPLES, outcome_sites)
                
                # post_samples, post_samples_y = obtain_posterior_pred_samples(factor_model,
                #                                                             guide,
                #                                                             N_POSTERIOR_SAMPLES,
                #                                                             train_X_l_list_clean,
                #                                                             test_X_l_list_clean,
                #                                                             sites,
                #                                                             outcome_sites)
                    
                
                #################
                # Calculate simulated and posterior sample structures
                # Compare structure decomposition in training data with prediction performance
                #       in test data
                SIM_decomp = extract_sim_decomp(sim_data)
                
                # Filter corresponding rows of test set structures
                train_SIM_decomp = {k: v for k, v in SIM_decomp.items()}
                train_SIM_decomp['SIM_Z'] = train_SIM_decomp['SIM_Z'][train_idx, :]
                train_SIM_decomp['SIM_Phi_l_list'] = [Phi[train_idx, :] for Phi in train_SIM_decomp['SIM_Phi_l_list']]
            
                eval_structures = calc_all_structures_with_rescaling(L,
                                                                    X_l_list_column_filters,
                                                                    X_l_mean_list,
                                                                    X_l_sd_list,
                                                                    post_samples,
                                                                    post_samples_y,
                                                                    train_SIM_decomp,
                                                                    test_y_clean.squeeze())
                    

                #################
                # Evaluate
                # Compare estimated structures (targeting mean 0 var 1 data) with 
                #   RESCALED simulated structures
                
                eval_metric_table = eval_performance(**eval_structures)
                eval_metric_table = eval_metric_table.\
                    assign(n = n,
                        p = p_l,
                        snr_x = snr_x,
                        snr_y = snr_y,
                        rep = rep,
                        sparsity = sparsity,
                        k = k,
                        runtime = run_minibatch,
                        predict_runtime = test_run_minibatch
                        )

                # Save evaluation
                metric_out_filename = os.path.join(
                    metric_out_path, 
                    metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep))
                print("Exporting metrics:")
                print(metric_out_filename)

                eval_metric_table.to_csv(metric_out_filename + ".csv")
                metric_list.append(eval_metric_table)
            except Exception as e:
                print(e)
                
                
