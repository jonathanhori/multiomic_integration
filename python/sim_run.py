import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects import r, vectors
from rpy2 import rinterface

import pandas as pd
import seaborn as sns
import itertools

from rpy2.robjects import pandas2ri, default_converter, conversion

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO

import importlib

from torchvision import transforms

from pathlib import Path
import time

sys.path.insert(1, "/Users/jonathanhori/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/python")

from data_utils import load_and_process_rds_data_for_condition, normalize_tensor_by_col, \
    extract_est_decomp, extract_sim_decomp, calc_struct, scale_est_struct, scale_sim_struct, \
        eval_rse
from model import SupMultiviewDecomp, do_inference


sim_data_path = "~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/data3/"

# Absolute paths to directories containing dataset replicates
dirs = [os.path.join(sim_data_path, dir) \
    for dir in os.listdir(os.path.expanduser(sim_data_path)) \
        if os.path.isdir(os.path.expanduser(os.path.join(sim_data_path, dir)))]

# dirs[0:5]

filename_dict = {os.path.basename(name): os.listdir(os.path.expanduser(name)) for name in dirs}


n_array = (50, 100, 500, 1000)
p_array = (50, 100, 1000)
snr_x_array = (2, 1, 0.5)
snr_y_array = (2, 1, 0.5)
reps = 1
loading_sparsity = 0

sim_grid = itertools.product(
    n_array,
    p_array,
    snr_x_array,
    snr_y_array,
    # range(reps),
    [loading_sparsity]
)


MINIBATCH_SIZE = 32
NUM_EPOCHS = 1000
initial_lr = 0.01

gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / (NUM_EPOCHS * MINIBATCH_SIZE))

file_dir_base = "n{}p{}_snr{}.{}_sparse{}" #/sim_data_ywithview_rep{}.rds"
file_name_base = "sim_data_ywithview_rep{}"

model_out_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/results/integration/models")
model_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_rep{}"

metric_out_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/results/integration/metrics")
metric_out_filename_base = "run_autonormalguide_n{}p{}_snr{}.{}_sparse{}_rep{}"

metric_list = []

if __name__ == "__main__":
    for n, p, snr_x, snr_y, sparsity in sim_grid:
        pyro.clear_param_store()
        print("Loading:")
        print(file_dir_base.format(n, p, snr_x, snr_y, sparsity))
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
            ##################
            # Setup replicate data
            print(file_name_base.format(rep))
            sim_data = files_for_condition.get(file_name_base.format(rep))
            
            L = int(sim_data.get("L"))
            k = int(sim_data.get("K"))
            X_l_list = sim_data.get("X_l")
            y = sim_data.get("y")
            k_l_list = sim_data.get("K_l").int().tolist()
            n = X_l_list[0].shape[0]
            p_l = int(sim_data.get("p_l")[0])

            # Clean
            clean_X_dict_list = list(map(normalize_tensor_by_col, X_l_list))
            # normalize_tensor_by_col(X)

            # X_clean = clean_X_dict.get("data_clean")
            # X_means = clean_X_dict.get("means")
            # X_stds = clean_X_dict.get("sds")
        
            X_l_list_clean = [clean["data_clean"] for clean in clean_X_dict_list]
            X_l_mean_list = [clean["means"] for clean in clean_X_dict_list]
            X_l_sd_list = [clean["sds"] for clean in clean_X_dict_list]

            y_mean = torch.mean(y)
            y_std = torch.std(y)
            y_clean = (y - y_mean) / y_std

            ##################
            # Perform model inference
            factor_model = SupMultiviewDecomp(
                k,
                k_l_list,
                n,
                include_view_factors = True
            )

            def subsample_create_plates(X, batch_idx, y = None):
                return pyro.plate("obs", factor_model.n, subsample = batch_idx)
            guide = pyro.infer.autoguide.AutoNormal(
                factor_model, 
                create_plates = subsample_create_plates)

            # OPT = Adam({"lr": initial_lr})
            OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
            LOSS = Trace_ELBO(num_particles = 1)
            if n == 50: 
                TOL = 0.001
            elif n == 100:
                TOL = 0.002
            elif n == 500:
                TOL = 0.01
            elif n == 1000:
                TOL = 0.1
            # TOL = 0.5 * n / 1000 # just a heuristic for now
            
            t0 = time.time()
            svi, opt = do_inference(
                X_l_list_clean,
                y_clean,
                model = factor_model,
                guide = guide,
                opt = OPT,
                elbo = LOSS,
                tol = TOL, # tolerance is on epoch loss per datum
                epochs = NUM_EPOCHS,
                minibatch_flag = True,
                minibatch_size = MINIBATCH_SIZE
                )
            t1 = time.time()
            run_minibatch = t1 - t0
            
            #################
            # Inference finished, save and evaluate
            # Save model parameters

            model_out_filename = os.path.join(
                model_out_path, 
                model_out_filename_base.format(n, p, snr_x, snr_y, sparsity, rep))
            print("Exporting:")
            print(model_out_filename)
            

            torch.save({
                "inference_time": run_minibatch,
                # "model_param_store": pyro.get_param_store(),
                "model_state_dict": pyro.get_param_store().get_state(),
                "optimizer_state": opt.get_state(),
                "loss_history": factor_model.loss_history
            }, model_out_filename + ".pth")

            pyro.get_param_store().save(model_out_filename + "_paramstore.pth") #os.path.join(model_out_path, model_out_filename + "_paramstore.pth"))
            
            #################
            # Evaluate model 
            EST_decomp = extract_est_decomp(L, True)

            EST_Struct_shared_l_list = list(map(calc_struct, \
                [EST_decomp.get("EST_Z")] * L, EST_decomp.get("EST_Lambda_l_list")))
            EST_Struct_view_l_list = list(map(calc_struct, \
                EST_decomp.get("EST_Phi_l_list"), EST_decomp.get("EST_Gamma_l_list")))

            SIM_decomp = extract_sim_decomp(sim_data)

            SIM_Struct_shared_l_list = list(map(calc_struct, \
                [SIM_decomp.get("SIM_Z")]* L, SIM_decomp.get("SIM_Lambda_l_list")))
            SIM_Struct_view_l_list = list(map(calc_struct, \
                SIM_decomp.get("SIM_Phi_l_list"), SIM_decomp.get("SIM_Gamma_l_list")))

            EST_Struct_shared_l_list_rescaled = scale_est_struct(EST_Struct_shared_l_list, X_l_mean_list, X_l_sd_list)
            SIM_Struct_shared_l_list_rescaled = scale_sim_struct(SIM_Struct_shared_l_list, X_l_mean_list, X_l_sd_list)

            EST_Struct_view_l_list_rescaled = scale_est_struct(EST_Struct_view_l_list, X_l_mean_list, X_l_sd_list)
            SIM_Struct_view_l_list_rescaled = scale_sim_struct(SIM_Struct_view_l_list, X_l_mean_list, X_l_sd_list)

            shared_rse = [eval_rse(est, sim).detach().item() for est, sim in zip(EST_Struct_shared_l_list_rescaled, SIM_Struct_shared_l_list)]
            view_rse = [eval_rse(est, sim).detach().item() for est, sim in zip(EST_Struct_view_l_list_rescaled, SIM_Struct_view_l_list)]

            eval_metric_table = pd.melt(pd.DataFrame({
                "shared": shared_rse,
                "view_specific": view_rse
                }
                ).rename_axis('view').reset_index(),
                id_vars = ["view"],
                value_vars = ["shared", "view_specific"],
                var_name = "structure",
                value_name = "rse").\
                assign(n = n,
                    p = p_l,
                    snr_x = snr_x,
                    snr_y = snr_y,
                    runtime = run_minibatch
                    )

            # Save evaluation
            metric_out_filename = os.path.join(
                metric_out_path, 
                metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, rep))
            print("Exporting metrics:")
            print(metric_out_filename + ".csv")

            eval_metric_table.to_csv(metric_out_filename)
            metric_list.append(eval_metric_table)
