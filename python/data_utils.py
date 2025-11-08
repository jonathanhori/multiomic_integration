import os
import torch

import pyro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter, conversion
from rpy2 import rinterface
from rpy2.robjects import vectors

from pathlib import Path

######## 
# Utilities for converting data from R
def floatmatrix_to_torch(m):
    t = torch.tensor(list(m), dtype=torch.float32)
    # R matrices are stored column-major, so reshape and transpose
    t = t.view(m.ncol, m.nrow).t()
    return t

def r_to_torch(obj):
    """Recursively convert R objects (via rpy2) to Python/PyTorch equivalents."""
    
    # Case 1: R list (possibly named)
    if isinstance(obj, vectors.ListVector):
        names = getattr(obj, "names", None)
        if names is not None and not isinstance(names, rinterface.NULLType):
            return {name: r_to_torch(obj.rx2(name)) for name in names}
        else:
            return [r_to_torch(el) for el in obj]
    
    # Case 2: R matrix
    elif isinstance(obj, vectors.FloatMatrix):
        return floatmatrix_to_torch(obj)
    
    # Case 3: Numeric / integer / logical R vector
    elif isinstance(obj, (vectors.FloatSexpVector, vectors.IntSexpVector, vectors.BoolSexpVector)):
        return torch.tensor(list(obj), dtype=torch.float32)
    
    # Case 4: R data frame
    elif isinstance(obj, vectors.DataFrame):
        return {col: r_to_torch(obj.rx2(col)) for col in obj.names}
    
    # Case 5: Python list
    elif isinstance(obj, list):
        return [r_to_torch(el) for el in obj]
    
    # Case 6: R NULL
    elif isinstance(obj, rinterface.NULLType):
        return None
    
    # Fallback
    else:
        return obj
    
    
def load_and_process_rds_data_for_condition(
    directory_name, # n100p100_snr2.2_sparse0
    directory_path, # "~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/data3/",
    filename_dict,
    num_reps = 10
):
    """
    Each directory name corresponds to a condition. The directory contains replicates.
    This function returns the processed RDS files ready to load into a Pyro model.
    """
    # Code to use R functions in python
    conversion._converter = None
    conversion.set_conversion(default_converter)
    readRDS = robjects.r['readRDS']

    # What are the RDS filenames we need to load
    filenames = filename_dict.get(directory_name)[0:num_reps]

    directory_names_with_path = [os.path.expanduser(os.path.join(directory_path, \
        directory_name, \
            name)) \
                for name in filenames]

    print(directory_names_with_path)

    files = {Path(name).stem: readRDS(name) for name in directory_names_with_path}
    files = {n: r_to_torch(v) for n, v in files.items()}
    return files


##########
# Input data processing
def normalize_tensor_by_col(data):
    # Means and sds are are broadcast across rows
    data_means = torch.mean(data, 0)
    data_stds = torch.std(data, 0)
    return {
        "data_clean": (data - data_means) / data_stds,
        "means": data_means,
        "sds": data_stds
    }
    
###########
# Use estimated and simulated data
def extract_est_decomp(L, autoguide = True):
    """
    Extracts from current param store
    """
    if autoguide:
        # From AutoNormal locs
        EST_Lambda_l_list = [pyro.param(f"AutoNormal.locs.Lambda_l{l}") for l in range(L)]
        EST_Z = pyro.param("AutoNormal.locs.Z")

        EST_Gamma_l_list = [pyro.param(f"AutoNormal.locs.Gamma_l{l}") for l in range(L)]
        EST_Phi_l_list = [pyro.param(f"AutoNormal.locs.Phi_l{l}") for l in range(L)]

        return {
            "EST_Lambda_l_list": EST_Lambda_l_list,
            "EST_Z": EST_Z,
            "EST_Gamma_l_list": EST_Gamma_l_list,
            "EST_Phi_l_list": EST_Phi_l_list
        }
    else:
        raise NotImplementedError

def extract_sim_decomp(sim_data_dict):
    SIM_Lambda_l_list = sim_data_dict["Lambda_l"]
    SIM_Z = sim_data_dict["Z"] 

    SIM_Gamma_l_list = sim_data_dict["Gamma_l"]
    SIM_Phi_l_list = sim_data_dict["Phi"]

    return {
            "SIM_Lambda_l_list": SIM_Lambda_l_list,
            "SIM_Z": SIM_Z,
            "SIM_Gamma_l_list": SIM_Gamma_l_list,
            "SIM_Phi_l_list": SIM_Phi_l_list
        }
    
    
def calc_struct(scores, loadings):
    return scores @ loadings.T

def scale_est_struct(struct_list, mean_list, sd_list):
    return [struct * sd + mean for struct, mean, sd in zip(struct_list, mean_list, sd_list)]
    # return [(struct_list[i] - mean_list[i]) / sd_list[i] for i in range(len(struct_list))]

def scale_sim_struct(struct_list, mean_list, sd_list):
    return [(struct - mean) / sd for struct, mean, sd in zip(struct_list, mean_list, sd_list)]
    # return [struct_list[i] * sd_list[i] + mean_list[i] for i in range(len(struct_list))]


###############
# Evaluate structures

def eval_rse(est_struct, sim_struct):
    return torch.norm(sim_struct - est_struct) ** 2 / torch.norm(sim_struct) ** 2


def eval_mse(pred_outcome, sim_outcome):
    return torch.nn.functional.mse_loss(sim_outcome, pred_outcome)


def summarise_post_samples(sample_tensor): 
    """
    Calculate summary statistics for posterior samples
    """
    summary = {
        "mean": torch.mean(sample_tensor, dim = 0),
        "std": torch.std(sample_tensor, dim = 0),
        "q2.5": torch.quantile(sample_tensor, 0.025, dim = 0),
        "q5": torch.quantile(sample_tensor, 0.05, dim = 0),
        "q25": torch.quantile(sample_tensor, 0.25, dim = 0),
        "q50": torch.quantile(sample_tensor, 0.50, dim = 0),
        "q75": torch.quantile(sample_tensor, 0.75, dim = 0),
        "q95": torch.quantile(sample_tensor, 0.95, dim = 0),
        "q97.5": torch.quantile(sample_tensor, 0.975, dim = 0)
    }
    return summary


def summarise_structure_list(post_samples, L, struct_type):
    """
    Summarise all joint or individual structures from a given model
    """
    if struct_type == "joint":
        return [summarise_post_samples(post_samples.get(f'joint_structure_l{l}')) for l in range(L)]
    elif struct_type == "individual": 
        return [summarise_post_samples(post_samples.get(f'view_structure_l{l}')) for l in range(L)]

def eval_post_coverage(sim_struct, lower, upper):
    """
    Calculate the nominal entrywise coverage probability for posterior samples and simulated data
    """
    return torch.sum(torch.logical_and(
            torch.less_equal(lower, sim_struct),
            torch.greater_equal(upper, sim_struct)
        )) / torch.numel(sim_struct)
    
def eval_credible_interval(sim_structure_list, summary_list, region = '95'):
    """
    Determine coverage of credible intervals for given lists of simulated structures and
        associated dictionaries of summary statistics
    """
    if region == '95':
        lower = 'q2.5'
        upper = 'q97.5'
    elif region == '90':
        lower = 'q5'
        upper = 'q95'
    elif region == '50':
        lower = 'q25'
        upper = 'q75'
    return [eval_post_coverage(sim_struct, post_summary.get(lower), post_summary.get(upper)).item() \
        for sim_struct, post_summary in zip(sim_structure_list, summary_list)]