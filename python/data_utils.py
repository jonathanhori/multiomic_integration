import os
import torch

import pyro
from pyro.infer import Predictive

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter, conversion
from rpy2 import rinterface
from rpy2.robjects import vectors

from pathlib import Path

import pandas as pd

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
def normalize_tensor_by_col(data, training_means = None, training_stds = None):
    # Means and sds are are broadcast across rows
    if training_means is None and training_stds is None:
        data_means = torch.mean(data, 0)
        data_stds = torch.std(data, 0)
    else:
        data_means = training_means
        data_stds = training_stds
    return {
        "data_clean": (data - data_means) / data_stds,
        "means": data_means,
        "sds": data_stds
    }

# Remove columns that are all 0s
def zero_variance_col_filter(tens_2d):
    return tens_2d.abs().sum(dim=0).bool()
    
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
    
def obtain_posterior_pred_samples(#model,
                                # guide,
                                handler,
                                num_samples,
                                train_X_l_list_clean,
                                test_X_l_list_clean,
                                structure_return_sites,
                                outcome_return_sites):
    """TODO: possibly remove, handler is specific to train/test set"""
    n_train = train_X_l_list_clean[0].shape[0]
    predictive = Predictive(handler.model, 
                        guide = handler.guide, 
                        num_samples = num_samples,
                        return_sites = structure_return_sites)
    post_samples = predictive(
        train_X_l_list_clean,
        torch.arange(n_train)
        )
    # post_samples = handler.predict(train_X_l_list_clean, structure_return_sites)

    # Sample posterior predictive for outcome - specify return_sites
    n_test = test_X_l_list_clean[0].shape[0]
    predictive_y = Predictive(handler.model, 
                            guide = handler.guide, 
                            num_samples = num_samples,
                            return_sites = outcome_return_sites)
    # Do not supply y
    post_samples_y = predictive_y(
        test_X_l_list_clean,
        torch.arange(n_test)
        )
    # post_samples_y = handler.predict(test_X_l_list_clean, outcome_return_sites)
    return (post_samples, post_samples_y)
    
    
def calc_struct(scores, loadings):
    return scores @ loadings.T

def scale_est_struct(struct_list, mean_list, sd_list):
    return [struct * sd + mean for struct, mean, sd in zip(struct_list, mean_list, sd_list)]
    # return [(struct_list[i] - mean_list[i]) / sd_list[i] for i in range(len(struct_list))]

def scale_sim_struct(struct_list, mean_list, sd_list):
    return [(struct - mean) / sd for struct, mean, sd in zip(struct_list, mean_list, sd_list)]
    # return [struct_list[i] * sd_list[i] + mean_list[i] for i in range(len(struct_list))]
    
    
def calc_all_structures_with_rescaling(L,
                                       column_filters,
                                        X_l_mean_list,
                                        X_l_sd_list,
                                        post_pred_structure_samples,
                                        post_pred_outcome_samples,
                                        sim_decomp,
                                        sim_outcome):

    SIM_joint_struct_l_list = list(map(calc_struct, \
        [sim_decomp.get("SIM_Z")] * L, sim_decomp.get("SIM_Lambda_l_list")))
    SIM_individual_struct_l_list = list(map(calc_struct, \
        sim_decomp.get("SIM_Phi_l_list"), sim_decomp.get("SIM_Gamma_l_list")))
    
    # Filter out zero variance columns from sim data structures
    SIM_joint_struct_l_list = [struct[:, filter] for struct, filter 
                               in zip(SIM_joint_struct_l_list, column_filters)]
    SIM_individual_struct_l_list = [struct[:, filter] for struct, filter 
                               in zip(SIM_individual_struct_l_list, column_filters)]
    

    # EST_Struct_shared_l_list_rescaled = scale_est_struct(EST_Struct_shared_l_list, X_l_mean_list, X_l_sd_list)
    SIM_joint_struct_l_list_rescaled = scale_sim_struct(SIM_joint_struct_l_list, X_l_mean_list, X_l_sd_list)

    # EST_Struct_view_l_list_rescaled = scale_est_struct(EST_Struct_view_l_list, X_l_mean_list, X_l_sd_list)
    SIM_individual_struct_l_list_rescaled = scale_sim_struct(SIM_individual_struct_l_list, X_l_mean_list, X_l_sd_list)
    
    
    
    # Structures
    joint_structure_summaries = summarise_structure_list(post_pred_structure_samples, L, "joint")
    individual_structure_summaries = summarise_structure_list(post_pred_structure_samples, L, "individual")
    
    POST_joint_struct_l_list = [summary.get('mean') for summary in joint_structure_summaries]
    POST_individual_struct_l_list = [summary.get('mean') for summary in individual_structure_summaries]
    
    if "y" in post_pred_outcome_samples.keys():
        PRED_outcome_summary = summarise_post_samples(post_pred_outcome_samples.get("y"))
    elif "y_pred" in post_pred_outcome_samples.keys():
        PRED_outcome_summary = summarise_post_samples(post_pred_outcome_samples.get("y_pred"))
    PRED_y = PRED_outcome_summary.get('mean')
    
    return {
        'sim_joint_data_struct': SIM_joint_struct_l_list_rescaled, 
        'post_pred_joint_data_struct': POST_joint_struct_l_list,
        'post_pred_joint_summaries': joint_structure_summaries,
        'sim_individual_data_struct': SIM_individual_struct_l_list_rescaled, 
        'post_pred_individual_data_struct': POST_individual_struct_l_list,
        'post_pred_individual_summaries': individual_structure_summaries,
        'sim_outcome': sim_outcome, 
        'post_pred_outcome': PRED_y,
        'post_pred_outcome_summaries': PRED_outcome_summary
    }


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
        # "std": torch.std(sample_tensor, dim = 0),
        "q2.5": torch.quantile(sample_tensor, 0.025, dim = 0),
        # "q5": torch.quantile(sample_tensor, 0.05, dim = 0),
        # "q25": torch.quantile(sample_tensor, 0.25, dim = 0),
        "q50": torch.quantile(sample_tensor, 0.50, dim = 0),
        # "q75": torch.quantile(sample_tensor, 0.75, dim = 0),
        # "q95": torch.quantile(sample_tensor, 0.95, dim = 0),
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

    
def eval_performance(sim_joint_data_struct, 
                     post_pred_joint_data_struct,
                     post_pred_joint_summaries,
                     sim_individual_data_struct, 
                     post_pred_individual_data_struct,
                     post_pred_individual_summaries,
                     sim_outcome, 
                     post_pred_outcome,
                     post_pred_outcome_summaries):    
    """
    Inputs for each structure (joint, individual, outcome) are:
        - simulated strucures (ground truth)
        - posterior predicted point estimates (means)
        - posterior prediction summary statistics (for intervals)
    """
    joint_rse = [eval_rse(est, sim).detach().item() for est, sim in \
        zip(post_pred_joint_data_struct, sim_joint_data_struct)]
    individual_rse = [eval_rse(est, sim).detach().item() for est, sim in \
        zip(post_pred_individual_data_struct, sim_individual_data_struct)]
    
    joint_coverage = eval_credible_interval(
        sim_joint_data_struct,
        post_pred_joint_summaries,
        '95')
    individual_coverage = eval_credible_interval(
        sim_individual_data_struct,
        post_pred_individual_summaries,
        '95')
    
    outcome_mse = eval_mse(post_pred_outcome, sim_outcome).item()
    outcome_coverage = eval_post_coverage(
        sim_outcome,
        post_pred_outcome_summaries.get('q2.5'),
        post_pred_outcome_summaries.get('q97.5')
        ).item()

    eval_metric_table = pd.wide_to_long(pd.DataFrame({
        "joint.rse": joint_rse,
        "individual.rse": individual_rse,
        "joint.95p_coverage": joint_coverage,
        "individual.95p_coverage": individual_coverage,
        "outcome.mse": outcome_mse,
        "outcome.95p_coverage": outcome_coverage
        }
        ).rename_axis('view').reset_index(),
        stubnames = ['joint', 'individual', 'outcome'],
        i = 'view',
        j = 'metric',
        sep = '.',
        suffix = '.+')
    return eval_metric_table