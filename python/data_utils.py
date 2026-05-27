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

from constants import Sites

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


def load_rds_rep(directory_name, directory_path, filename_dict, rep_stem):
    """Load a single replicate RDS file by its stem name (filename without extension)."""
    conversion._converter = None
    conversion.set_conversion(default_converter)
    readRDS = robjects.r['readRDS']

    all_files = filename_dict.get(directory_name, [])
    match = next((f for f in all_files if Path(f).stem == rep_stem), None)
    if match is None:
        raise FileNotFoundError(f"No file with stem '{rep_stem}' in {directory_name}")
    filepath = os.path.expanduser(os.path.join(directory_path, directory_name, match))
    return r_to_torch(readRDS(filepath))


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
    SIM_Phi_l_perp_list = sim_data_dict["Phi_perp"]

    return {
            "SIM_Lambda_l_list": SIM_Lambda_l_list,
            "SIM_Z": SIM_Z,
            "SIM_Gamma_l_list": SIM_Gamma_l_list,
            "SIM_Phi_l_list": SIM_Phi_l_list,
            "SIM_Phi_l_perp_list": SIM_Phi_l_perp_list
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
    
    
def scale_sim_cov(sim_cov, sim_sd1, sim_sd2):
    # For one matrix at a time
    return torch.diag(1 / sim_sd1) @ sim_cov @ torch.diag(1 / sim_sd2)
    
    
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

def calc_cov(loading1, loading2 = None):
    if loading2 is None:
        return loading1 @ loading1.T
    else:
        return loading1 @ loading2.T
    
def eval_cov(est_cov, sim_cov):
    # Sim cov should be rescaled by corresponding standard deviations to be on the same scale
    return torch.norm(sim_cov - est_cov) ** 2 / (est_cov.shape[0] * est_cov.shape[1])


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
    
    # covariance_diff = [eval_cov(est, sim).detach().item() for est, sim in \
    #     ]

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


def extract_sim_decomp_shared(sim_data_dict):
    """Shared-structure analogue of extract_sim_decomp. No view-specific factors."""
    return {
        "SIM_Lambda_l_list": sim_data_dict["Lambda_l"],
        "SIM_Z": sim_data_dict["Z"],
    }


def calc_all_structures_with_rescaling_shared(
    L,
    column_filters,
    X_l_mean_list,
    X_l_sd_list,
    post_pred_structure_samples,
    post_pred_outcome_samples,
    sim_decomp,
    sim_outcome,
    post_loading_samples=None,
):
    """Shared-structure analogue of calc_all_structures_with_rescaling.

    If post_loading_samples is provided (a dict that includes Lambda_l{l} keys),
    also computes within-view and cross-view covariance matrices for both the
    posterior mean estimate and the rescaled simulation truth, and includes them
    in the returned dict under 'est_cov' and 'sim_cov'.
    """
    SIM_joint_struct_l_list = list(map(calc_struct,
        [sim_decomp["SIM_Z"]] * L, sim_decomp["SIM_Lambda_l_list"]))
    SIM_joint_struct_l_list = [s[:, f] for s, f in zip(SIM_joint_struct_l_list, column_filters)]
    SIM_joint_struct_l_list_rescaled = scale_sim_struct(
        SIM_joint_struct_l_list, X_l_mean_list, X_l_sd_list)

    mean_Z = post_pred_structure_samples[Sites.Z].mean(dim=0).squeeze()
    POST_joint_struct_l_list = [
        mean_Z @ post_pred_structure_samples[Sites.Lambda_l.format(l=l)].mean(dim=0).squeeze().mT
        for l in range(L)
    ]

    PRED_outcome_summary = summarise_post_samples(post_pred_outcome_samples["y_pred"])

    result = {
        "sim_joint_data_struct": SIM_joint_struct_l_list_rescaled,
        "post_pred_joint_data_struct": POST_joint_struct_l_list,
        "post_pred_joint_summaries": None,
        "sim_outcome": sim_outcome,
        "post_pred_outcome": PRED_outcome_summary["mean"].squeeze(),
        "post_pred_outcome_summaries": PRED_outcome_summary,
    }

    if post_loading_samples is not None:
        # Posterior mean loading matrix (p_l x k) for each view
        est_Lambda_list = [
            post_loading_samples[Sites.Lambda_l.format(l=l)].mean(dim=0).squeeze()
            for l in range(L)
        ]
        # Simulated loadings filtered to the training feature subset
        sim_Lambda_list = [
            sim_decomp["SIM_Lambda_l_list"][l][column_filters[l]]
            for l in range(L)
        ]

        est_cov = {}
        sim_cov = {}
        for l in range(L):
            for m in range(l, L):
                loading2_est = est_Lambda_list[m] if l != m else None
                loading2_sim = sim_Lambda_list[m] if l != m else None
                est_cov[(l, m)] = calc_cov(est_Lambda_list[l], loading2_est)
                raw_cov = calc_cov(sim_Lambda_list[l], loading2_sim)
                sim_cov[(l, m)] = scale_sim_cov(raw_cov, X_l_sd_list[l], X_l_sd_list[m])

        result["est_cov"] = est_cov
        result["sim_cov"] = sim_cov

    return result


def eval_performance_shared(
    sim_joint_data_struct,
    post_pred_joint_data_struct,
    post_pred_joint_summaries,
    sim_outcome,
    post_pred_outcome,
    post_pred_outcome_summaries
):
    """Shared-structure analogue of eval_performance. Joint structure + outcome only."""
    joint_rse = [eval_rse(est, sim).detach().item()
                 for est, sim in zip(post_pred_joint_data_struct, sim_joint_data_struct)]
    outcome_mse = eval_mse(post_pred_outcome, sim_outcome).item()
    outcome_coverage = eval_post_coverage(
        sim_outcome,
        post_pred_outcome_summaries["q2.5"],
        post_pred_outcome_summaries["q97.5"]
    ).item()

    return pd.DataFrame({
        "view": range(len(joint_rse)),
        "joint_rse": joint_rse,
        "joint_95p_coverage": float("nan"),
        "outcome_mse": outcome_mse,
        "outcome_95p_coverage": outcome_coverage,
    })


def eval_cov_shared(est_cov, sim_cov, L):
    """Evaluate within-view and cross-view covariance reconstruction.

    Uses eval_cov (normalised squared Frobenius norm) for every (l, m) pair
    with l <= m, reusing the same helper as the single-view model.

    Parameters
    ----------
    est_cov, sim_cov : dict mapping (l, m) -> tensor
        As produced by calc_all_structures_with_rescaling_shared when
        post_loading_samples is provided.

    Returns
    -------
    DataFrame with columns: view_l, view_m, cov_type ('within'/'cross'), cov_frob
    """
    rows = []
    for l in range(L):
        for m in range(l, L):
            rows.append({
                "view_l":   l,
                "view_m":   m,
                "cov_type": "within" if l == m else "cross",
                "cov_frob": eval_cov(est_cov[(l, m)], sim_cov[(l, m)]).item(),
            })
    return pd.DataFrame(rows)


def extract_sim_decomp_single_view(sim_data_torch):
    """Single-view analogue of extract_sim_decomp. Returns SIM_Lambda and SIM_Z."""
    return {
        'SIM_Lambda': sim_data_torch['Lambda_l'][0],
        'SIM_Z': sim_data_torch['Z']
    }


def calc_all_structures_with_rescaling_single_view(
    col_filter,
    X_mean,
    X_sd,
    post_pred_train_samples,
    post_pred_test_samples,
    sim_decomp,
    train_idx,
    sim_outcome
):
    """
    Single-view analogue of calc_all_structures_with_rescaling for MatrixDecomp.

    Parameters
    ----------
    col_filter : BoolTensor (p,)
        Column filter from zero_variance_col_filter applied to the raw X.
    X_mean, X_sd : Tensor (p_filtered,)
        Column means and sds from normalize_tensor_by_col on the training set.
    post_pred_train_samples : dict
        Output of train_handler.predict with return_sites=['structure', 'Lambda_l0'].
    post_pred_test_samples : dict
        Output of test_handler.predict with return_sites=['y_pred'].
    sim_decomp : dict
        Output of extract_sim_decomp_single_view.
    train_idx : LongTensor
        Row indices of training observations in the full simulated dataset.
    sim_outcome : Tensor (n_test,)
        Ground-truth y values for the test set (normalized).
    """
    SIM_Lambda = sim_decomp['SIM_Lambda'][col_filter]   # (p_filtered, k_true)
    SIM_Z_train = sim_decomp['SIM_Z'][train_idx]        # (n_train, k_true)

    # Simulated structure in normalized space: (Z @ Lambda.T - X_mean) / X_sd
    SIM_struct = scale_sim_struct([SIM_Z_train @ SIM_Lambda.T], [X_mean], [X_sd])[0]

    # Posterior summaries for structure (training data)
    # train_handler.predict stores structure under the key "structure"
    structure_summary = summarise_post_samples(post_pred_train_samples['structure'])

    # Covariance in normalized space: diag(1/sd) @ (Lambda @ Lambda.T) @ diag(1/sd)
    # Use posterior mean of Lambda (samples shape: N_samples x p_filtered x k)
    Lambda_post_mean = post_pred_train_samples['Lambda_l0'].mean(dim=0).squeeze()
    est_cov = calc_cov(Lambda_post_mean)
    sim_cov_norm = scale_sim_cov(calc_cov(SIM_Lambda), X_sd, X_sd)

    # Outcome summaries (test set)
    # test_handler.predict stores outcome under the key Sites.y_pred = 'y_pred'
    pred_outcome_summary = summarise_post_samples(post_pred_test_samples['y_pred']) #['y']) #['y_pred'])

    return {
        'sim_struct': SIM_struct,
        'structure_summary': structure_summary,
        'est_cov': est_cov,
        'sim_cov_norm': sim_cov_norm,
        'sim_outcome': sim_outcome,
        'pred_outcome_summary': pred_outcome_summary,
    }


def eval_performance_single_view(
    sim_struct,
    structure_summary,
    est_cov,
    sim_cov_norm,
    sim_outcome,
    pred_outcome_summary
):
    """
    Single-view analogue of eval_performance for MatrixDecomp.
    Returns a one-row DataFrame with structure RSE, covariance Frobenius error,
    outcome MSE, and 95% credible interval coverage for structure and outcome.
    """
    structure_rse = eval_rse(structure_summary['mean'], sim_struct).item()
    structure_coverage = eval_post_coverage(
        sim_struct, structure_summary['q2.5'], structure_summary['q97.5']
    ).item()
    cov_norm_frob = eval_cov(est_cov, sim_cov_norm).item()
    outcome_mse = eval_mse(pred_outcome_summary['mean'], sim_outcome).item()
    outcome_coverage = eval_post_coverage(
        sim_outcome,
        pred_outcome_summary['q2.5'],
        pred_outcome_summary['q97.5']
    ).item()

    return pd.DataFrame([{
        'structure_rse':        structure_rse,
        'structure_95p_cov':    structure_coverage,
        'cov_norm_frob':        cov_norm_frob,
        'outcome_mse':          outcome_mse,
        'outcome_95p_cov':      outcome_coverage,
    }])


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    # Source - https://stackoverflow.com/a
    # Posted by bmcmenamin
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R), R


def align_tensor_shapes(target, input = None, additional = None):
    '''
    ensures that the target tensor shape is the same as the input
    applied to loading matrices, the target might have fewer cols than the input because
        of the shrinkage process prior. so pad the target
    '''
    if additional is None:
        num_to_pad = input.shape[1] - target.shape[1]
    elif input is None and additional is not None:
        num_to_pad = additional
        
    # print(num_to_pad)
    assert num_to_pad >= 0, "input has fewer cols than target. adjust k_max"

    return torch.nn.functional.pad(target, (0, num_to_pad))


def inv_gamma_moments(a, b):
    print(f'mean: {b / (a - 1)}')
    print(f'variance: {(b ** 2) / ((a - 1) ** 2) * (a - 2)}')