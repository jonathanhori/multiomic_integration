import gc
import os
import time
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import TensorDataset
import pyro
from pyro.optim import AdagradRMSProp
from pyro.infer import Trace_ELBO

from constants import Sites
from data_utils import (
    calc_all_structures_with_rescaling, eval_performance,
    extract_sim_decomp, normalize_tensor_by_col, zero_variance_col_filter,
    extract_sim_decomp_single_view, calc_all_structures_with_rescaling_single_view,
    eval_performance_single_view, calc_cov, scale_sim_cov, summarise_post_samples,
    extract_sim_decomp_shared, calc_all_structures_with_rescaling_shared,
    eval_performance_shared, eval_cov_shared, eval_rse,
)
from handler import ModelHandler
from model import SupMultiviewDecomp


def load_data(file_name_base,
              rep,
              files_for_condition,
              test_files_for_condition):
    """Read data

    TODO: read the data in addition to pulling out relevant reps. currently input
    is file name base, but should be the directory plus data config
    """

    # DATA LOADING HERE

    # Extract desired replicate
    print(file_name_base.format(rep))
    sim_data = files_for_condition.get(file_name_base.format(rep))
    if test_files_for_condition is not None:
        sim_data_test = test_files_for_condition.get(file_name_base.format(rep))
    else:
        sim_data_test = None

    # L = int(sim_data.get("L"))

    # X_l_list = sim_data.get("X_l")
    # y = sim_data.get("y")

    # X_l_list_test = sim_data_test.get("X_l")
    # y_test = sim_data_test.get("y")

    # k = int(sim_data.get("K"))
    # k_l_list = sim_data.get("K_l").int().tolist() #[int(k_l + k_delta) for k_l in sim_data.get("K_l").int().tolist()]

    # n = X_l_list[0].shape[0]
    # n_test = X_l_list_test[0].shape[0]

    # p_l = int(sim_data.get("p_l")[0])

    return sim_data, sim_data_test

    # return {
    #     'L': L,
    #     'X_l_list': X_l_list,
    #     'y': y,
    #     'X_l_list_test': X_l_list_test,
    #     'y_test': y_test,
    #     'k': k,
    #     'k_l_list': k_l_list,
    #     'n': n,
    #     'n_test': n_test,
    #     'p_l': p_l,
    #     'sim_data': X_l_sd_list
    # }


def load_rds(path):
    """Load a single RDS file via rpy2 and return a dict of tensors."""
    import rpy2.robjects as robjects
    from rpy2.robjects import default_converter, conversion
    from data_utils import r_to_torch

    conversion._converter = None
    conversion.set_conversion(default_converter)
    r_obj = robjects.r["readRDS"](path)
    return r_to_torch(r_obj)


##############################################################################
# Pipeline helpers — one set for all three models
#
# Every helper takes model_type, one of:
#   "single_view" : MatrixDecomp         (one view, supervised)
#   "shared"      : SupMultiviewShared   (shared factors only)
#   "viper"       : SupMultiviewDecomp   (shared + view-specific factors)
##############################################################################

MODEL_TYPES = ("single_view", "shared", "viper")

# Per-model settings used when the caller does not override them. These
# reproduce what each model's original helpers did.
_MODEL_DEFAULTS = {
    "single_view": {"opt": "adagrad",   "orthogonal_projection": False},
    "shared":      {"opt": "adagrad",   "orthogonal_projection": True},
    "viper":       {"opt": "scheduled", "orthogonal_projection": True},
}


def _check_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")


def process_data(sim_data, sim_data_test, training_split, training_size, seed=123,
                 model_type="viper"):
    """Train/test split, drop zero-variance columns, and standardise each view
    and the outcome using training-set statistics.

    Parameters
    ----------
    sim_data : dict of tensors with keys X_l (list of views), y, K, and the
        simulated structures (Z, Lambda_l, ...). Structures may be None for
        real data.
    sim_data_test : dict of tensors or None
        Separate test set; required when training_split=False.
    training_split : bool
        True splits sim_data at random; False uses sim_data_test as the test set.

    Returns
    -------
    train_subset, test_subset : TensorDataset(idx, X_1, ..., X_L, y)
    data_package : dict used by evaluate_fitted_model
    """
    _check_model_type(model_type)

    X_l_list = sim_data["X_l"]
    y = sim_data["y"]
    k = int(sim_data["K"])
    L = len(X_l_list)
    n = X_l_list[0].shape[0]

    if training_split:
        train_idx, test_idx = train_test_split(
            torch.arange(n), train_size=training_size, random_state=seed, shuffle=True
        )
        train_X_l_list = [X[train_idx] for X in X_l_list]
        test_X_l_list  = [X[test_idx]  for X in X_l_list]
        train_y = y[train_idx]
        test_y  = y[test_idx]
    else:
        assert sim_data_test is not None, "sim_data_test required when training_split=False"
        test_X_l_list = sim_data_test["X_l"]
        test_y        = sim_data_test["y"]
        train_idx = torch.arange(n)
        test_idx  = torch.arange(test_X_l_list[0].shape[0])
        train_X_l_list = X_l_list
        train_y = y

    # Clean: remove zero-variance columns and standardise, using training data
    X_l_list_column_filters = [zero_variance_col_filter(X) for X in train_X_l_list]
    train_X_l_list = [X[:, f] for X, f in zip(train_X_l_list, X_l_list_column_filters)]
    test_X_l_list  = [X[:, f] for X, f in zip(test_X_l_list,  X_l_list_column_filters)]

    clean_train = [normalize_tensor_by_col(X) for X in train_X_l_list]
    X_l_mean_list        = [d["means"]      for d in clean_train]
    X_l_sd_list          = [d["sds"]        for d in clean_train]
    train_X_l_list_clean = [d["data_clean"] for d in clean_train]

    clean_test = [normalize_tensor_by_col(X, m, s)
                  for X, m, s in zip(test_X_l_list, X_l_mean_list, X_l_sd_list)]
    test_X_l_list_clean = [d["data_clean"] for d in clean_test]

    y_mean = torch.mean(train_y)
    y_std  = torch.std(train_y)
    train_y_clean = (train_y - y_mean) / y_std
    test_y_clean  = (test_y  - y_mean) / y_std

    # MatrixDecomp expects a 1-D outcome; the multiview models take y as loaded.
    train_y_data = train_y_clean.squeeze() if model_type == "single_view" else train_y_clean
    test_y_data  = test_y_clean.squeeze()  if model_type == "single_view" else test_y_clean
    train_subset = TensorDataset(torch.arange(train_X_l_list_clean[0].shape[0]),
                                 *train_X_l_list_clean, train_y_data)
    test_subset  = TensorDataset(torch.arange(test_X_l_list_clean[0].shape[0]),
                                 *test_X_l_list_clean, test_y_data)

    # Simulated structures, with score rows restricted to the training set
    if model_type == "single_view":
        SIM_decomp = extract_sim_decomp_single_view(sim_data)
        train_SIM_decomp = SIM_decomp     # rows are selected in evaluation via train_idx
    elif model_type == "shared":
        SIM_decomp = extract_sim_decomp_shared(sim_data)
        train_SIM_decomp = {**SIM_decomp, "SIM_Z": SIM_decomp["SIM_Z"][train_idx]}
    else:
        SIM_decomp = extract_sim_decomp(sim_data)
        train_SIM_decomp = {**SIM_decomp,
                            "SIM_Z": SIM_decomp["SIM_Z"][train_idx],
                            "SIM_Phi_l_list": [Phi[train_idx] for Phi in SIM_decomp["SIM_Phi_l_list"]]}

    data_package = {
        "model_type":              model_type,
        "k":                       k,
        "L":                       L,
        "train_idx":               train_idx,
        "test_idx":                test_idx,
        "train_X_l_list_clean":    train_X_l_list_clean,
        "test_X_l_list_clean":     test_X_l_list_clean,
        "train_y_clean":           train_y_clean,
        "test_y_clean":            test_y_clean,
        "X_l_list_column_filters": X_l_list_column_filters,
        "X_l_mean_list":           X_l_mean_list,
        "X_l_sd_list":             X_l_sd_list,
        "SIM_decomp":              SIM_decomp,
        "train_SIM_decomp":        train_SIM_decomp,
    }
    if model_type == "viper":
        data_package["k_l_list"] = sim_data["K_l"].int().tolist()

    return train_subset, test_subset, data_package


def compute_svd_init(X_l_list, k):
    """Top-k SVD of horizontally stacked views for guide initialisation.

    Returns (Z_init, Lambda_l_init_list) scaled so that Z columns have
    unit variance and Lambda entries are O(1), consistent with the model priors.

    The factorisation used is X_concat ≈ (sqrt(N)*U_k) @ (s_k/sqrt(N)*V_k)^T,
    which keeps Z ~ N(0,I) and Lambda entries ~ O(sqrt(P/p_l)).

    Parameters
    ----------
    X_l_list : list of (N, p_l) float tensors
        Normalised training views (a list of one for single-view data).
    k : int
        Number of factors to extract.

    Returns
    -------
    Z_init : (N, k) tensor
    Lambda_l_init_list : list of (p_l, k) tensors
    """
    X_concat = torch.cat(X_l_list, dim=1).float()
    n = X_concat.shape[0]
    U, s, Vt = torch.linalg.svd(X_concat, full_matrices=False)

    Z_init = U[:, :k] * (n ** 0.5)          # (N, k), each column has var ≈ 1

    Lambda_l_init_list = []
    col = 0
    for X_l in X_l_list:
        p_l = X_l.shape[1]
        # (p_l, k): rows of Vt scaled by singular values, normalised by sqrt(N)
        Lambda_l_init_list.append(
            (Vt[:k, col:col + p_l].T * s[:k].unsqueeze(0)) / (n ** 0.5)
        )
        col += p_l

    return Z_init, Lambda_l_init_list


def _build_model(model_type, model_config, device="cpu", Z_init=None, Lambda_l_init_list=None):
    """Construct the factor model for model_type from model_config.

    model_config keys: k, dense_model (or "dense"), outcome_weight (optional),
    and for "viper" also k_l_list and include_view_factors.
    """
    k = model_config["k"]
    dense = model_config.get("dense_model", model_config.get("dense"))
    outcome_weight = model_config.get("outcome_weight", 1.0)

    if model_type == "single_view":
        from matrix_decomp import MatrixDecomp
        return MatrixDecomp(
            k, supervised=True, dense=dense, outcome_weight=outcome_weight, device=device,
            scores_init=Z_init,
            loadings_init=Lambda_l_init_list[0] if Lambda_l_init_list is not None else None,
        )
    if model_type == "shared":
        from multiview_shared import SupMultiviewShared
        return SupMultiviewShared(
            k, dense=dense, outcome_weight=outcome_weight, device=device,
            joint_scores_init=Z_init, joint_loadings_list_init=Lambda_l_init_list,
        )
    return SupMultiviewDecomp(
        k, model_config["k_l_list"],
        include_view_factors=model_config["include_view_factors"],
        dense=dense, outcome_weight=outcome_weight, device=device,
        joint_scores_init=Z_init, joint_loadings_list_init=Lambda_l_init_list,
    )


def _build_handler(mode, factor_model, train_config, model_type, opt=None, device="cpu"):
    """ModelHandler for global training (mode="train") or local inference
    (mode="predict").

    opt : "adagrad" uses AdagradRMSProp (requires eta/delta/t in train_config);
        "scheduled" uses StepLR with Adam (requires initial_lr/betas/
        lr_step_size/lr_decay_factor). None uses the model's default.
    """
    defaults = _MODEL_DEFAULTS[model_type]
    opt = opt or defaults["opt"]
    loss = Trace_ELBO(num_particles=train_config["num_particles"])

    if opt == "adagrad":
        optimizer = AdagradRMSProp({
            "eta":   train_config["eta"],
            "delta": train_config["delta"],
            "t":     train_config["t"],
        })
        return ModelHandler(mode, factor_model, opt=optimizer, loss=loss, opt_scheduler=None,
                            orthogonal_projection=defaults["orthogonal_projection"],
                            device=device)

    scheduler = pyro.optim.StepLR({
        "optimizer":  torch.optim.Adam,
        "optim_args": {"lr": train_config["initial_lr"], "betas": train_config["betas"]},
        "step_size":  train_config["lr_step_size"],
        "gamma":      train_config["lr_decay_factor"],
    })
    return ModelHandler(mode, factor_model, loss=loss, opt_scheduler=scheduler,
                        orthogonal_projection=defaults["orthogonal_projection"],
                        device=device)


def _minibatch_size(train_config, n):
    """Absolute minibatch size from minibatch_fraction (preferred) or
    minibatch_size, halved until it fits in n observations."""
    if "minibatch_fraction" in train_config:
        mb = max(1, int(n * train_config["minibatch_fraction"]))
    else:
        mb = int(train_config["minibatch_size"])
    while mb > n:
        mb //= 2
    return mb


def _convergence_kwargs(train_config):
    """Stopping-rule settings present in train_config; anything missing falls
    back to the ModelHandler.do_inference defaults."""
    keys = ("convergence_criterion", "window", "snr_threshold", "grad_norm_tol")
    return {key: train_config[key] for key in keys if key in train_config}


def train_model(model_config, train_config, train_subset, model_out_filename=None,
                model_type="viper", opt=None, verbose=False, write=True, device="cpu",
                svd_init=False, optuna_trial=None, optuna_step_offset=0):
    """Train a model globally on the training set.

    Parameters
    ----------
    model_out_filename : str or None
        Path stem for the saved state; nothing is written when None or write=False.
    opt : "adagrad", "scheduled" or None (the model's default: "scheduled"
        for viper, "adagrad" otherwise).
    svd_init : bool
        If True, initialise the score and shared-loading locations from the
        top-k SVD of the stacked training views before starting SVI.

    Returns
    -------
    factor_model, train_handler, model_state_dict
    """
    _check_model_type(model_type)
    k = model_config["k"]

    Z_init, Lambda_l_init_list = None, None
    svd_time = 0
    if svd_init:
        svd_t0 = time.time()
        # train_subset is TensorDataset(idx, X_1, ..., X_L, y); views are [1:-1]
        X_l_list = [t.to(device) for t in train_subset.tensors[1:-1]]
        Z_init, Lambda_l_init_list = compute_svd_init(X_l_list, k)
        svd_time = time.time() - svd_t0

    t0 = time.time()
    factor_model = _build_model(model_type, model_config, device, Z_init, Lambda_l_init_list)
    train_handler = _build_handler("train", factor_model, train_config, model_type, opt, device)

    train_handler.do_inference(
        train_dataset=train_subset,
        min_epochs=train_config["min_epochs"],
        epochs=train_config["max_epochs"],
        minibatch_flag=True,
        minibatch_size=_minibatch_size(train_config, len(train_subset)),
        variational_diff_func=np.mean,
        verbose=verbose,
        optuna_trial=optuna_trial,
        optuna_step_offset=optuna_step_offset,
        **_convergence_kwargs(train_config),
    )
    run_time = time.time() - t0

    model_state_dict = {
        "svd_time":                  svd_time,
        "inference_time":            run_time,
        "epochs":                    factor_model.total_epochs,
        "model_state_dict":          pyro.get_param_store().get_state(),
        "optimizer_state":           train_handler.opt.get_state(),
        "loss_history":              factor_model.loss_history,
        "param_convergence_history": factor_model.var_param_convergence_history,
        "config":                    train_config,
    }
    if write and model_out_filename:
        torch.save(model_state_dict, model_out_filename + ".pth")
        pyro.get_param_store().save(model_out_filename + "_paramstore.pth")

    return factor_model, train_handler, model_state_dict


def train_model_locally(factor_model, train_config, data_subset, model_out_filename=None,
                        model_type="viper", opt=None, verbose=False, device="cpu"):
    """Local inference on new observations: infers their scores with the
    global variational parameters frozen.

    Parameters
    ----------
    model_out_filename : str or None
        If given, the local state is saved to <stem>_local.pth and
        <stem>_local_paramstore.pth.

    Returns
    -------
    factor_model, test_handler, model_state_dict
    """
    _check_model_type(model_type)
    test_handler = _build_handler("predict", factor_model, train_config, model_type, opt, device)

    t0 = time.time()
    test_handler.do_inference(
        train_dataset=data_subset,
        min_epochs=train_config["min_epochs"],
        epochs=train_config["max_epochs"],
        minibatch_flag=True,
        minibatch_size=_minibatch_size(train_config, len(data_subset)),
        variational_diff_func=np.mean,
        verbose=verbose,
        **_convergence_kwargs(train_config),
    )
    runtime = time.time() - t0

    model_state_dict = {
        "inference_time":            runtime,
        "epochs":                    factor_model.local_epochs,
        "model_state_dict":          pyro.get_param_store().get_state(),
        "optimizer_state":           test_handler.opt.get_state(),
        "loss_history":              factor_model.local_loss_history,
        "param_convergence_history": factor_model.local_var_param_convergence_history,
        "config":                    train_config,
    }
    if model_out_filename:
        torch.save(model_state_dict, model_out_filename + "_local.pth")
        pyro.get_param_store().save(model_out_filename + "_local_paramstore.pth")

    return factor_model, test_handler, model_state_dict


def evaluate_fitted_model(rep_config, data_package, train_handler, test_handler,
                          global_train_state, local_train_state,
                          N_POSTERIOR_SAMPLES, metric_out_filename=None,
                          model_type="viper"):
    """Draw posterior samples and compute evaluation metrics after global
    training and local inference.

    Parameters
    ----------
    rep_config : dict or None
        Simulation condition (n, p_l, snr_x, snr_y, rep, sparsity, k_delta),
        added as columns to the metric table. None adds no condition columns.
    metric_out_filename : unused; kept for existing call sites.

    Returns
    -------
    eval_metric_table : DataFrame of structure RSE, coverage, outcome MSE, ...
    cov_metric_table : DataFrame of covariance error per view pair (shared
        model only; None otherwise)
    post_samples : posterior samples on the training set (Z, beta, y, Lambda_l, ...)
    post_samples_test : posterior samples on the test set (Z_pred, y_pred)
    """
    _check_model_type(model_type)
    L                       = data_package["L"]
    train_X_l_list_clean    = data_package["train_X_l_list_clean"]
    test_X_l_list_clean     = data_package["test_X_l_list_clean"]
    test_y_clean            = data_package["test_y_clean"].squeeze()
    X_l_list_column_filters = data_package["X_l_list_column_filters"]
    X_l_mean_list           = data_package["X_l_mean_list"]
    X_l_sd_list             = data_package["X_l_sd_list"]

    # Posterior samples on the training set (global model) and test set (local model)
    sites = [Sites.Z, Sites.y, Sites.beta] + [Sites.Lambda_l.format(l=l) for l in range(L)]
    if model_type == "single_view":
        sites += ["structure"]
    elif model_type == "viper":
        sites += ([Sites.joint_structure_l.format(l=l) for l in range(L)]
                  + [Sites.view_structure_l.format(l=l) for l in range(L)])
    test_sites = [Sites.y_pred, Sites.Z_pred]

    post_samples      = train_handler.predict(train_X_l_list_clean, N_POSTERIOR_SAMPLES, sites)
    post_samples_test = test_handler.predict(test_X_l_list_clean,   N_POSTERIOR_SAMPLES, test_sites)
    # Sim data from R (via rpy2) is always on CPU; move posterior samples to CPU so all
    # downstream comparisons (eval_rse, numpy conversions, etc.) see consistent devices.
    post_samples      = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                         for k, v in post_samples.items()}
    post_samples_test = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                         for k, v in post_samples_test.items()}

    # Compare estimated structures (targeting mean 0 var 1 data) with
    # RESCALED simulated structures
    cov_metric_table = None
    if model_type == "single_view":
        eval_structures = calc_all_structures_with_rescaling_single_view(
            X_l_list_column_filters[0], X_l_mean_list[0], X_l_sd_list[0],
            post_samples, post_samples_test,
            data_package["SIM_decomp"], data_package["train_idx"], test_y_clean,
        )
        eval_metric_table = eval_performance_single_view(**eval_structures)
    elif model_type == "shared":
        eval_structures = calc_all_structures_with_rescaling_shared(
            L, X_l_list_column_filters, X_l_mean_list, X_l_sd_list,
            post_samples, post_samples_test, data_package["train_SIM_decomp"], test_y_clean,
            post_loading_samples=post_samples,
        )
        eval_metric_table = eval_performance_shared(
            **{k: v for k, v in eval_structures.items() if k not in ("est_cov", "sim_cov")})
        if "est_cov" in eval_structures:
            cov_metric_table = eval_cov_shared(
                eval_structures["est_cov"], eval_structures["sim_cov"], L)
    else:
        eval_structures = calc_all_structures_with_rescaling(
            L, X_l_list_column_filters, X_l_mean_list, X_l_sd_list,
            post_samples, post_samples_test, data_package["train_SIM_decomp"], test_y_clean,
        )
        eval_metric_table = eval_performance(**eval_structures)

    # Per-view RSE of test data reconstruction (Z_test @ Lambda_l^T vs X_l_test).
    # VIPER's test reconstruction also needs the view-specific scores, so skip it there.
    if model_type != "viper":
        mean_Z_pred = post_samples_test[Sites.Z_pred].mean(dim=0).squeeze()
        eval_metric_table = eval_metric_table.assign(test_rse=[
            eval_rse(
                mean_Z_pred @ post_samples[Sites.Lambda_l.format(l=l)].mean(dim=0).squeeze().mT,
                test_X_l_list_clean[l],
            ).item()
            for l in range(L)
        ])

    if rep_config is not None:
        condition_cols = dict(
            n=rep_config["n"],         p=rep_config["p_l"],
            snr_x=rep_config["snr_x"], snr_y=rep_config["snr_y"],
            rep=rep_config["rep"],     sparsity=rep_config["sparsity"],
            k_delta=rep_config["k_delta"],
            runtime=global_train_state["inference_time"],
            predict_runtime=local_train_state["inference_time"],
            svd_time=global_train_state.get("svd_time", 0),
        )
        eval_metric_table = eval_metric_table.assign(**condition_cols)
        if cov_metric_table is not None:
            cov_metric_table = cov_metric_table.assign(**condition_cols)

    return eval_metric_table, cov_metric_table, post_samples, post_samples_test


def load_model(model_out_filename, model_config, n_train, train_config,
               model_type="viper", opt=None, device="cpu"):
    """Restore a saved model for prediction without retraining.

    Sets the two instance attributes that are not persisted to disk but are
    required before any forward/guide or predict_forward/predict_guide call:
      - model.n      : training-set size (needed by the guide to size loc_Z/scale_Z init tensors)
      - model.params : reference to the param store (needed by predict_forward/predict_guide)
    """
    _check_model_type(model_type)
    pyro.clear_param_store()
    state = torch.load(model_out_filename + "_paramstore.pth",
                       map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)

    factor_model = _build_model(model_type, model_config, device)
    factor_model.n = n_train
    factor_model.params = pyro.get_param_store()

    train_handler = _build_handler("train", factor_model, train_config, model_type, opt, device)
    global_state = torch.load(model_out_filename + ".pth", weights_only=False)
    return factor_model, train_handler, global_state


##############################################################################
# Model-specific names, kept so existing scripts and notebooks run unchanged.
# New code should call the helpers above with model_type.
##############################################################################

svd_init_shared = compute_svd_init


def process_data_shared(sim_data, sim_data_test, training_split, training_size, seed=123):
    return process_data(sim_data, sim_data_test, training_split, training_size, seed,
                        model_type="shared")


def train_model_shared(model_config, train_config, train_subset,
                       model_out_filename, opt="adagrad", verbose=False, write=True,
                       device="cpu", optuna_trial=None, optuna_step_offset=0,
                       svd_init=False):
    return train_model(model_config, train_config, train_subset, model_out_filename,
                       model_type="shared", opt=opt, verbose=verbose, write=write,
                       device=device, svd_init=svd_init, optuna_trial=optuna_trial,
                       optuna_step_offset=optuna_step_offset)


def train_model_locally_shared(factor_model, train_config, data_subset,
                               opt="adagrad", verbose=False, device="cpu"):
    return train_model_locally(factor_model, train_config, data_subset,
                               model_type="shared", opt=opt, verbose=verbose, device=device)


def evaluate_fitted_model_shared(rep_config, data_package,
                                 train_handler, test_handler,
                                 global_train_state, local_train_state,
                                 N_POSTERIOR_SAMPLES, metric_out_filename):
    return evaluate_fitted_model(rep_config, data_package, train_handler, test_handler,
                                 global_train_state, local_train_state,
                                 N_POSTERIOR_SAMPLES, model_type="shared")


def load_model_shared(model_out_filename, model_config, n_train, train_config,
                      opt="adagrad", device="cpu"):
    return load_model(model_out_filename, model_config, n_train, train_config,
                      model_type="shared", opt=opt, device=device)


def process_data_single_view(sim_data, training_size, seed=123,
                             training_split=True, sim_data_test=None):
    """Returns the package from process_data plus the single-view key names
    (X_train, y_train, col_filter, ...) that older notebooks read directly."""
    train_subset, test_subset, data_package = process_data(
        sim_data, sim_data_test, training_split, training_size, seed,
        model_type="single_view")
    data_package.update({
        "X_train":    data_package["train_X_l_list_clean"][0],
        "X_test":     data_package["test_X_l_list_clean"][0],
        "y_train":    data_package["train_y_clean"].squeeze(),
        "y_test":     data_package["test_y_clean"].squeeze(),
        "col_filter": data_package["X_l_list_column_filters"][0],
        "X_mean":     data_package["X_l_mean_list"][0],
        "X_sd":       data_package["X_l_sd_list"][0],
        "sim_decomp": data_package["SIM_decomp"],
    })
    return train_subset, test_subset, data_package


def train_model_single_view(model_config, train_config, train_subset,
                            opt="adagrad", scheduled=False, verbose=False):
    """Returns (factor_model, train_handler, runtime in seconds)."""
    factor_model, train_handler, state = train_model(
        model_config, train_config, train_subset, model_type="single_view",
        opt="scheduled" if scheduled else opt, verbose=verbose, write=False)
    return factor_model, train_handler, state["inference_time"]


def train_model_locally_single_view(factor_model, train_config, test_subset,
                                    opt="adagrad", verbose=False):
    """Returns (test_handler, runtime in seconds)."""
    _, test_handler, state = train_model_locally(
        factor_model, train_config, test_subset, model_type="single_view",
        opt=opt, verbose=verbose)
    return test_handler, state["inference_time"]


def evaluate_fitted_model_single_view(train_handler, test_handler, data_package,
                                      n_posterior_samples=300):
    """Returns the metric table only."""
    eval_metric_table, _, _, _ = evaluate_fitted_model(
        None, data_package, train_handler, test_handler, None, None,
        n_posterior_samples, model_type="single_view")
    return eval_metric_table


##############################################################################
# Cross-validation helpers
##############################################################################

def make_cv_folds(n, n_folds=5, seed=123):
    """Return list of (train_idx, val_idx) LongTensor pairs for k-fold CV."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (torch.tensor(tr, dtype=torch.long), torch.tensor(va, dtype=torch.long))
        for tr, va in kf.split(range(n))
    ]


def train_eval_cv_fold(X_raw, y_raw, train_idx, val_idx, model_config, train_config,
                       n_posterior_samples=100, opt="adagrad", verbose=False):
    """Train a MatrixDecomp on one CV fold and evaluate on the held-out fold.

    Designed to be called either directly (sequential) or inside an isolated
    worker process (parallel via run_cv_grid).  Each call resets the Pyro param
    store so folds do not share variational parameters.

    For real data (no ground truth), the training X is used as the reference
    structure and its sample covariance stands in for the simulated covariance,
    mirroring the pattern in the full real-data evaluation.

    Parameters
    ----------
    X_raw : (n, p) CPU float tensor -- pre-filtered but NOT normalised.
    y_raw : (n,) CPU float tensor -- raw (e.g. log-transformed), NOT z-scored.

    Returns
    -------
    dict with keys from eval_performance_single_view plus 'epochs' / 'local_epochs'.
    """
    X_tr_raw = X_raw[train_idx]
    X_va_raw = X_raw[val_idx]
    y_tr_raw = y_raw[train_idx]
    y_va_raw = y_raw[val_idx]

    # Normalise using training-fold statistics only -- no cross-fold leakage
    train_norm = normalize_tensor_by_col(X_tr_raw)
    X_tr   = train_norm["data_clean"]
    X_mean = train_norm["means"]
    X_sd   = train_norm["sds"]

    val_norm = normalize_tensor_by_col(X_va_raw, X_mean, X_sd)
    X_va = val_norm["data_clean"]

    y_mean = torch.mean(y_tr_raw)
    y_std  = torch.std(y_tr_raw)
    y_tr   = ((y_tr_raw - y_mean) / y_std).squeeze()
    y_va   = ((y_va_raw - y_mean) / y_std).squeeze()

    train_ds = TensorDataset(torch.arange(len(train_idx)), X_tr, y_tr)
    val_ds   = TensorDataset(torch.arange(len(val_idx)),   X_va, y_va)

    pyro.clear_param_store()

    factor_model, train_handler, _ = train_model(
        model_config, train_config, train_ds, model_type="single_view",
        opt=opt, verbose=verbose, write=False,
    )
    _, val_handler, _ = train_model_locally(
        factor_model, train_config, val_ds, model_type="single_view",
        opt=opt, verbose=verbose,
    )

    post_train = train_handler.predict(
        [X_tr], n_posterior_samples,
        return_sites=["structure", Sites.Lambda_l.format(l=0)],
    )
    post_val = val_handler.predict(
        [X_va], n_posterior_samples,
        return_sites=["y_pred"],
    )

    Lambda_post_mean = post_train[Sites.Lambda_l.format(l=0)].mean(dim=0).squeeze()
    est_cov      = calc_cov(Lambda_post_mean)
    sample_cov   = X_tr.T @ X_tr / X_tr.shape[0]
    sim_cov_norm = scale_sim_cov(sample_cov, X_sd, X_sd)

    metrics_df = eval_performance_single_view(
        sim_struct           = X_tr,
        structure_summary    = summarise_post_samples(post_train["structure"]),
        est_cov              = est_cov,
        sim_cov_norm         = sim_cov_norm,
        sim_outcome          = y_va,
        pred_outcome_summary = summarise_post_samples(post_val["y_pred"]),
    )

    result = metrics_df.iloc[0].to_dict()
    result["epochs"]       = factor_model.total_epochs
    result["local_epochs"] = factor_model.local_epochs
    return result


def _fold_worker(args):
    """Single-argument entry point for multiprocessing.Pool.map.

    Wraps train_eval_cv_fold with error capture so one failed fold does not
    abort the pool.  Returns a dict with '_error' key on failure.
    """
    try:
        return train_eval_cv_fold(*args)
    except Exception as exc:
        return {"_error": str(exc)}


def run_cv_grid(X_raw, y_raw, model_config, train_grid, train_grid_names,
                n_folds=5, seed=123, n_posterior_samples=100,
                opt="adagrad", n_workers=1, verbose=False):
    """Run k-fold CV over all configs in train_grid.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes per config.
        ``n_workers=1`` (default) runs sequentially in the current process --
        straightforward to debug and safe in any environment.
        ``n_workers=n_folds`` fits all folds concurrently using
        torch.multiprocessing (spawn context); each worker gets an isolated
        Python interpreter and a fresh Pyro param store so there is no
        global-state interference between folds.
        Values between 1 and n_folds control the degree of parallelism.

    Returns
    -------
    pd.DataFrame with columns: config_idx, fold, eval_performance_single_view
    metrics, epochs, local_epochs, and one column per train_grid_names entry.
    """
    import pandas as pd

    folds = make_cv_folds(X_raw.shape[0], n_folds=n_folds, seed=seed)
    rows  = []

    for i, train_params in enumerate(train_grid):
        train_config = dict(zip(train_grid_names, train_params))
        if opt == "adagrad":
            config_str = (f"eta={train_config.get('eta')}  "
                          f"delta={train_config.get('delta')}  "
                          f"t={train_config.get('t')}  "
                          f"bs={train_config.get('minibatch_size')}")
        else:
            config_str = (f"lr={train_config.get('initial_lr')}  "
                          f"step={train_config.get('lr_step_size')}  "
                          f"decay={train_config.get('lr_decay_factor')}  "
                          f"bs={train_config.get('minibatch_size')}")
        print(f"[{i+1}/{len(train_grid)}] {config_str}", flush=True)

        fold_args = [
            (X_raw, y_raw, tr_idx, va_idx, model_config, train_config,
             n_posterior_samples, opt, verbose)
            for tr_idx, va_idx in folds
        ]

        if n_workers == 1:
            # Sequential: run folds in the current process, one at a time
            fold_results = [_fold_worker(a) for a in fold_args]
        else:
            # Parallel: up to n_workers folds at once via joblib/loky.
            #
            # loky is joblib's default backend and is specifically designed
            # for interactive environments (Jupyter, Quarto).  Unlike
            # multiprocessing.spawn, it does NOT try to re-import __main__,
            # so it works correctly when called from a notebook.  It also
            # propagates the parent's sys.path to workers automatically.
            from joblib import Parallel, delayed
            fold_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_fold_worker)(a) for a in fold_args
            )

        for fold_idx, m in enumerate(fold_results):
            if m is None:
                continue
            if "_error" in m:
                print(f"  fold {fold_idx} FAILED: {m['_error']}")
                continue
            row = {"config_idx": i, "fold": fold_idx, **m}
            row.update({
                k: str(v) if isinstance(v, tuple) else v
                for k, v in train_config.items()
            })
            rows.append(row)
            print(
                f"  fold {fold_idx}  outcome_mse={m['outcome_mse']:.4f}  "
                f"outcome_95p_cov={m['outcome_95p_cov']:.2f}",
                flush=True,
            )

    return pd.DataFrame(rows)



##############################################################################
# Cross-validation helpers — SupMultiviewShared
##############################################################################

def train_eval_cv_fold_shared(X_l_raw_list, y_raw, train_idx, val_idx,
                               model_config, train_config,
                               n_posterior_samples=100, opt="adagrad", verbose=False,
                               sim_decomp=None, column_filters=None,
                               minibatch_grid=None):
    """Train SupMultiviewShared on one CV fold and evaluate on the held-out fold.

    Normalises each view independently using training-fold statistics only.
    No files are written; param store is cleared at the start of each call.

    Parameters
    ----------
    X_l_raw_list : list of (n, p_l) float tensors
        Pre-filtered (zero-variance columns removed) but NOT normalised.
    y_raw : (n,) float tensor
    sim_decomp : dict, optional
        Output of extract_sim_decomp_shared. When supplied alongside
        column_filters, the fold additionally reports cov_frob, the mean
        normalised Frobenius error of the reconstructed within- and
        cross-view covariance matrices.
    column_filters : list of bool tensors, optional
        Per-view column filters matching X_l_raw_list, used to align
        sim_decomp["SIM_Lambda_l_list"] with the fitted loadings.
    minibatch_grid : list of int, optional
        Original grid of minibatch sizes. If the requested minibatch_size
        exceeds the fold training-set size, it is replaced with the largest
        grid value that fits (falling back to the training-set size itself
        if no grid value fits).

    Returns
    -------
    dict with keys: outcome_mse, epochs, local_epochs, minibatch_size
    (and cov_frob when sim_decomp/column_filters are supplied).
    """
    n_train = len(train_idx)

    if train_config.get("minibatch_size") is not None and train_config["minibatch_size"] > n_train:
        if minibatch_grid:
            valid = sorted(m for m in minibatch_grid if m <= n_train)
            new_mb = valid[-1] if valid else n_train
        else:
            new_mb = n_train
        train_config = {**train_config, "minibatch_size": int(new_mb)}

    X_l_tr_raw = [X[train_idx] for X in X_l_raw_list]
    X_l_va_raw = [X[val_idx]   for X in X_l_raw_list]
    y_tr_raw   = y_raw[train_idx]
    y_va_raw   = y_raw[val_idx]

    train_norms = [normalize_tensor_by_col(X) for X in X_l_tr_raw]
    X_l_tr   = [d["data_clean"] for d in train_norms]
    X_l_mean = [d["means"]      for d in train_norms]
    X_l_sd   = [d["sds"]        for d in train_norms]

    val_norms = [normalize_tensor_by_col(X, m, s)
                 for X, m, s in zip(X_l_va_raw, X_l_mean, X_l_sd)]
    X_l_va = [d["data_clean"] for d in val_norms]

    y_mean = torch.mean(y_tr_raw)
    y_std  = torch.std(y_tr_raw)
    y_tr   = ((y_tr_raw - y_mean) / y_std).squeeze()
    y_va   = ((y_va_raw - y_mean) / y_std).squeeze()

    train_ds = TensorDataset(torch.arange(len(train_idx)), *X_l_tr, y_tr)
    val_ds   = TensorDataset(torch.arange(len(val_idx)),   *X_l_va, y_va)

    pyro.clear_param_store()

    factor_model, train_handler, global_state = train_model(
        model_config, train_config, train_ds, model_type="shared",
        opt=opt, verbose=verbose, write=False,
    )

    factor_model, val_handler, local_state = train_model_locally(
        factor_model, train_config, val_ds, model_type="shared",
        opt=opt, verbose=verbose,
    )

    post_val     = val_handler.predict(X_l_va, n_posterior_samples, [Sites.y_pred])
    pred_mean    = summarise_post_samples(post_val[Sites.y_pred])["mean"]
    outcome_mse  = float(torch.mean((pred_mean - y_va) ** 2).item())

    train_time = float(global_state["inference_time"])
    local_time = float(local_state["inference_time"])
    metrics = {
        "outcome_mse":   outcome_mse,
        "epochs":        factor_model.total_epochs,
        "local_epochs":  factor_model.local_epochs,
        "train_time":    train_time,
        "local_time":    local_time,
        "total_time":    train_time + local_time,
        "effective_minibatch_size": _minibatch_size(train_config, n_train),
    }

    if sim_decomp is not None and column_filters is not None:
        L = len(X_l_raw_list)
        lambda_sites = [Sites.Lambda_l.format(l=l) for l in range(L)]
        post_train = train_handler.predict(X_l_tr, n_posterior_samples, lambda_sites)
        est_Lambda_list = [
            post_train[Sites.Lambda_l.format(l=l)].mean(dim=0).squeeze().cpu()
            for l in range(L)
        ]
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
                sim_cov[(l, m)] = scale_sim_cov(raw_cov, X_l_sd[l], X_l_sd[m])
        cov_table = eval_cov_shared(est_cov, sim_cov, L)
        metrics["cov_frob"] = float(cov_table["cov_frob"].mean())
        del post_train, est_Lambda_list, sim_Lambda_list, est_cov, sim_cov, cov_table

    # Free per-fold memory so long sequential runs (or loky workers that get
    # reused across folds) don't accumulate Pyro/Torch tensors.
    del factor_model, train_handler, val_handler, post_val, pred_mean
    del X_l_tr, X_l_va, X_l_tr_raw, X_l_va_raw, train_ds, val_ds
    del train_norms, val_norms, X_l_mean, X_l_sd
    pyro.clear_param_store()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return metrics


def _fold_worker_shared(args):
    """Single-argument entry point for multiprocessing.Pool.map (shared model)."""
    try:
        return train_eval_cv_fold_shared(*args)
    except Exception as exc:
        return {"_error": str(exc)}


def run_cv_grid_shared(X_l_raw_list, y_raw, model_config, train_grid, train_grid_names,
                       n_folds=5, seed=123, n_posterior_samples=100,
                       opt="adagrad", n_workers=1, verbose=False,
                       sim_decomp=None, column_filters=None,
                       minibatch_grid=None,
                       checkpoint_path=None, resume=True):
    """Run k-fold CV over all configs in train_grid for SupMultiviewShared.

    Parameters
    ----------
    X_l_raw_list : list of (n, p_l) float tensors
        Pre-filtered but NOT normalised, one tensor per view.
    y_raw : (n,) float tensor
    n_workers : int
        1 = sequential; >1 = parallel via joblib/loky (n_workers folds at once).
    sim_decomp, column_filters : optional
        Forwarded to train_eval_cv_fold_shared so that each fold also reports
        cov_frob (mean Frobenius error of within- and cross-view covariance
        reconstruction).
    minibatch_grid : list of int, optional
        Forwarded to train_eval_cv_fold_shared. If a config's minibatch_size
        exceeds the fold training-set size, it is dropped down to the largest
        grid value that fits.
    checkpoint_path : str, optional
        If set, the in-progress DataFrame is written to this CSV after every
        config completes. Crash-safe for long sweeps.
    resume : bool, default True
        If True and `checkpoint_path` exists, load it and skip any
        `config_idx` already recorded. Set to False to start fresh and
        overwrite the file. Configs are identified by their index in
        `train_grid`, so the grid order must be stable across resumes.

    Returns
    -------
    pd.DataFrame with columns: config_idx, fold, outcome_mse, epochs,
    local_epochs, minibatch_size (effective), and one column per
    train_grid_names entry. cov_frob and train_time/local_time/total_time
    are included when sim_decomp is supplied / timings are recorded.
    """
    import pandas as pd

    n     = X_l_raw_list[0].shape[0]
    folds = make_cv_folds(n, n_folds=n_folds, seed=seed)
    rows  = []
    done_config_ids = set()

    if checkpoint_path is not None and resume and os.path.exists(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        # Only treat configs with all folds present as "done"; partials get redone.
        fold_counts     = prior.groupby("config_idx").size()
        done_config_ids = set(int(c) for c, k in fold_counts.items() if k >= n_folds)
        rows.extend(prior[prior["config_idx"].isin(done_config_ids)].to_dict("records"))
        print(f"Resuming from {checkpoint_path}: {len(done_config_ids)} configs "
              f"already complete, {len(train_grid) - len(done_config_ids)} to go.",
              flush=True)

    def _flush(rs):
        if checkpoint_path is None:
            return
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        pd.DataFrame(rs).to_csv(checkpoint_path, index=False)

    for i, train_params in enumerate(train_grid):
        if i in done_config_ids:
            continue
        train_config = dict(zip(train_grid_names, train_params))
        config_str = "  ".join(f"{k}={v}" for k, v in train_config.items())
        print(f"[{i+1}/{len(train_grid)}] {config_str}", flush=True)

        fold_args = [
            (X_l_raw_list, y_raw, tr_idx, va_idx, model_config, train_config,
             n_posterior_samples, opt, verbose,
             sim_decomp, column_filters, minibatch_grid)
            for tr_idx, va_idx in folds
        ]

        if n_workers == 1:
            fold_results = [_fold_worker_shared(a) for a in fold_args]
        else:
            from joblib import Parallel, delayed
            fold_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_fold_worker_shared)(a) for a in fold_args
            )

        for fold_idx, m in enumerate(fold_results):
            if m is None:
                continue
            if "_error" in m:
                print(f"  fold {fold_idx} FAILED: {m['_error']}")
                continue
            row = {"config_idx": i, "fold": fold_idx, **m}
            row.update({
                k: str(v) if isinstance(v, tuple) else v
                for k, v in train_config.items()
            })
            rows.append(row)
            cov_str  = (f"  cov_frob={m['cov_frob']:.4f}" if "cov_frob" in m else "")
            time_str = (f"  train_time={m['train_time']:.1f}s"
                        if "train_time" in m else "")
            print(f"  fold {fold_idx}  outcome_mse={m['outcome_mse']:.4f}"
                  f"{cov_str}{time_str}",
                  flush=True)

        # Drop the parallel fold-result list before flushing/next config so any
        # Pyro/Torch state captured in worker exceptions doesn't accumulate.
        del fold_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        _flush(rows)

    return pd.DataFrame(rows)