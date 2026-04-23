import time
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import TensorDataset
import pyro
from pyro.infer import Trace_ELBO

from constants import Sites
from data_utils import (
    calc_all_structures_with_rescaling, eval_performance,
    extract_sim_decomp, normalize_tensor_by_col, zero_variance_col_filter,
    extract_sim_decomp_single_view, calc_all_structures_with_rescaling_single_view,
    eval_performance_single_view, calc_cov, scale_sim_cov, summarise_post_samples,
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
    
def process_data(sim_data,
                 sim_data_test,
                training_split,
                training_size,
                seed = 123):
    """After loading input data, perform train/test split if necessary, and 
    create data loaders"""

    L = int(sim_data.get("L"))
    
    X_l_list = sim_data.get("X_l")
    y = sim_data.get("y")
    
    k = int(sim_data.get("K"))
    k_l_list = sim_data.get("K_l").int().tolist() #[int(k_l + k_delta) for k_l in sim_data.get("K_l").int().tolist()]
    
    n = X_l_list[0].shape[0]
    
    p_l = int(sim_data.get("p_l")[0])

    # Perform train/test split if necessary
    if training_split:
        train_idx, test_idx = train_test_split(
            torch.arange(X_l_list[0].shape[0]),
            train_size = training_size,
            random_state = seed,
            shuffle = True
        )
        train_X_l_list = [X[train_idx, :] for X in X_l_list]
        test_X_l_list = [X[test_idx, :] for X in X_l_list]
        
        train_y = y[train_idx]
        test_y = y[test_idx]
    else:
        X_l_list_test = sim_data_test.get("X_l")
        y_test = sim_data_test.get("y")
        n_test = X_l_list_test[0].shape[0]
    
        train_idx = torch.arange(n)
        test_idx = torch.arange(n_test)
        
        train_X_l_list = X_l_list
        test_X_l_list = X_l_list_test
        
        train_y = y
        test_y = y_test

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
    
    # Extract simulated structures
    SIM_decomp = extract_sim_decomp(sim_data)
    train_SIM_decomp = {k: v for k, v in SIM_decomp.items()}
    train_SIM_decomp['SIM_Z'] = train_SIM_decomp['SIM_Z'][train_idx, :]
    train_SIM_decomp['SIM_Phi_l_list'] = [Phi[train_idx, :] for Phi in train_SIM_decomp['SIM_Phi_l_list']]
    
    # contains items needed for later steps ==> evaluation 
    input_data_package = {
        'k': k,
        'k_l_list': k_l_list,
        'L': L,
        'train_X_l_list_clean': train_X_l_list_clean,
        'test_X_l_list_clean': test_X_l_list_clean,
        'train_y_clean': train_y_clean,
        'test_y_clean': test_y_clean,
        'X_l_list_column_filters': X_l_list_column_filters,
        'X_l_mean_list': X_l_mean_list,
        'X_l_sd_list': X_l_sd_list,
        'SIM_decomp': SIM_decomp,
        'train_SIM_decomp': train_SIM_decomp
    }
    
    return train_subset, test_subset, input_data_package


def train_model(model_config,
                train_config,
                train_subset,
                model_out_filename,
                verbose = False,
                write = True,
                # mode = "global",
                out_dir = "./model_output/"):
    """Train the supervised multiview decomposition model
    """
    
    k = model_config['k']
    k_l_list = model_config['k_l_list']
    include_view_factors = model_config['include_view_factors']
    dense_model = model_config['dense_model']
    
    initial_lr = train_config['initial_lr']
    betas = train_config['betas']
    num_particles = train_config['num_particles']
    lr_step_size = train_config['lr_step_size']
    lr_decay_factor = train_config['lr_decay_factor']
    mini_size = train_config['minibatch_size']
    min_epochs = train_config['min_epochs']
    max_epochs = train_config['max_epochs']
    
    
    ###############################################
    factor_model = SupMultiviewDecomp(
        k,
        k_l_list,
        include_view_factors = include_view_factors,
        dense = dense_model
    )

    # OPT = Adam({"lr": initial_lr})
    opt_args = {"lr": initial_lr, "betas": betas}
    # OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    LOSS = Trace_ELBO(num_particles = num_particles)
    scheduler_args = {
        'optimizer': torch.optim.Adam, 
        'optim_args': opt_args,
        'step_size': lr_step_size,
        'gamma': lr_decay_factor #,
        # 'verbose': True
    }
    scheduler = pyro.optim.StepLR(scheduler_args)
    
    train_handler = ModelHandler(
        "train", 
        factor_model,
        # opt = OPT, # Opt is not needed if opt_scheduler is provided
        loss = LOSS,
        orthogonal_projection = True,
        opt_scheduler = scheduler
        )

    try:            
        t0 = time.time()
        svi = train_handler.do_inference(
            train_dataset = train_subset,
            min_epochs = min_epochs,
            epochs = max_epochs,
            minibatch_flag = True,
            minibatch_size = mini_size,
            variational_diff_func = np.mean,
            verbose = verbose
            )
        t1 = time.time()
        run_minibatch = t1 - t0
        
        #################
        # Inference finished, save and evaluate
        # Save model parameters

        model_state_dict = {
            "inference_time": run_minibatch,
            "epochs": factor_model.total_epochs,
            # "model_param_store": pyro.get_param_store(),
            "model_state_dict": pyro.get_param_store().get_state(),
            "optimizer_state": train_handler.opt.get_state(),
            "loss_history": factor_model.loss_history,
            "param_convergence_history": factor_model.var_param_convergence_history,
            "config": train_config              
        }
        
        if write:
            print("Exporting:")
            print(model_out_filename)
            torch.save(model_state_dict, model_out_filename + ".pth")
            pyro.get_param_store().save(model_out_filename + "_paramstore.pth") #os.path.join(model_out_path, model_out_filename + "_paramstore.pth"))
        
    except Exception as e:
        print(e)
    
    return factor_model, train_handler, model_state_dict


def train_model_locally(factor_model,
                        train_config,
                        data_subset,
                        model_out_filename,
                        out_dir = "./model_output/"):
    """After the model has been globally trained, perform local training on new observations
    """
    initial_lr = train_config['initial_lr']
    betas = train_config['betas']
    num_particles = train_config['num_particles']
    lr_step_size = train_config['lr_step_size']
    lr_decay_factor = train_config['lr_decay_factor']
    mini_size = train_config['minibatch_size']
    min_epochs = train_config['min_epochs']
    max_epochs = train_config['max_epochs']
    
     #################
    # Inference for predictive model
    print('local training')
    # LOCAL_OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    # LOCAL_LOSS = Trace_ELBO(num_particles = 1)
    opt_args = {"lr": initial_lr, "betas": betas}
    # OPT = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    LOCAL_LOSS = Trace_ELBO(num_particles = num_particles)
    scheduler_args = {
        'optimizer': torch.optim.Adam, 
        'optim_args': opt_args,
        'step_size': lr_step_size,
        'gamma': lr_decay_factor,
        'verbose': True
    }
    scheduler = pyro.optim.StepLR(scheduler_args)
    
    test_handler = ModelHandler("predict",
                                factor_model,
                                loss = LOCAL_LOSS,
                                orthogonal_projection = True,
                                opt_scheduler = scheduler)  
    
    t0 = time.time()
    svi = test_handler.do_inference(
        data_subset,
        min_epochs = min_epochs,
        epochs = max_epochs,
        minibatch_flag = True,
        minibatch_size = mini_size,
        verbose = False
        )
    t1 = time.time()
    test_run_minibatch = t1 - t0
    
    model_state_dict = {
        "inference_time": test_run_minibatch,
        "epochs": factor_model.local_epochs,
        # "model_param_store": pyro.get_param_store(),
        "model_state_dict": pyro.get_param_store().get_state(),
        "optimizer_state": test_handler.opt.get_state(),
        "loss_history": factor_model.local_loss_history,
        "param_convergence_history": factor_model.local_var_param_convergence_history,
        "config": train_config                
    }
    
    torch.save(model_state_dict, model_out_filename + "_local.pth")

    pyro.get_param_store().save(model_out_filename + "_local_paramstore.pth")
    
    return factor_model, test_handler, model_state_dict
            
def evaluate_fitted_model(rep_config,
                          data_package,
                          train_handler,
                          test_handler,
                          global_train_state,
                          local_train_state,
                          N_POSTERIOR_SAMPLES,
                          metric_out_filename):
    """After training a model and performing local retraining, evaluate metrics.
    
        rep_config (dict): information on simulation scenario
        data_package (dict): raw data 
    
    """



    train_X_l_list_clean = data_package['train_X_l_list_clean']
    test_X_l_list_clean = data_package['test_X_l_list_clean']
    test_y_clean = data_package['test_y_clean']
    X_l_list_column_filters = data_package['X_l_list_column_filters']
    X_l_mean_list = data_package['X_l_mean_list']
    X_l_sd_list = data_package['X_l_sd_list']
    train_SIM_decomp = data_package['train_SIM_decomp']
    L = data_package['L']
    
    n = rep_config['n']
    p_l = rep_config['p_l']
    snr_x = rep_config['snr_x']
    snr_y = rep_config['snr_y']
    sparsity = rep_config['sparsity']
    k_delta = rep_config['k_delta']
    rep = rep_config['rep']
    
    run_minibatch = global_train_state['inference_time']
    test_run_minibatch = local_train_state['inference_time']
    
    

    #################
    # Sample from posterior
    print('Sampling from posterior')
    sites = [Sites.joint_structure_l.format(l = l) for l in range(L)] \
        + [Sites.view_structure_l.format(l = l) for l in range(L)]
    outcome_sites = [Sites.y_pred]
    post_samples = train_handler.predict(train_X_l_list_clean, N_POSTERIOR_SAMPLES, sites)
    post_samples_y = test_handler.predict(test_X_l_list_clean, N_POSTERIOR_SAMPLES, outcome_sites)

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
            k_delta = k_delta,
            runtime = run_minibatch,
            predict_runtime = test_run_minibatch
            )

    # Save evaluation

    # print("Exporting metrics:")
    # print(metric_out_filename)

    # eval_metric_table.to_csv(metric_out_filename + ".csv")
    return eval_metric_table


##############################################################################
# Single-view helpers (MatrixDecomp)
##############################################################################

def load_rds(path):
    """Load a single RDS file via rpy2 and return a dict of tensors."""
    import rpy2.robjects as robjects
    from rpy2.robjects import default_converter, conversion
    from data_utils import r_to_torch

    conversion._converter = None
    conversion.set_conversion(default_converter)
    r_obj = robjects.r["readRDS"](path)
    return r_to_torch(r_obj)


def process_data_single_view(sim_data, training_size, seed=123,
                              training_split=True, sim_data_test=None):
    """Single-view analogue of process_data.

    Parameters
    ----------
    sim_data : dict of tensors
        Output of load_rds; expected keys: X_l (list of 1), y, Z, Lambda_l, K.
    training_size : float
        Fraction of observations used for training (only used when training_split=True).
    seed : int
    training_split : bool
        If True, perform a random train/test split on sim_data.
        If False, use sim_data as training and sim_data_test as test.
    sim_data_test : dict of tensors or None
        Separate test dataset (required when training_split=False).

    Returns
    -------
    train_subset, test_subset : TensorDataset
    data_package : dict
    """
    X = sim_data["X_l"][0]
    y = sim_data["y"]
    k = int(sim_data["K"].item())
    n = X.shape[0]

    if training_split:
        train_idx, test_idx = train_test_split(
            torch.arange(n), train_size=training_size, random_state=seed, shuffle=True
        )
        X_train_raw = X[train_idx]
        X_test_raw  = X[test_idx]
        y_train_raw = y[train_idx]
        y_test_raw  = y[test_idx]
    else:
        assert sim_data_test is not None, "sim_data_test required when training_split=False"
        X_test_full = sim_data_test["X_l"][0]
        y_test_full = sim_data_test["y"]
        train_idx   = torch.arange(n)
        test_idx    = torch.arange(X_test_full.shape[0])
        X_train_raw = X
        X_test_raw  = X_test_full
        y_train_raw = y
        y_test_raw  = y_test_full

    col_filter   = zero_variance_col_filter(X_train_raw)
    X_train_filt = X_train_raw[:, col_filter]
    X_test_filt  = X_test_raw[:, col_filter]

    train_norm = normalize_tensor_by_col(X_train_filt)
    X_train    = train_norm["data_clean"]
    X_mean     = train_norm["means"]
    X_sd       = train_norm["sds"]

    test_norm = normalize_tensor_by_col(X_test_filt, X_mean, X_sd)
    X_test    = test_norm["data_clean"]

    y_mean  = torch.mean(y_train_raw)
    y_std   = torch.std(y_train_raw)
    y_train = (y_train_raw - y_mean) / y_std
    y_test  = (y_test_raw  - y_mean) / y_std

    train_subset = TensorDataset(torch.arange(len(train_idx)), X_train, y_train.squeeze())
    test_subset  = TensorDataset(torch.arange(X_test.shape[0]),  X_test,  y_test.squeeze())

    data_package = {
        "k":          k,
        "train_idx":  train_idx,
        "X_train":    X_train,
        "X_test":     X_test,
        "y_train":    y_train.squeeze(),
        "y_test":     y_test.squeeze(),
        "col_filter": col_filter,
        "X_mean":     X_mean,
        "X_sd":       X_sd,
        "sim_decomp": extract_sim_decomp_single_view(sim_data),
    }

    return train_subset, test_subset, data_package


def train_model_single_view(model_config, train_config, train_subset, verbose=False):
    """Single-view analogue of train_model. Trains a MatrixDecomp model globally.

    Parameters
    ----------
    model_config : dict
        Keys: k (int), dense (bool).
    train_config : dict
        Keys: initial_lr, betas, num_particles, lr_step_size, lr_decay_factor,
        minibatch_size, min_epochs, max_epochs.

    Returns
    -------
    factor_model, train_handler, runtime (seconds)
    """
    from matrix_decomp import MatrixDecomp

    factor_model = MatrixDecomp(
        model_config["k"], supervised=True, dense=model_config["dense"]
    )

    LOSS = Trace_ELBO(num_particles=train_config["num_particles"])
    scheduler = pyro.optim.StepLR({
        "optimizer": torch.optim.Adam,
        "optim_args": {"lr": train_config["initial_lr"], "betas": train_config["betas"]},
        "step_size": train_config["lr_step_size"],
        "gamma":     train_config["lr_decay_factor"],
    })

    train_handler = ModelHandler(
        "train", factor_model,
        loss=LOSS,
        opt_scheduler=scheduler,
        orthogonal_projection=False,
    )

    t0 = time.time()
    train_handler.do_inference(
        train_dataset=train_subset,
        min_epochs=train_config["min_epochs"],
        epochs=train_config["max_epochs"],
        minibatch_flag=True,
        minibatch_size=train_config["minibatch_size"],
        variational_diff_func=np.mean,
        verbose=verbose,
    )
    runtime = time.time() - t0

    return factor_model, train_handler, runtime


def train_model_locally_single_view(factor_model, train_config, test_subset, verbose=False):
    """Single-view analogue of train_model_locally. Infers local Z for test observations.

    Global variational parameters are frozen (read from factor_model.params).
    No files are written.

    Returns
    -------
    test_handler, runtime (seconds)
    """
    LOSS = Trace_ELBO(num_particles=train_config["num_particles"])
    scheduler = pyro.optim.StepLR({
        "optimizer": torch.optim.Adam,
        "optim_args": {"lr": train_config["initial_lr"], "betas": train_config["betas"]},
        "step_size": train_config["lr_step_size"],
        "gamma":     train_config["lr_decay_factor"],
    })

    test_handler = ModelHandler(
        "predict", factor_model,
        loss=LOSS,
        opt_scheduler=scheduler,
        orthogonal_projection=False,
    )

    t0 = time.time()
    test_handler.do_inference(
        train_dataset=test_subset,
        min_epochs=train_config["min_epochs"],
        epochs=train_config["max_epochs"],
        minibatch_flag=True,
        minibatch_size=train_config["minibatch_size"],
        variational_diff_func=np.mean,
        verbose=verbose,
    )
    runtime = time.time() - t0

    return test_handler, runtime


def evaluate_fitted_model_single_view(train_handler, test_handler, data_package,
                                       n_posterior_samples=300):
    """Single-view analogue of evaluate_fitted_model.

    Returns a one-row DataFrame with structure RSE, covariance Frobenius error,
    outcome MSE, and 95% credible interval coverage.
    """
    post_train = train_handler.predict(
        [data_package["X_train"]], n_posterior_samples,
        return_sites=["structure", Sites.Lambda_l.format(l=0)],
    )
    post_test = test_handler.predict(
        [data_package["X_test"]], n_posterior_samples,
        return_sites=["y_pred"],
    )

    eval_structures = calc_all_structures_with_rescaling_single_view(
        data_package["col_filter"],
        data_package["X_mean"],
        data_package["X_sd"],
        post_train,
        post_test,
        data_package["sim_decomp"],
        data_package["train_idx"],
        data_package["y_test"],
    )

    return eval_performance_single_view(**eval_structures)


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
                       n_posterior_samples=100, verbose=False):
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

    factor_model, train_handler, _ = train_model_single_view(
        model_config, train_config, train_ds, verbose=verbose
    )
    val_handler, _ = train_model_locally_single_view(
        factor_model, train_config, val_ds, verbose=verbose
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
                n_workers=1, verbose=False):
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
        print(
            f"[{i+1}/{len(train_grid)}] "
            f"lr={train_config['initial_lr']}  "
            f"step={train_config['lr_step_size']}  "
            f"decay={train_config['lr_decay_factor']}  "
            f"bs={train_config['minibatch_size']}",
            flush=True,
        )

        fold_args = [
            (X_raw, y_raw, tr_idx, va_idx, model_config, train_config,
             n_posterior_samples, verbose)
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