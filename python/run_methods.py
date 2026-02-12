import numpy as np

def load_data(file_name_base):
    """Read data
    
    TODO: read the data in addition to pulling out relevant reps. currently input 
    is file name base, but should be the directory plus data config
    """
    
    print(file_name_base.format(rep))
    sim_data = files_for_condition.get(file_name_base.format(rep))
    sim_data_test = test_files_for_condition.get(file_name_base.format(rep))
    
    L = int(sim_data.get("L"))
    
    X_l_list = sim_data.get("X_l")
    y = sim_data.get("y")
    
    X_l_list_test = sim_data_test.get("X_l")
    y_test = sim_data_test.get("y")
    
    k = int(sim_data.get("K") + k_delta)
    k_l_list = [int(k_l + k_delta) for k_l in sim_data.get("K_l").int().tolist()]
    
    n = X_l_list[0].shape[0]
    n_test = X_l_list_test[0].shape[0]
    
    p_l = int(sim_data.get("p_l")[0])
    
    return {
        'L': L,
        'X_l_list': X_l_list,
        'y': y,
        'X_l_list_test': X_l_list_test,
        'y_test': y_test,
        'k': k,
        'k_l_list': k_l_list,
        'n': n,
        'n_test': n_test,
        'p_l': p_l
    }
    
def process_data(data_inputs,
                training_split,
                training_size,
                seed = 123):
    """After loading input data, perform train/test split if necessary, and 
    create data loaders"""

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
    return train_subset, test_subset


def train_model(model_config,
                train_config,
                train_subset,
                out_dir = "./model_output/"):
    """Train the supervised multiview decomposition model"""
    
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
        'gamma': lr_decay_factor,
        'verbose': True
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
    epochs = MIN_EPOCHS

    try:            
        t0 = time.time()
        svi = train_handler.do_inference(
            train_dataset = train_subset,
            min_epochs = min_epochs,
            epochs = max_epochs,
            minibatch_flag = True,
            minibatch_size = mini_size,
            variational_diff_func = np.mean,
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
            "optimizer_state": train_handler.opt.get_state(),
            "loss_history": factor_model.loss_history,
            "param_convergence_history": factor_model.var_param_convergence_history                
        }, model_out_filename + ".pth")

        pyro.get_param_store().save(model_out_filename + "_paramstore.pth") #os.path.join(model_out_path, model_out_filename + "_paramstore.pth"))
    except Exception as e:
        print(e)
    
    return
            
def evaluate_fitted_model(mod):
    """After training a model and performing local retraining, evaluate metrics."""

    #################
    # Sample from posterior
    print('Sampling from posterior')
    sites = [Sites.joint_structure_l.format(l = l) for l in range(L)] \
        + [Sites.view_structure_l.format(l = l) for l in range(L)]
    outcome_sites = [Sites.y_pred]
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
    return