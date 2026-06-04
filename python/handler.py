
import math
import numpy as np

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO, SVGD

from torch.utils.data import DataLoader

from constants import Sites, Params

class ModelHandler:
    def __init__(self,
                 mode,
                 model,
                 guide = None,
                 opt = Adam({"lr": 0.001}), # Opt is not needed if opt_scheduler is provided
                 loss = Trace_ELBO(),
                 orthogonal_projection = True,
                 inference_class = None,
                 opt_scheduler = None,
                 local = False,
                 device = "cpu"
                 ):
        assert mode in ("train", "predict"), \
            "Argument 'mode' must be set to 'train' or 'predict'. If any new observations are present, set 'predict'."
            
        self.model = model
        
        self.mode = mode
        # if local:
        #     self.forward = self.model.predict_forward
        # else:
        #     self.forward = self.model.forward
        if self.mode == "train":
            self.forward = self.model.forward
            self.guide = self.model.guide
        # elif mode == "train_local":
        #     self.forward = self.model.
        elif self.mode == "predict":
            self.forward = self.model.predict_forward
            self.guide = self.model.predict_guide
        
        # Overwrite guide if provided
        if guide is not None:
            self.guide = guide
        # self.guide = guide
        self.loss = loss
        
        self.opt_scheduler = opt_scheduler
        self.inference_class = inference_class
        if opt_scheduler is not None:
            self.opt = opt_scheduler 
        else:
            self.opt = opt
        
        self.orthogonal_projection = orthogonal_projection
        self.device = torch.device(device) if isinstance(device, str) else device
        self.learning_rate_vec = []
        
    def get_param_values(self):
        return
    
    def _set_model_n(self, n):
        if self.mode == "train": self.model.n = n
        elif self.mode == "predict": self.model.n_predict = n
    
    def do_inference(self,
                     train_dataset,
                    # model,
                    # guide,
                    # opt = Adam({"lr": 0.001}), #"Adam",
                    # elbo = Trace_ELBO(),
                    inference_type = "svi",
                    min_epochs = 10,
                    epochs = 20,
                    max_iter = 20000,
                    minibatch_flag = True,
                    minibatch_size = 32,
                    tol = 1e-4,
                    variational_tol = 0.1,
                    variational_diff_func = np.mean,
                    convergence_criterion = "elbo_min",
                    window = 50,
                    snr_threshold = 0.1,
                    spike_tol = 0.1,
                    grad_norm_tol = 1e-3,
                    device = "cpu",
                    verbose = False,
                    optuna_trial = None,
                    optuna_step_offset = 0):
        
        # min_epochs = 10
        
        # if minibatch_flag == False, then do not minibatch, data loader not necessary
        if not minibatch_flag:
            minibatch_size = self.model.n
        self._set_model_n(train_dataset.__len__())
            
        ########################
        # Handle minibatching of dataset
        ########################
        loader = DataLoader(train_dataset, batch_size = minibatch_size, shuffle = True, drop_last=True)
        if len(loader) == 0:
            raise ValueError(
                f"DataLoader produced 0 batches: minibatch_size={minibatch_size} "
                f"exceeds dataset size={len(train_dataset)} (drop_last=True). "
                f"Reduce minibatch_size to <= {len(train_dataset)}."
            )
        
        ########################
        # Initialize instances for optimization
        ########################
        
        # svi = self.inference_class()
        if self.inference_class is None and inference_type == "svi":
            inference = SVI(self.forward, self.guide, self.opt, loss = self.loss)
        elif self.inference_class is None and inference_type == "svgd":
            # SVGD kernel is stored as the guide
            inference = SVGD(self.forward, self.guide, self.opt, 
                            num_particles = 50, max_plate_nesting = 1)
        elif self.inference_class is not None:
            inference = self.inference_class
            
        ###
        # Projection matrix
        def projection_by_qr(tens):
            Q, _ = torch.linalg.qr(tens)
            return Q @ Q.T
        # def proj_onto_perp(P, tens):
        #     return torch.sub(P, torch.mm(P, tens))
        
        ########################
        # Train
        # If minibatching: pass batch from loader
        # If not minibatching: pass original data as tensor
        ########################
        if minibatch_flag:
            # prev_loss = None
            min_loss = math.inf
            epoch_at_min_loss = 0

            params_epoch_last = None
            params_epoch_curr = None

            # Running loss history for elbo_snr criterion (local copy, avoids mode branching)
            _local_loss_history = []
            mean_grad_norm = math.inf

            for epoch in range(epochs):
                epoch_loss = 0.
                batch_grad_norms = []
                for batch in loader:
                    # batch is subsampled [idx, X_l_list, y]
                    batch_idx = batch.pop(0).to(self.device)
                    if self.model.model_type == "MatrixDecomp":
                        y_batch = batch.pop(-1).squeeze().to(self.device)
                        batch = [t.to(self.device) for t in batch]
                        loss = inference.step(batch, batch_idx, y_batch)
                    elif self.model.model_type in ("SupMultiviewDecomp", "SupMultiviewShared"):
                        y_batch = batch.pop(-1).squeeze().to(self.device)
                        batch = [t.to(self.device) for t in batch]
                        # In predict mode y is not available at inference time; only X informs Z.
                        if self.mode == "predict":
                            loss = inference.step(batch, batch_idx)
                        else:
                            loss = inference.step(batch, batch_idx, y_batch)
                    if self.device.type != "cpu":
                        for _p in pyro.get_param_store()._params.values():
                            if torch.isnan(_p).any():
                                _p.data.nan_to_num_(nan=0.0)
                                for _opt in self.opt.optim_objs.values():
                                    if _p in _opt.state:
                                        _opt.state[_p] = {}
                    epoch_loss += loss

                    # Collect gradient norms after each SVI step (grads still populated).
                    # Must use store._params (unconstrained leaf tensors) — store[n]
                    # returns the constrained view, a derived non-leaf with grad=None.
                    if convergence_criterion == "grad_norm":
                        store_now = pyro.get_param_store()
                        norms = [p.grad.norm().item()
                                 for p in store_now._params.values()
                                 if p.grad is not None]
                        if norms:
                            batch_grad_norms.append(np.mean(norms))

                if self.opt_scheduler is not None:
                    self.opt_scheduler.step()
                if self.model.penalty_obj is not None:
                    self.model.penalty_obj.update()

                # Update running list of loss and variational param histories
                param_store_curr = pyro.get_param_store()
                if self.mode == "train":
                    self.model.total_epochs += 1
                    self.model.loss_history.append(epoch_loss)

                    ####
                    params_epoch_curr = {k: v.detach().clone() for k, v in param_store_curr.items()}
                    if epoch == 0:
                        params_epoch_last = params_epoch_curr
                    param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item()
                                            for k in params_epoch_curr}
                    variational_diff = variational_diff_func(list(param_diff_norm_dict.values()))
                    self.model.var_param_convergence_history.append(variational_diff)

                elif self.mode == "predict":
                    self.model.local_epochs += 1
                    self.model.local_loss_history.append(epoch_loss)

                    ####
                    # only consider convergence in local params
                    params_epoch_curr = {k: v.detach().clone()
                                         for k, v in param_store_curr.items()
                                         if any(param in k for param in ('loc_Z', 'scale_Z', 'loc_Phi_l', 'scale_Phi_l'))}
                    if epoch == 0:
                        params_epoch_last = params_epoch_curr
                    param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item()
                                            for k in params_epoch_curr}
                    variational_diff = variational_diff_func(list(param_diff_norm_dict.values()))
                    self.model.local_var_param_convergence_history.append(variational_diff)

                # Determine the epoch when the min loss is obtained
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    epoch_at_min_loss = epoch

                _local_loss_history.append(epoch_loss)

                if optuna_trial is not None:
                    import optuna as _optuna
                    optuna_trial.report(epoch_loss / self.model.n,
                                        optuna_step_offset + epoch)
                    if optuna_trial.should_prune():
                        raise _optuna.TrialPruned()

                if convergence_criterion == "grad_norm" and batch_grad_norms:
                    mean_grad_norm = np.mean(batch_grad_norms)

                ########
                # Convergence logic
                past_min_epochs = epoch > min_epochs
                params_converged = True

                if convergence_criterion == "elbo_min":
                    loss_converged = epoch - epoch_at_min_loss > math.sqrt(epoch)

                elif convergence_criterion == "elbo_snr":
                    if len(_local_loss_history) < 2 * window:
                        loss_converged = False
                    else:
                        recent = _local_loss_history[-window:]
                        prev   = _local_loss_history[-2 * window:-window]
                        std_recent = np.std(recent)
                        if std_recent == 0:
                            loss_converged = True
                        else:
                            snr = abs(np.mean(recent) - np.mean(prev)) / std_recent
                            # Reject convergence if the current epoch is a spike: a single
                            # outlier loss can both shrink the SNR (by inflating std) and
                            # mislead us about whether we're actually at an optimum.
                            median_recent = np.median(recent)
                            no_spike = epoch_loss <= median_recent * (1.0 + spike_tol)
                            loss_converged = (snr < snr_threshold) and no_spike

                elif convergence_criterion == "grad_norm":
                    loss_converged = mean_grad_norm < grad_norm_tol

                else:
                    raise ValueError(f"Unknown convergence_criterion: {convergence_criterion!r}. "
                                     "Choose 'elbo_min', 'elbo_snr', or 'grad_norm'.")

                if verbose and epoch % 10 == 0:
                    print("--------------------")
                    print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / self.model.n:.4f}")
                    print(f"Loss at epoch {epoch+1}: {loss / self.model.n}")
                    print(f"Variational parameter difference: {variational_diff}")
                    if convergence_criterion == "elbo_min":
                        print(f"Number of epochs since minimum loss: {epoch - epoch_at_min_loss}")
                    elif convergence_criterion == "elbo_snr" and len(_local_loss_history) >= 2 * window:
                        recent = _local_loss_history[-window:]
                        prev   = _local_loss_history[-2 * window:-window]
                        std_r  = np.std(recent)
                        snr_val = abs(np.mean(recent) - np.mean(prev)) / std_r if std_r > 0 else 0.0
                        print(f"ELBO SNR: {snr_val:.4f} (threshold {snr_threshold})")
                        med_r = np.median(recent)
                        if snr_val < snr_threshold and epoch_loss > med_r * (1.0 + spike_tol):
                            print(f"Spike guard: epoch loss {epoch_loss:.2f} exceeds "
                                  f"{(1.0 + spike_tol):.2f}x recent median ({med_r:.2f}) — "
                                  f"convergence held off.")
                    elif convergence_criterion == "grad_norm":
                        print(f"Mean gradient norm: {mean_grad_norm:.6f} (threshold {grad_norm_tol})")
                    print(f"Loss converged? {str(loss_converged)}")
                    print(f"Params converged? {str(params_converged)}")
                    if self.model.penalty_obj is not None:
                        print(f"Ortho penalty: {self.model.penalty_obj.weight}")
                    else:
                        print(f"Ortho penalty: {self.model.ortho_penalty}")

                # Converged?
                all_converged = past_min_epochs and loss_converged and params_converged \
                    and params_epoch_last is not None
                if all_converged :
                    if verbose:
                        print("---TERMINATED-------------")
                        print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / self.model.n:.4f}")
                        print(f"Loss at epoch {epoch+1}: {loss / self.model.n}")
                        print(f"Variational parameter difference: {variational_diff}")
                        if convergence_criterion == "elbo_min":
                            print(f"Number of epochs since minimum loss: {epoch - epoch_at_min_loss}")
                        elif convergence_criterion == "elbo_snr" and len(_local_loss_history) >= 2 * window:
                            recent = _local_loss_history[-window:]
                            prev   = _local_loss_history[-2 * window:-window]
                            std_r  = np.std(recent)
                            snr_val = abs(np.mean(recent) - np.mean(prev)) / std_r if std_r > 0 else 0.0
                            print(f"ELBO SNR: {snr_val:.4f} (threshold {snr_threshold})")
                        elif convergence_criterion == "grad_norm":
                            print(f"Mean gradient norm: {mean_grad_norm:.6f} (threshold {grad_norm_tol})")
                        print(f"Loss converged? {str(loss_converged)}")
                        print(f"Params converged? {str(params_converged)}")
                        if self.model.penalty_obj is not None:
                            print(f"Ortho penalty: {self.model.penalty_obj.weight}")
                        else:
                            print(f"Ortho penalty: {self.model.ortho_penalty}")
                    
                
                    ########
                    # Project variational parameters for orthogonality constraint
                    # Do not project at last iteration
                    # Means = P @ M
                    # Vars = P @ diag(S) @ P.T
                    if self.orthogonal_projection and hasattr(self.model, 'k_l_list'):
                        if isinstance(self.guide, pyro.infer.autoguide.guides.AutoNormal):
                            loc_Z_name = 'AutoNormal.locs.Z'
                            loc_Lambda_l_name = 'AutoNormal.locs.Lambda_l{l}'
                            loc_Phi_l_name = 'AutoNormal.locs.Phi_l{l}'
                            scale_Phi_l_name = 'AutoNormal.scales.Phi_l{l}'
                        else:
                            loc_Z_name = Params.loc_Z
                            loc_Lambda_l_name = Params.loc_Lambda_l
                            loc_Phi_l_name = Params.loc_Phi_l
                            scale_Phi_l_name = Params.scale_Phi_l
                            
                        store = pyro.get_param_store()
                        loc_Z = store[loc_Z_name]
                        locs_Lambda_l_list = [store[loc_Lambda_l_name.format(l = l)] \
                            for l in range(len(self.model.k_l_list))]
                        locs_Phi_l_list = [store[loc_Phi_l_name.format(l = l)] \
                            for l in range(len(self.model.k_l_list))]
                        scales_Phi_l_list = [store[scale_Phi_l_name.format(l = l)] \
                            for l in range(len(self.model.k_l_list))]
                        
                        A = loc_Z @ torch.cat([Lambda.T for Lambda in locs_Lambda_l_list],
                                            dim = 1)
                        
                        P_Z = projection_by_qr(A) # TODO make below faster
                        P_Z_perp = torch.sub(torch.eye(P_Z.shape[0], device=self.device), P_Z)
                        # Means
                        proj_locs_Phi_l_list = [torch.mm(P_Z_perp, Phi_l) \
                            for Phi_l in locs_Phi_l_list]
                        # Variances
                        proj_scales_Phi_l_list = []
                        for vars in scales_Phi_l_list:
                            proj_scales = torch.stack(
                                [torch.diag(P_Z_perp * vars[:, k] @ P_Z_perp.T) \
                                    for k in range(vars.shape[1])],
                                dim = 1
                            )
                            proj_scales_Phi_l_list.append(proj_scales)
                        # proj_scales_Phi_l_list = [torch.mm(torch.square(P_Z_perp), vars) \
                        #     for vars in scales_Phi_l_list]
                        
                        # Overwrite param store
                        with torch.no_grad():
                            for l in range(len(self.model.k_l_list)):
                                store[loc_Phi_l_name.format(l = l)].copy_(proj_locs_Phi_l_list[l])
                                store[scale_Phi_l_name.format(l = l)].copy_(proj_scales_Phi_l_list[l])
                            
                    break
                params_epoch_last = params_epoch_curr
        else:
            raise NotImplementedError    
        
        self.model.params = pyro.get_param_store()        
            
        return inference
    
        
    def predict(self,
                X_list,
                num_samples,
                return_sites):
        X_list_dev = [x.to(self.device) for x in X_list]
        if self.model.model_type == "MatrixDecomp":
            n = X_list_dev[0].shape[0]
            predictive = Predictive(self.forward,
                                    guide = self.guide,
                                    num_samples = num_samples,
                                    return_sites = return_sites)
            return predictive(
                X_list_dev,
                torch.arange(n, device=self.device)
                )
        elif self.model.model_type in ("SupMultiviewDecomp", "SupMultiviewShared"):
            n = X_list_dev[0].shape[0]
            predictive = Predictive(self.forward,
                                    guide = self.guide,
                                    num_samples = num_samples,
                                    return_sites = return_sites)
            return predictive(
                X_list_dev,
                torch.arange(n, device=self.device)
                )