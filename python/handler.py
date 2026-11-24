
import math
import numpy as np

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO, SVGD

from torch.utils.data import DataLoader

from constants import Sites, Params


class AveragedParamChange:
    """Stopping statistic for the "avg_param_change" criterion.

    Every epoch the current values of each parameter group (default: all shared
    loadings loc_Lambda_l*, stacked, and loc_beta) are added to a running block
    sum. Every `block` epochs the block mean is stored and the sum reset; only the
    last 2 * window_blocks block means are kept. Once that many exist, for each
    group
        A = mean of the last window_blocks block means   (the last block * window_blocks epochs)
        B = mean of the window_blocks block means before them
        change = ||A - B||_F / ||A||_F
    i.e. how far the parameters' time average moved between two consecutive,
    disjoint windows. Averaging over every epoch cancels the optimiser's
    epoch-to-epoch jitter, so the statistic measures drift. The rule is met when
    every group's change is below tol (a float, or a dict group -> float).

    Memory: one running sum plus 2 * window_blocks block means per group, i.e.
    (2 * window_blocks + 1) float32 copies of the tracked parameters, on the CPU,
    independent of the number of epochs (13 copies of Lambda and beta by default).
    Statistics are checked only when a block completes (every `block` epochs).
    """

    def __init__(self, block=25, window_blocks=6, tol=0.05, groups=None):
        from collections import deque
        self.block, self.window_blocks = int(block), int(window_blocks)
        self.groups = groups or {"Lambda": "loc_Lambda_l", "beta": "loc_beta"}
        self.tol = tol if isinstance(tol, dict) else {g: float(tol) for g in self.groups}
        self.sums, self.count = {}, 0
        self.means = {g: deque(maxlen=2 * self.window_blocks) for g in self.groups}
        self.history = []          # (epoch, {group: change}) at every completed block once full

    def _values(self, store):
        out = {}
        for g, prefix in self.groups.items():
            names = sorted(n for n in store.keys() if n == prefix or n.startswith(prefix))
            out[g] = torch.cat([store[n].detach().reshape(-1).to("cpu", torch.float32) for n in names])
        return out

    def update(self, store, epoch):
        """Add this epoch's values; returns True when the rule is met at this epoch."""
        vals = self._values(store)
        for g, v in vals.items():
            if g in self.sums:
                self.sums[g] += v
            else:
                self.sums[g] = v.clone()
        self.count += 1
        if self.count < self.block:
            return False
        for g in self.groups:
            self.means[g].append(self.sums[g] / self.count)
        self.sums, self.count = {}, 0
        if len(self.means[next(iter(self.groups))]) < 2 * self.window_blocks:
            return False
        stats = {}
        for g, q in self.means.items():
            seq = list(q)
            B = torch.stack(seq[:self.window_blocks]).mean(0)
            A = torch.stack(seq[self.window_blocks:]).mean(0)
            stats[g] = float(torch.linalg.norm(A - B) / torch.linalg.norm(A).clamp_min(1e-12))
        self.history.append((epoch, stats))
        return all(stats[g] < self.tol[g] for g in self.groups)

    def write_window_mean(self, store):
        """Replace each tracked parameter in `store` by its mean over the last
        window_blocks completed blocks (the window A of the last check; fewer blocks
        if fewer have completed). Used for local inference, where the last iterate is
        noisy because every step resamples the frozen global parameters. Values are
        averaged on the constrained scale and written back through the param store,
        which keeps each parameter's constraint. Returns the number of epochs averaged."""
        q0 = self.means[next(iter(self.groups))]
        w = min(self.window_blocks, len(q0))
        if w == 0:
            return 0
        for g, prefix in self.groups.items():
            A = torch.stack(list(self.means[g])[-w:]).mean(0)
            names = sorted(n for n in store.keys() if n == prefix or n.startswith(prefix))
            i = 0
            for n in names:
                cur = store[n]
                k = cur.numel()
                store[n] = A[i:i + k].reshape(cur.shape).to(cur.device, cur.dtype)
                i += k
        return w * self.block

    def nbytes(self):
        """Bytes held by the tracker (block means plus running sums)."""
        held = sum(t.numel() * t.element_size() for q in self.means.values() for t in q)
        return held + sum(t.numel() * t.element_size() for t in self.sums.values())

class ModelHandler:
    """Runs stochastic variational inference for one of the factor models and
    draws posterior samples from the fit.

    A handler is either a training handler (mode="train"), which fits the
    model's global parameters and training-set scores with model.forward /
    model.guide, or a prediction handler (mode="predict"), which infers scores
    for new observations with model.predict_forward / model.predict_guide while
    the global parameters stay fixed. The model object carries the training
    history (loss_history, total_epochs, ... for training; local_loss_history,
    local_epochs, ... for prediction).

    Usually built by run_methods.train_model / train_model_locally rather than
    directly.
    """
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
        """
        Parameters
        ----------
        mode : "train" or "predict"
        model : MatrixDecomp, SupMultiviewShared or SupMultiviewDecomp
        guide : optional guide replacing the model's own (e.g. an AutoNormal)
        opt : Pyro optimiser; ignored when opt_scheduler is given
        loss : ELBO objective, e.g. Trace_ELBO(num_particles=2)
        orthogonal_projection : for models with view-specific factors
            (SupMultiviewDecomp), project the view-specific scores onto the
            orthogonal complement of the shared signal when training converges.
            Has no effect for the other models.
        inference_class : optional pre-built inference object used instead of SVI
        opt_scheduler : Pyro learning-rate scheduler (e.g. pyro.optim.StepLR),
            stepped once per epoch; replaces opt
        local : unused
        device : "cpu", "mps" or "cuda"
        """
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
        """Placeholder; not implemented."""
        return

    def _set_model_n(self, n):
        """Record the dataset size on the model (n for training, n_predict for
        prediction); the guide uses it to size the per-observation parameters."""
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
                    ma_window = 50,
                    ma_patience = 50,
                    avg_block = 25,
                    avg_window_blocks = 6,
                    avg_tol = 0.05,
                    avg_groups = None,
                    avg_return_window_mean = False,
                    device = "cpu",
                    verbose = False,
                    optuna_trial = None,
                    optuna_step_offset = 0):
        """Fit the model by SVI over minibatches until the stopping rule is met.

        Each epoch makes one pass over shuffled minibatches (the last partial
        batch is dropped). After each epoch the loss and the mean change in the
        variational parameters are appended to the model's history. Training
        stops when epoch > min_epochs and the loss has converged under
        convergence_criterion, or after `epochs` epochs.

        Parameters
        ----------
        train_dataset : TensorDataset(idx, X_1, ..., X_L, y); for a censored
            outcome, TensorDataset(idx, X_1, ..., X_L, event, y). In predict mode
            y is never passed to the model.
        inference_type : "svi" or "svgd"
        min_epochs, epochs : minimum and maximum number of epochs
        minibatch_flag : must be True (full-batch training is not implemented)
        minibatch_size : observations per minibatch
        variational_diff_func : summary of per-parameter change norms recorded
            each epoch (default: mean)
        convergence_criterion :
            "elbo_min"  — epochs since the lowest loss exceed sqrt(current epoch)
            "elbo_snr"  — |mean(last window) - mean(previous window)| / sd(last
                          window) < snr_threshold, and the current epoch's loss is
                          at most (1 + spike_tol) x the median of the last window
            "grad_norm" — mean minibatch gradient norm < grad_norm_tol
            "ma_patience" — Tran, Nguyen & Dao (2021, arXiv:2103.01327, Alg. 4-7):
                          the mean loss over the last ma_window epochs has not set
                          a new minimum for ma_patience epochs
        window, snr_threshold, spike_tol : elbo_snr settings
        grad_norm_tol : grad_norm setting
        ma_window, ma_patience : ma_patience settings
            "avg_param_change" — time-averaged parameter change (training only):
                          every epoch the shared loadings (loc_Lambda_l*) and
                          loc_beta are averaged into blocks of avg_block epochs;
                          the mean over the last avg_window_blocks blocks is
                          compared with the mean over the avg_window_blocks
                          before, and training stops when the relative change
                          ||A - B|| / ||A|| of both is below avg_tol. See
                          AveragedParamChange. The tracker persists on the handler
                          across do_inference calls (chunked training); its
                          statistics are in self.avg_tracker.history.
        avg_block, avg_window_blocks, avg_tol : avg_param_change settings; avg_tol is a
            float or a dict {"Lambda": tol, "beta": tol}
            (defaults 25, 6, 0.05: two 150-epoch windows, 5%)
            In prediction mode (local inference) avg_param_change tracks the local
            parameters instead: by default the test score means loc_Z_pred and sds
            scale_Z_pred (groups "Z" and "Z_scale"), with a fresh tracker each call.
        avg_groups : dict group -> parameter-name prefix tracked by avg_param_change
            (None: the defaults above for the handler's mode)
        avg_return_window_mean : avg_param_change only: when inference ends (rule met or
            max epochs), replace the tracked parameters by their mean over the last
            window (AveragedParamChange.write_window_mean) instead of the last iterate
        verbose : print progress every 10 epochs and on termination
        optuna_trial, optuna_step_offset : report the per-datum loss to an Optuna
            trial each epoch so unpromising trials can be pruned
        max_iter, tol, variational_tol, device : unused

        Returns
        -------
        The Pyro inference object (e.g. SVI). The fitted parameters are in the
        Pyro param store, which is also attached to the model as model.params.
        """

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
            """Projection matrix onto the column space of tens."""
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
            # ma_patience state: best moving-average loss so far and epochs since it
            _ma_best, _ma_count = math.inf, 0
            mean_grad_norm = math.inf
            if convergence_criterion == "avg_param_change":
                if self.mode == "train":
                    if getattr(self, "avg_tracker", None) is None:
                        self.avg_tracker = AveragedParamChange(avg_block, avg_window_blocks, avg_tol,
                                                               avg_groups)
                else:
                    self.avg_tracker = AveragedParamChange(
                        avg_block, avg_window_blocks, avg_tol,
                        avg_groups or {"Z": Params.loc_Z_pred, "Z_scale": Params.scale_Z_pred})
            _avg_met = False

            for epoch in range(epochs):
                epoch_loss = 0.
                batch_grad_norms = []
                for batch in loader:
                    # batch is subsampled [idx, X_l_list, y]
                    batch_idx = batch.pop(0).to(self.device)
                    if self.model.model_type in ("MatrixDecomp", "SupMultiviewDecomp", "SupMultiviewShared"):
                        # Expected batch order: [idx, X_1, ..., X_L, y]
                        # For censored outcome:  [idx, X_1, ..., X_L, event, y]  (event: 1 observed, 0 censored)
                        y_batch = batch.pop(-1).squeeze().to(self.device)
                        if getattr(self.model, 'outcome', 'gaussian') in ("censored", "lognormal_aft"):
                            cens_batch = batch.pop(-1).squeeze().to(self.device)
                        batch = [t.to(self.device) for t in batch]
                        # In predict mode only X informs Z; y is not observed.
                        if self.mode == "predict":
                            loss = inference.step(batch, batch_idx)
                        elif getattr(self.model, 'outcome', 'gaussian') in ("censored", "lognormal_aft"):
                            loss = inference.step(batch, batch_idx, y_batch, cens_batch)
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

                if convergence_criterion == "avg_param_change":
                    _avg_met = self.avg_tracker.update(
                        param_store_curr,
                        self.model.total_epochs if self.mode == "train" else self.model.local_epochs)

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

                elif convergence_criterion == "ma_patience":
                    if len(_local_loss_history) >= ma_window:
                        ma = float(np.mean(_local_loss_history[-ma_window:]))
                        if ma < _ma_best:
                            _ma_best, _ma_count = ma, 0
                        else:
                            _ma_count += 1
                    loss_converged = _ma_count >= ma_patience

                elif convergence_criterion == "avg_param_change":
                    loss_converged = _avg_met

                else:
                    raise ValueError(f"Unknown convergence_criterion: {convergence_criterion!r}. "
                                     "Choose 'elbo_min', 'elbo_snr', 'grad_norm', 'ma_patience' "
                                     "or 'avg_param_change'.")

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

            self.window_mean_epochs = 0
            if convergence_criterion == "avg_param_change" and avg_return_window_mean:
                self.window_mean_epochs = self.avg_tracker.write_window_mean(pyro.get_param_store())
        else:
            raise NotImplementedError    
        
        self.model.params = pyro.get_param_store()        
            
        return inference
    
        
    def predict(self,
                X_list,
                num_samples,
                return_sites):
        """Draw posterior samples of the requested sites.

        Samples the guide num_samples times and runs the model forward for
        each draw. The outcome is not observed, so outcome sites (y, y_pred)
        are posterior predictive draws.

        Parameters
        ----------
        X_list : list of (n, p_l) tensors — the observations this handler was
            fitted on (training set for "train", new observations for "predict")
        num_samples : number of posterior draws
        return_sites : site names to return, e.g. [Sites.Z, Sites.y_pred]

        Returns
        -------
        dict mapping each site name to a tensor with leading dimension num_samples
        """
        X_list_dev =[x.to(self.device) for x in X_list]
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