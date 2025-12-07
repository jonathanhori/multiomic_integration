
import math
import numpy as np

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO

from torch.utils.data import DataLoader

class ModelHandler:
    def __init__(self,
                 mode,
                 model,
                #  guide,
                 opt = Adam({"lr": 0.001}),
                 loss = Trace_ELBO(),
                 local = False
                 ):
        assert mode in ("train", "test"), \
            "Argument 'mode' must be set to 'train' or 'test'. If any new observations are present, set 'test'."
            
        self.model = model
        
        assert self.model.params is not None, "params are None"
        print(self.model.params.keys())
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
        elif self.mode == "test":
            self.forward = self.model.predict_forward
            self.guide = self.model.predict_guide
        # self.guide = guide
        self.loss = loss
        self.opt = opt 
        
    def get_param_values(self):
        return
    
    def _set_model_n(self, n):
        self.model.n = n
    
    def do_inference(self,
                     train_dataset,
                    # model, 
                    # guide, 
                    # opt = Adam({"lr": 0.001}), #"Adam",
                    # elbo = Trace_ELBO(),
                    inference = "svi",
                    min_epochs = 10,
                    epochs = 20,
                    max_iter = 20000,
                    minibatch_flag = True,
                    minibatch_size = 32,
                    tol = 1e-4,
                    variational_tol = 1e-4,
                    variational_diff_func = np.mean,
                    device = "cpu",
                    verbose = False):
        
        # min_epochs = 10
        
        # if minibatch_flag == False, then do not minibatch, data loader not necessary
        if not minibatch_flag:
            minibatch_size = self.model.n
        self._set_model_n(train_dataset.__len__())
            
        ########################
        # Handle minibatching of dataset
        ########################
        loader = DataLoader(train_dataset, batch_size = minibatch_size, shuffle = True, drop_last=True)
        
        ########################
        # Initialize instances for optimization
        ########################
        
        svi = SVI(self.forward, self.guide, self.opt, loss = self.loss)
        
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
            for epoch in range(epochs):
                epoch_loss = 0.
                for batch in loader:
                    # batch is subsampled [idx, X_l_list, y]
                    batch_idx = batch.pop(0)
                    y_batch = batch.pop(-1).squeeze()
                    # print(batch.shape)
                    loss = svi.step(batch, batch_idx, y_batch)
                    # print(loss)
                    epoch_loss += loss
                
                self.model.loss_history.append(epoch_loss)
                
                # Determine the epoch when the min loss is obtained
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    epoch_at_min_loss = epoch
                
                self.model.total_epochs += 1
                
                # Check Euclidean norm of difference in variational params for convergence
                param_store_curr = pyro.get_param_store()
                if self.mode == "train":
                    params_epoch_curr = {k: v.detach().clone() for k, v in param_store_curr.items()}
                    if epoch == 0: 
                        params_epoch_last = params_epoch_curr
                    param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item() 
                                            for k in params_epoch_curr}
                    self.model.var_param_convergence_history.append(
                        variational_diff_func(list(param_diff_norm_dict.values()))
                        )
                    params_converged = all(n < variational_tol for n in param_diff_norm_dict.values())                    
                elif self.mode == "test":
                    # only consider convergence in local params
                    params_epoch_curr = {k: v.detach().clone() 
                                         for k, v in param_store_curr.items()
                                         if any(param in k for param in ('loc_Z', 'scale_Z', 'loc_Phi_l', 'scale_Phi_l'))}
                    if epoch == 0: 
                        params_epoch_last = params_epoch_curr
                    param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item() 
                                            for k in params_epoch_curr}
                    self.model.local_var_param_convergence_history.append(variational_diff_func(param_diff_norm_dict.values()))
                    params_converged = all(n < variational_tol for n in param_diff_norm_dict.values())   
                    
                    
                if verbose:
                        print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / self.model.n:.4f}")
                        print(f"Loss at epoch {epoch+1}: {loss / self.model.n}")
                        print(f"Variational parameter difference: {self.model.var_param_convergence_history[-1]}")
                    
                    # Converged?
                if epoch > min_epochs and epoch - epoch_at_min_loss > math.sqrt(epoch)\
                    and params_epoch_last is not None and params_converged:
                # if prev_loss is not None and abs(epoch_loss - prev_loss) / model.n < tol:
                    # model.total_epochs = epoch
                    break
                params_epoch_last = params_epoch_curr
                
                # prev_loss = epoch_loss
        else:
            raise NotImplementedError    
        
        self.model.params = pyro.get_param_store()        
            
        return svi
    
    # def do_local_inference(self,
    #                        test_dataset,
    #                        model, 
    #                        guide):
    #     pass
        
    def predict(self,
                X_list,
                num_samples,
                return_sites):
        n = X_list[0].shape[0]
        predictive = Predictive(self.model, 
                                guide = self.guide, 
                                num_samples = num_samples,
                                return_sites = return_sites)
        return predictive(
            X_list,
            torch.arange(n)
            )