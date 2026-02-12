
import math
import numpy as np

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO

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
                 inference_class = SVI,
                 opt_scheduler = None,
                 local = False
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
        # self.inference_class = inference_class
        if opt_scheduler is not None:
            self.opt = opt_scheduler 
        else:
            self.opt = opt
        
        self.orthogonal_projection = orthogonal_projection
        # self.opt_scheduler = opt_scheduler
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
                    # inference = "svi",
                    min_epochs = 10,
                    epochs = 20,
                    max_iter = 20000,
                    minibatch_flag = True,
                    minibatch_size = 32,
                    tol = 1e-4,
                    variational_tol = 0.1,
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
        
        # svi = self.inference_class()
        svi = SVI(self.forward, self.guide, self.opt, loss = self.loss)
        
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
            
            min_variational_diff = math.inf
            epoch_at_min_diff = 0
            
            params_epoch_last = None
            params_epoch_curr = None
            for epoch in range(epochs):
                epoch_loss = 0.
                for batch in loader:
                    # batch is subsampled [idx, X_l_list, y]
                    batch_idx = batch.pop(0)
                    # print(batch)
                    # print(batch.shape)
                    # print(batch_idx)
                    # print(batch_idx.shape)
                    # print(batch.shape)
                    if self.model.model_type == "MatrixDecomp":
                        loss = svi.step(batch, batch_idx)
                    elif self.model.model_type == "SupMultiviewDecomp":
                        y_batch = batch.pop(-1).squeeze()
                        loss = svi.step(batch, batch_idx, y_batch)
                    # print(loss)
                    epoch_loss += loss
                
                ########
                # Project variational parameters for orthogonality constraint
                # Means = P @ M
                # Vars = P^.2 @ M
                if self.orthogonal_projection:
                    if isinstance(self.guide, pyro.infer.autoguide.guides.AutoNormal):
                        loc_Z_name = 'AutoNormal.locs.Z'
                        loc_Phi_l_name = 'AutoNormal.locs.Phi_l{l}'
                        scale_Phi_l_name = 'AutoNormal.scales.Phi_l{l}'
                    else:
                        loc_Z_name = Params.loc_Z
                        loc_Phi_l_name = Params.loc_Phi_l
                        scale_Phi_l_name = Params.scale_Phi_l
                        
                    store = pyro.get_param_store()
                    loc_Z = store[loc_Z_name]
                    locs_Phi_l_list = [store[loc_Phi_l_name.format(l = l)] \
                        for l in range(len(self.model.k_l_list))]
                    scales_Phi_l_list = [store[scale_Phi_l_name.format(l = l)] \
                        for l in range(len(self.model.k_l_list))]
                    
                    P_Z = projection_by_qr(loc_Z)
                    P_Z_perp = torch.sub(torch.eye(P_Z.shape[0]), P_Z)
                    proj_locs_Phi_l_list = [torch.mm(P_Z_perp, Phi_l) \
                        for Phi_l in locs_Phi_l_list]
                    proj_scales_Phi_l_list = [torch.mm(torch.square(P_Z_perp), vars) \
                        for vars in scales_Phi_l_list]
                    with torch.no_grad():
                        for l in range(len(self.model.k_l_list)):
                            store[loc_Phi_l_name.format(l = l)].copy_(proj_locs_Phi_l_list[l])
                            store[scale_Phi_l_name.format(l = l)].copy_(proj_scales_Phi_l_list[l])
                
                
                
                if self.opt_scheduler is not None:
                    self.opt_scheduler.step()
                if self.model.penalty_obj is not None:
                    self.model.penalty_obj.update()
                    # self.opt.param_groups
                    # self.learning_rate_vec.append()
                
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
                    # print(param_diff_norm_dict.values())
                    variational_diff = variational_diff_func(list(param_diff_norm_dict.values()))
                    self.model.var_param_convergence_history.append(variational_diff)
                    
                    # params_converged = variational_diff < variational_tol
                    # params_converged = all(n < variational_tol for n in param_diff_norm_dict.values())      
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
                    # variational_diff = variational_diff_func(param_diff_norm_dict.values())
                    # params_converged = variational_diff < variational_tol
                    # param_converge_metric < variational_tol
                    # params_converged = variational_diff_func(n < variational_tol for n in param_diff_norm_dict.values())   
                    
                    
                    # if verbose:
                    #         print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / self.model.n:.4f}")
                    #         print(f"Loss at epoch {epoch+1}: {loss / self.model.n}")
                    #         print(f"Variational parameter difference: {self.model.local_var_param_convergence_history[-1]}")
                    #         print(f"Number of epochs since minimum loss: {epoch - epoch_at_min_loss}")
                    #         # print(f"Variational param Param convergence metric: ")
                
                # Determine the epoch when the min loss is obtained
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    epoch_at_min_loss = epoch
                    
                # Determine the epoch when the minimum difference in variational params is obtained
                if variational_diff < min_variational_diff and variational_diff != 0:
                    min_variational_diff = variational_diff
                    epoch_at_min_diff = epoch
                
                ########
                # Convergence logic
                past_min_epochs = epoch > min_epochs
                loss_converged = epoch - epoch_at_min_loss > math.sqrt(epoch)
                params_converged = True #epoch - epoch_at_min_diff > math.sqrt(epoch)
                # params_converged = variational_diff < variational_tol
                # params_converged = all(n < variational_tol for n in param_diff_norm_dict.values()) 
                
                if verbose and epoch % 10 == 0:
                        print("--------------------")
                        print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / self.model.n:.4f}")
                        print(f"Loss at epoch {epoch+1}: {loss / self.model.n}")
                        print(f"Variational parameter difference: {variational_diff}")
                        print(f"Number of epochs since minimum loss: {epoch - epoch_at_min_loss}")
                        print(f"Loss converged? {str(loss_converged)}")
                        print(f"Params converged? {str(params_converged)}")
                        if self.model.penalty_obj is not None:
                            print(f"Ortho penalty: {self.model.penalty_obj.weight}")
                        else:
                            print(f"Ortho penalty: {self.model.ortho_penalty}")
                
                # Converged?
                if past_min_epochs and loss_converged and params_converged \
                    and params_epoch_last is not None :
                    break
                params_epoch_last = params_epoch_curr
        else:
            raise NotImplementedError    
        
        self.model.params = pyro.get_param_store()        
            
        return svi
    
        
    def predict(self,
                X_list,
                num_samples,
                return_sites):
        if self.model.model_type == "MatrixDecomp":
            # X_list is just a single matrix
            n = X_list[0].shape[0]
            # n = X_list.shape[0]
            predictive = Predictive(self.forward, 
                                    guide = self.guide, 
                                    num_samples = num_samples,
                                    return_sites = return_sites)
            return predictive(
                X_list,
                torch.arange(n)
                )
        elif self.model.model_type == "SupMultiviewDecomp":
            n = X_list[0].shape[0]
            predictive = Predictive(self.forward, 
                                    guide = self.guide, 
                                    num_samples = num_samples,
                                    return_sites = return_sites)
            return predictive(
                X_list,
                torch.arange(n)
                )