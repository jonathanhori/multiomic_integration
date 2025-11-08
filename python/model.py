import os
import sys
# import pyreadr

from functools import partial

import math
import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO
from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal

from torch.utils.data import TensorDataset, DataLoader

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class SupMultiviewDecomp(PyroModule):
    def __init__(self, 
                 k, 
                 k_l_list,
                 n,
                 include_view_factors = False,
                 a_sigma_joint=2.0, # Hyperparams: ARD prior per factor: IG same for all views
                 b_sigma_joint=2.0,
                 a_sigma_view=2.0, # Hyperparams: ARD prior per factor: IG same for all views
                 b_sigma_view=2.0,
                 a_psi=2.0, # Hyperparams: per-view error: IG same for all views
                 b_psi=2.0,
                 a_sigma_y = 2.0, # Hyperparams: outcome error: IG
                 b_sigma_y = 2.0,
                 a_sigma_beta = 2.0, # Hyperparams: outcome coefficients: IG
                 b_sigma_beta = 2.0
                 ):
        super().__init__()
        # Model parameters
        self.n = n
        self.k = k
        self.k_l_list = k_l_list # assumes k_l > 0 for all l
        self.p_l = None
        
        # Hyperparams
        self.a_sigma_joint = a_sigma_joint
        self.b_sigma_joint = b_sigma_joint
        self.a_sigma_view = a_sigma_view
        self.b_sigma_view = b_sigma_view
        self.a_psi = a_psi
        self.b_psi = b_psi
        self.a_sigma_y = a_sigma_y
        self.b_sigma_y = b_sigma_y
        self.a_sigma_beta = a_sigma_beta
        self.b_sigma_beta = b_sigma_beta
        
        # Should view-specific factors be included in the outcome model?
        self.include_view_factors = include_view_factors
        self.num_outcome_factors = k if not include_view_factors else sum((k, *k_l_list))
        
        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []
        
    
    def forward(self, 
                X_list, 
                batch_idx,
                y = None):
        """
        X_l = Z @ Lambda_l^T + Phi_l @ Gamma_l^T + E,   l = 1, ... , L
        y = Z @ beta + e
        """
        m = len(batch_idx)
        p_l_list = [X_l.shape[1] for X_l in X_list]
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        # self.p = 
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # ARD prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda^l_jk ~ N(0, sigma_lambda^l_k^2)
        # sigma^l_k^2 ~ InvGamma(a_sigma, b_sigma))
        #
        # Gamma^l_jk ~ N(0, sigma_gamma^l_k^2)
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(p_l_list, self.k_l_list)):
            sigma2_gamma_l = pyro.sample(f"sigma2_gamma_l{l}", 
                                   dist.InverseGamma(self.a_sigma_view, self.b_sigma_view).expand([k_l]).to_event(1))
            sigma_gamma_l = torch.sqrt(sigma2_gamma_l)
        
            # sigma is broadcast across rows of Lambda
            Gamma_l = pyro.sample(f"Gamma_l{l}", dist.Normal(torch.zeros(p_l, k_l), sigma_gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED
        Lambda_l_list = []
        for l, p_l in enumerate(p_l_list):
            sigma2_lambda_l = pyro.sample(f"sigma2_lambda_l{l}", 
                                   dist.InverseGamma(self.a_sigma_joint, self.b_sigma_joint).expand([self.k]).to_event(1))
            sigma_lambda_l = torch.sqrt(sigma2_lambda_l)
        
            # sigma is broadcast across rows of Lambda
            Lambda_l = pyro.sample(f"Lambda_l{l}", dist.Normal(torch.zeros(p_l, self.k), sigma_lambda_l).to_event(2))
            
            Lambda_l_list.append(Lambda_l)
        
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        #
        # Outcome y can be dependent on just shared factors, or all factors
        ########################
        
        # Idiosyncratic error variance 
        psi_sqrt_l_list = []           
        for l, p_l in enumerate(p_l_list):
            psi_l = pyro.sample(f"psi_l{l}", dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_sqrt = torch.sqrt(psi_l)
            psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        sigma2_beta = pyro.sample("sigma2_beta",
                                  dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        beta = pyro.sample("beta",
                           dist.Normal(torch.zeros(self.num_outcome_factors), sigma2_beta).to_event(1)).\
                               squeeze(0)
        
        # Outcome variances
        sigma2_y = pyro.sample("sigma2_y",
                            dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
        sigma_y = torch.sqrt(sigma2_y)
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(f"Phi_l{l}", dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)
                
            
            # Compute structures
            joint_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(f"joint_structure_l{l}", 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(f"view_structure_l{l}",
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                
                pyro.sample(f"X_l{l}", dist.Normal(total_structure_l, 
                                                   psi_sqrt_l_list[l]).to_event(1), 
                            obs = X_list[l])
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # Outcome model
            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic("outcome_structure",
                                                    torch.matmul(full_factors, beta))
                pyro.sample("y", dist.Normal(outcome_structure, 
                                            sigma_y), #.to_event(1),
                            obs = y)
            else:
                outcome_structure = pyro.deterministic("outcome_structure",
                                                    torch.matmul(Z, beta))
                pyro.sample("y", dist.Normal(outcome_structure, 
                                            sigma_y), #.to_event(1),
                            obs = y)
            # pyro.sample("y", dist.Normal(outcome_structure.index_select(0, batch_idx), 
            #                              sigma_y),
            #             obs = y.index_select(0, batch_idx))
            
            
    def guide(self, 
              X_list, 
              batch_idx,
              y = None):
        raise NotImplementedError
        # TODO
    
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
            
        # ######
        # # Loadings
        # # The model specifies 
        # ######
        # # Sigma2: Variational parameters a, b
        # a_sigma_q = pyro.param("a_sigma_q", 
        #                        lambda: torch.ones((self.k)),
        #                        constraint = dist.constraints.positive)
        # b_sigma_q = pyro.param("b_sigma_q", 
        #                        lambda: torch.ones((self.k)),
        #                        constraint = dist.constraints.positive)
        
        # sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_q, b_sigma_q).to_event(1))
        # # sigma = torch.sqrt(sigma2)
        
        # # Lambda: Variational parameters loc, scale
        # Lambda_loc = pyro.param("Lambda_loc", lambda: torch.zeros(p, self.k))
        # Lambda_scale = pyro.param("Lambda_scale", lambda: torch.ones(p, self.k),
        #                           constraint = dist.constraints.positive)
        # Lambda = pyro.sample("Lambda", dist.Normal(Lambda_loc, Lambda_scale).to_event(2))
        
        # ########################
        # # Scores
        # ########################
        # # Psi: Variational params a, b
        # a_psi_q = pyro.param("a_psi_q",
        #                      lambda: torch.ones((p)),
        #                      constraint = dist.constraints.positive)     
        # b_psi_q = pyro.param("b_psi_q",
        #                      lambda: torch.ones((p)),
        #                      constraint = dist.constraints.positive)  
        # psi = pyro.sample("psi", dist.InverseGamma(a_psi_q, b_psi_q).to_event(1))
        # # psi_sqrt = torch.sqrt(psi)
        
        # # Local latent variables Z: Variational params loc, scale
        # Z_loc = pyro.param("Z_loc", lambda: torch.zeros(self.n, self.k))
        # Z_scale = pyro.param("Z_scale", lambda: torch.ones(self.n, self.k),
        #                         constraint = dist.constraints.positive)
        # with pyro.plate("obs", self.n, subsample = batch_idx):
        #     Z_loc_batch = Z_loc[batch_idx]
        #     Z_scale_batch = Z_scale[batch_idx]
        #     Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
        #     pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
    
##############
def do_inference(X_list, 
                 y,
                 model, 
                 guide, 
                 opt = Adam({"lr": 0.001}), #"Adam",
                 elbo = Trace_ELBO(),
                #  opt_args = {"lr": 0.001},
                 min_epochs = 10,
                 epochs = 20,
                 max_iter = 20000,
                 minibatch_flag = False,
                 minibatch_size = 32,
                 tol = 1e-4,
                 variational_tol = 1e-4,
                 device = "cpu",
                 verbose = False):
    
    # min_epochs = 10
    
    # if minibatch_flag == False, then do not minibatch, data loader not necessary
    if not minibatch_flag:
        minibatch_size = model.n
        
    ########################
    # Handle minibatching of dataset
    ########################
    tensor_dataset = TensorDataset(torch.arange(X_list[0].shape[0]), *X_list, y)
    loader = DataLoader(tensor_dataset, batch_size = minibatch_size, shuffle = True, drop_last=True)
    
    ########################
    # Initialize instances for optimization
    ########################
    # guide = self.guide
    # if opt == "Adam":
    #     opt = Adam(opt_args)
    # elif opt == "ClippedAdam":
    #     opt = ClippedAdam(opt_args)
    # elbo = Trace_ELBO(num_particles = 1)
    # elbo = TraceGraph_ELBO()
    svi = SVI(model, guide, opt, loss = elbo)
    
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
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / model.n:.4f}")
                print(f"Loss at epoch {epoch+1}: {loss / model.n}")
            
            model.loss_history.append(epoch_loss)
            
            # Determine the epoch when the min loss is obtained
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                epoch_at_min_loss = epoch
            
            model.total_epochs += 1
            param_store_curr = pyro.get_param_store()
            params_epoch_curr = {k: v.detach().clone() for k, v in param_store_curr.items()}
            if epoch == 0: 
                params_epoch_last = params_epoch_curr
            # Check Euclidean norm of difference in variational params for convergence
            param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item() for k in pyro.get_param_store()}
            model.var_param_convergence_history.append(max(param_diff_norm_dict.values()))
            # print(params_epoch_curr.items())
            params_converged = all(n < variational_tol for n in param_diff_norm_dict.values())
            
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
        # # total_loss = 0.
        # prev_loss = None
        # for iter in range(max_iter):
        #     # total_loss = 0.
        #     loss = svi.step(X, torch.arange(X.shape[0])) #batch[0])
        #     # print(loss)
        #     # total_loss += loss
        #     # for batch in loader:
        #     #     loss = svi.step(batch[0])
        #     #     epoch_loss += loss
        #     if iter % 100 == 0:
        #         print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
        #         model.loss_history.append(loss)
        #         # print(f"Epoch {iter+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss:.4f}")
                
        #     # if prev_loss is not None and abs((loss- prev_loss) / prev_loss) < tol:
        #     if prev_loss is not None and abs((loss- prev_loss)) < tol:
        #         # print(f"Stopping: {abs((loss- prev_loss) / prev_loss)} < {tol}")
        #         print(f"Iteration {iter-1}  loss: {prev_loss / model.n:.4f}")
        #         print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
        #         model.total_iters = iter
        #         break
        #     prev_loss = loss
            
        
    return svi, opt