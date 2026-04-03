import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule

from torch.utils.data import TensorDataset, DataLoader

from constants import Sites, Params
from data_utils import align_tensor_shapes

pyro.enable_validation(True)
pyro.set_rng_seed(0)
 

class MatrixDecomp(PyroModule):
    def __init__(self, k,
                 supervised = False,
                 loading_target = None,
                 dense = False,
                 penalty_obj = None,
                 a_sigma=2.0, # Hyperparams: Conjugate prior per factor: IG
                 b_sigma=2.0,
                 a1_sigma=2.1, # Hyperparams: Conjugate prior per factor: IG
                 a2_sigma=3.1,
                 alpha=3.0,
                 a_psi=2.0, # Hyperparams: per-view error: IG
                 b_psi=2.0):
        super().__init__()
        
        self.model_type = "MatrixDecomp"
        # Model parameters
        self.k = k
        self.n = None
        self.p = None
        
        self.ortho_penalty = None
        self.penalty_obj = penalty_obj
        self.supervised_model = supervised
        
        # Should the loading matrix be penalized toward a target?
        if loading_target is not None:
            self.loading_target = align_tensor_shapes(loading_target, 
                                                    additional = k - loading_target.shape[1])
        else:
            self.loading_target = None
        
        # Hyperparams
        # self.a_sigma = a_sigma
        # self.b_sigma = b_sigma
        
        self.a_psi = a_psi
        self.b_psi = b_psi
        
        if dense:
            self.a_sigma = a_sigma
            self.b_sigma = b_sigma
        if not dense:
            self.a1_sigma = a1_sigma
            self.a2_sigma = a2_sigma
            self.alpha = alpha
        
        # With outcome model
        self.outcome = "gaussian"
        self.a_sigma_y = 3.0
        self.b_sigma_y = 1.0
        self.a_sigma_beta = 2.0
        self.b_sigma_beta = 2.0
        
        self.inv_gamma_init_param = 2.1
        self.init_param = 0.1
        self.init_scale_param = 1.0
        
        self.dense = dense
        # self.loss_history = []
        
        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []
        
        
    def forward(self, 
                X, 
                batch_idx,
                y = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.forward_dense(X, 
                batch_idx,
                y)
        else:
            return self.forward_mgp(X, 
                batch_idx,
                y)
            
    def guide(self, 
            X, 
            batch_idx,
            y = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.guide_dense(X, 
                batch_idx,
                y)
        else:
            return self.guide_mgp(X, 
                batch_idx,
                y)
    
    def forward_mgp(self, X, batch_idx, y = None):
        l = 0
        
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        # self.p = 
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
        # if self.p_l_list is None:
        #     self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # MGP prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda_jk ~ N(0, N(0, (rho^l_{jk})^-1 * (tau^l_k)^-12))
        ########################
        # tau - global shrinkage
        
        delta_lambda = []
        for m in range(self.k):
            shape = self.a1_sigma if m == 0 else self.a2_sigma
            delta_l_m = pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                                    dist.Gamma(shape, 1.0))
            delta_lambda.append(delta_l_m)
        tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim = 0).squeeze()
        # print(tau_lambda_k_list.shape)

        # rho - local shrinkage
        rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l = l), 
                            dist.Gamma(self.alpha / 2, self.alpha / 2).expand([self.p, self.k]).to_event(2)).squeeze()
        
        # print(rho_lambda.shape)
        # print('Lambda - tau shape', tau_lambda_k_list.shape)
        # print('Lambda - rho shape', rho_lambda.shape)
        # sigma_lambda_l = (rho_lambda * tau_lambda_k_list).pow_(-0.5)   
        precision = rho_lambda * tau_lambda_k_list
        precision = torch.clamp(precision, min=1e-10, max=1e10)
        sigma_lambda_l = precision.pow_(-0.5)     
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                            dist.Normal(torch.zeros(self.p, self.k), sigma_lambda_l).to_event(2)).squeeze()
        
        # sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1))
        # sigma = torch.sqrt(sigma2).unsqueeze(-2).expand(self.p, self.k)
        
        # # sigma is broadcast across rows of Lambda
        # Lambda = pyro.sample("Lambda", dist.Normal(torch.zeros(self.p, self.k), sigma).to_event(2))
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample("psi", dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx): # X is a minibatch, can pass directly into subsample
            # Latent scores Z_i are local
            # Z_batch = pyro.sample("Z", dist.Normal(torch.zeros(m, self.k), torch.ones(self.k)).to_event(1))
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            # if torch.isnan(structure).any():
            #         print("NaN values found in X_l_tensor!")
            # if torch.isinf(structure).any():
            #     print("Inf values found in X_l_tensor!")
                
            # print("finite structure?") 
            # print(torch.isfinite(structure).all())
            # print("finite sd?")
            # print(torch.isfinite(psi_sqrt).all())
            # print("geq 0 sd?") 
            # print((psi_sqrt > 0).all())
            # print("X finite:", torch.isfinite(X).all())
            # print("NaNs:", torch.isnan(X).sum())
            # print("Infs:", torch.isinf(X).sum())
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0]) # list of 1 element 
        
        

    
    def forward_dense(self, X, batch_idx, y = None):
        l = 0
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        # self.p = 
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # ARD prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda_jk ~ N(0, sigma_k^2)
        # sigma_k^2 ~ InvGamma(a_sigma, b_sigma))
        ########################
        # No row-wise conditional independence of loadings.
        sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1))
        sigma = torch.sqrt(sigma2)
        
        # sigma is broadcast across rows of Lambda
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                             dist.Normal(torch.zeros(self.p, self.k), sigma).to_event(2))
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample("psi", dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.k]).to_event(1))
            beta = pyro.sample(Sites.beta,
                            dist.Normal(torch.zeros(self.k), sigma2_beta).to_event(1)).\
                                squeeze(0)
            
            # Outcome variances
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx): # X is a minibatch, can pass directly into subsample
            # Latent scores Z_i are local
            # Z_batch = pyro.sample("Z", dist.Normal(torch.zeros(m, self.k), torch.ones(self.k)).to_event(1))
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.squeeze(0).T))
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0]) # list of 1 element 
            
            if self.supervised_model:
                # Outcome model
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta.squeeze(0)))
                if self.outcome == "gaussian":
                    pyro.sample("y", dist.Normal(outcome_structure, 
                                                sigma_y), #.to_event(1),
                                obs = y)
                else:
                    raise NotImplementedError
            
            
            
    def guide_dense(self, X, batch_idx, y = None):
        l = 0
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        # print(X)
        # print(len(X))
        # print(X[0].shape)
        # print(X[1])
            
        ######
        # Loadings
        ######
        
        # Sigma2: Variational parameters a, b
        a_sigma_lambda = pyro.param("a_sigma_lambda", 
                               lambda: torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        b_sigma_lambda = pyro.param("b_sigma_lambda", 
                               lambda: torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        sigma2_lambda = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

        loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        scale_Lambda = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(self.p, self.k),
                                    constraint = dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(loc_Lambda, scale_Lambda).to_event(2))
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi = pyro.param("a_psi",
                             lambda: torch.ones((self.p)),
                             constraint = dist.constraints.positive)     
        b_psi = pyro.param("b_psi",
                             lambda: torch.ones((self.p)),
                             constraint = dist.constraints.positive)  
        psi = pyro.sample("psi", dist.InverseGamma(a_psi, b_psi).to_event(1))
        # psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                  torch.tensor(self.inv_gamma_init_param),
                                  constraint = dist.constraints.positive)
            b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                    torch.tensor(self.init_param),
                                    constraint = dist.constraints.positive)
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))
        
            loc_beta = pyro.param(Params.loc_beta, 
                                #   lambda: torch.randn(self.k))
                                  torch.zeros(self.k))
            scale_beta = pyro.param(Params.scale_beta, 
                                    # torch.ones(self.num_outcome_factors),
                                    torch.tensor(self.init_scale_param).expand([self.k]),
                                    constraint = dist.constraints.positive)
            beta = pyro.sample(Sites.beta,
                            dist.Normal(loc_beta, scale_beta).to_event(1)).\
                                squeeze(0)
                                
                            
            # Outcome variances
            if self.outcome == "gaussian":
                a_sigma_y = pyro.param(Params.a_sigma_y, 
                                    torch.tensor(self.a_sigma_y),
                                    constraint = dist.constraints.positive)
                b_sigma_y = pyro.param(Params.b_sigma_y, 
                                    torch.tensor(self.b_sigma_y),
                                    constraint = dist.constraints.positive)
                sigma2_y = pyro.sample(Sites.sigma2_y,
                                    dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
        
        # Local latent variables Z: Variational params loc, scale
        Z_loc = pyro.param("Z_loc", lambda: torch.zeros(self.n, self.k))
        Z_scale = pyro.param("Z_scale", lambda: torch.ones(self.n, self.k),
                                constraint = dist.constraints.positive)
        
        # y_loc = pyro.param("y_loc", lambda: torch.zeros(self.n))
        # y_scale = pyro.param("y_scale", lambda: torch.ones(self.n),
        #                         constraint = dist.constraints.positive)
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
            pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
            if self.supervised_model:
                # Outcome model
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta))
                # if self.outcome == "gaussian":
                #     loc_y_batch = y_loc[batch_idx]
                #     scale_y_batch = y_scale[batch_idx]
                #     pyro.sample("y", 
                #                 dist.Normal(loc_y_batch, scale_y_batch), #.to_event(1),
                #                 obs = y)
                # else:
                #     raise NotImplementedError
            
            
    def guide_mgp(self, X, batch_idx, y = None):
        l = 0
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        # print(X)
        # print(len(X))
        # print(X[0].shape)
        # print(X[1])
            
        ######
        # Loadings
        # The model specifies 
        ######
        for m in range(self.k):
            a_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(self.a1_sigma),
                                            constraint = dist.constraints.positive)
            b_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(1.0),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                        dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

        a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2),
                                        constraint = dist.constraints.positive)
        b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2),
                                        constraint = dist.constraints.positive)
        pyro.sample(Sites.rho_lambda_l.format(l = l), 
                    dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([self.p, self.k]).to_event(2)).squeeze()
        
        loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(self.p, self.k),
                                    constraint = dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
        
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi_q = pyro.param("a_psi_q",
                             lambda: torch.ones((self.p)),
                             constraint = dist.constraints.positive)     
        b_psi_q = pyro.param("b_psi_q",
                             lambda: torch.ones((self.p)),
                             constraint = dist.constraints.positive)  
        psi = pyro.sample("psi", dist.InverseGamma(a_psi_q, b_psi_q).to_event(1))
        # psi_sqrt = torch.sqrt(psi)
        
        # Local latent variables Z: Variational params loc, scale
        Z_loc = pyro.param("Z_loc", lambda: torch.zeros(self.n, self.k))
        Z_scale = pyro.param("Z_scale", lambda: torch.ones(self.n, self.k),
                                constraint = dist.constraints.positive)
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
            pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
        