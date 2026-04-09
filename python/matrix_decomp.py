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
                 a_sigma=2.1, # Hyperparams: Conjugate prior per factor: IG
                 b_sigma=2.0,
                 a1_sigma=2.1, # Hyperparams: Conjugate prior per factor: IG
                 a2_sigma=3.1,
                 alpha=3.0,
                 a_psi=3.0, # Hyperparams: per-view error: IG
                 b_psi=1.0,
                 scores_init = None,
                 loadings_init = None):
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
        self.b_sigma_y = 2.0
        self.a_sigma_beta = 2.1
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
        
        self.scores_init = scores_init
        self.loadings_init = loadings_init
        
        
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
            
    #######
    # SPARSE MODEL - MGP PRIOR ON LAMBDA
    #######
    
    def forward_mgp(self, X, batch_idx, y = None):
        l = 0
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
        
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
            delta_m = pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                                    dist.Gamma(shape, 1.0))
            delta_lambda.append(delta_m)
        tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim = 0).squeeze()

        # rho - local shrinkage
        rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l = l), 
                            dist.Gamma(self.alpha / 2, self.alpha / 2).expand([self.p, self.k]).to_event(2)).squeeze()
         
        precision = rho_lambda * tau_lambda_k_list
        precision = torch.clamp(precision, min=1e-10, max=1e10)
        sigma_lambda = precision.pow_(-0.5)     
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                            dist.Normal(torch.zeros(self.p, self.k), sigma_lambda).to_event(2)).squeeze()
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample(Sites.psi_l.format(l = l), dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.k]).to_event(1))
            sigma_beta = torch.sqrt(sigma2_beta)
            beta = pyro.sample(Sites.beta,
                            dist.Normal(torch.zeros(self.k), sigma_beta).to_event(1)).squeeze(0)
            
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
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0]) # list of 1 element 
            
            if self.supervised_model:
                # Outcome model
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta.squeeze(0)))
                if self.outcome == "gaussian":
                    pyro.sample("y", dist.Normal(outcome_structure, 
                                                sigma_y),
                                obs = y)
                else:
                    raise NotImplementedError
        
        
          
    def guide_mgp(self, X, batch_idx, y = None):
        l = 0
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        ######
        # Loadings
        # The model specifies 
        ######
        for m in range(self.k):
            a_delta_lambda_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(self.a1_sigma),
                                            constraint = dist.constraints.positive)
            b_delta_lambda_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(1.0),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                        dist.Gamma(a_delta_lambda_k, b_delta_lambda_k))

        a_rho_lambda = pyro.param(Params.a_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2) * torch.ones(self.p, self.k),
                                        constraint = dist.constraints.positive)
        b_rho_lambda = pyro.param(Params.b_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2) * torch.ones(self.p, self.k),
                                        constraint = dist.constraints.positive)
        pyro.sample(Sites.rho_lambda_l.format(l = l), 
                    dist.Gamma(a_rho_lambda, b_rho_lambda).to_event(2)).squeeze() # .expand([self.p, self.k]
        
        loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        scale_Lambda = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(self.p, self.k),
                                    constraint = dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(loc_Lambda, scale_Lambda).to_event(2))
        
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi = pyro.param(Params.a_psi_l.format(l = l),
                             lambda: self.a_psi * torch.ones((self.p)),
                             constraint = dist.constraints.positive)     
        b_psi = pyro.param(Params.b_psi_l.format(l = l),
                             lambda: self.b_psi * torch.ones((self.p)),
                             constraint = dist.constraints.positive)  
        psi = pyro.sample(Sites.psi_l.format(l = l), dist.InverseGamma(a_psi, b_psi).to_event(1))
        # psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                self.a_sigma_beta * torch.ones(self.k),
                                  constraint = dist.constraints.positive)
            b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                    self.b_sigma_beta * torch.ones(self.k),
                                    constraint = dist.constraints.positive)
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))
        
            loc_beta = pyro.param(Params.loc_beta, 
                                  torch.zeros(self.k))
            scale_beta = pyro.param(Params.scale_beta, 
                                    torch.tensor(self.init_scale_param).expand([self.k]),
                                    constraint = dist.constraints.positive)
            beta = pyro.sample(Sites.beta,
                            dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)
                                
                            
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
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
            pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
            if self.supervised_model:
                # Outcome model
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta))
      
      
        
    ########################################################################################
    # DENSE MODEL - NORMAL-IG PRIOR ON LAMBDA
    ########################################################################################   
    
    def forward_dense(self, X, batch_idx, y = None):
        l = 0
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
        sigma2_lambda = pyro.sample("sigma2_lambda", dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1))
        sigma_lambda = torch.sqrt(sigma2_lambda)
        
        # sigma is broadcast across rows of Lambda
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                             dist.Normal(torch.zeros(self.p, self.k), sigma_lambda).to_event(2))
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample(Sites.psi_l.format(l = l), dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.k]).to_event(1))
            sigma_beta = torch.sqrt(sigma2_beta)
            beta = pyro.sample(Sites.beta,
                            dist.Normal(torch.zeros(self.k), sigma_beta).to_event(1)).squeeze(0)
                                # squeeze(0) #\
            
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
                    # with pyro.poutine.scale(scale=self.p):
                    pyro.sample("y", dist.Normal(outcome_structure, 
                                                sigma_y), #.to_event(1),
                                obs = y)
                else:
                    raise NotImplementedError
            
            
            
    def guide_dense(self, X, batch_idx, y = None):
        l = 0
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        ######
        # Loadings
        ######
        
        # Sigma2: Variational parameters a, b
        a_sigma_lambda = pyro.param("a_sigma_lambda", 
                               lambda: self.a_sigma * torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        b_sigma_lambda = pyro.param("b_sigma_lambda", 
                               lambda: self.b_sigma * torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        sigma2_lambda = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

        if self.loadings_init is not None:
            loc_Lambda = align_tensor_shapes(self.loadings_init, 
                                            additional = self.k - self.loadings_init.shape[1])
            loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l = l), 
                                        loc_Lambda
                                        )
        else:
            loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        # loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        scale_Lambda = pyro.param(Params.scale_Lambda_l.format(l = l), 
                                  0.1 * torch.ones(self.p, self.k),
                                    constraint = dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(loc_Lambda, scale_Lambda).to_event(2))
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi = pyro.param("a_psi",
                             lambda: self.a_psi * torch.ones((self.p)),
                             constraint = dist.constraints.positive)     
        b_psi = pyro.param("b_psi",
                             lambda: self.b_psi * torch.ones((self.p)),
                             constraint = dist.constraints.positive)  
        psi = pyro.sample(Sites.psi_l.format(l = l), dist.InverseGamma(a_psi, b_psi).to_event(1))
        # psi_sqrt = torch.sqrt(psi)
        
        if self.supervised_model:
            # Outcome model coefficients
            a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                self.a_sigma_beta * torch.ones(self.k),
                                  constraint = dist.constraints.positive)
            b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                    self.b_sigma_beta * torch.ones(self.k),
                                    constraint = dist.constraints.positive)
            sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))
        
            loc_beta = pyro.param(Params.loc_beta,
                                  torch.zeros(self.k))
            scale_beta = pyro.param(Params.scale_beta, 
                                    # torch.ones(self.num_outcome_factors),
                                    torch.tensor(self.init_scale_param).expand([self.k]),
                                    constraint = dist.constraints.positive)
            beta = pyro.sample(Sites.beta,
                            dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0) # .\
                                # squeeze(0)
                                
                            
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
        if self.scores_init is not None:
            Z_loc = align_tensor_shapes(self.scores_init,
                                        additional = self.k - self.scores_init.shape[1])
            Z_loc = pyro.param(Params.loc_Z, 
                               Z_loc
            )
        else:
            Z_loc = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        # Z_loc = pyro.param("Z_loc", lambda: torch.zeros(self.n, self.k))
        Z_scale = pyro.param("Z_scale", lambda: torch.ones(self.n, self.k), # / torch.sqrt(torch.tensor(self.k)),
                                constraint = dist.constraints.positive)
        
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
            pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
            if self.supervised_model:
                # Outcome model
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta))
            
      