import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal
from pyro.poutine import mask

from constants import Sites, Params

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class SupMultiviewDecomp(PyroModule):
    def __init__(self, 
                 k, 
                 k_l_list,
                 n,
                 include_view_factors = False,
                 dense = True,
                 outcome = "gaussian",
                 a_sigma_joint=2.0, # Hyperparams: gamma process prior per factor loadings
                 b_sigma_joint=2.0,
                 a_sigma_view=2.0, # Hyperparams: gamma process prior per factor loadings
                 b_sigma_view=2.0,
                 a1_sigma_joint=2.1, # Hyperparams: gamma process prior per factor loadings
                 a2_sigma_joint=3.1,
                 a1_sigma_view=2.1, # Hyperparams: gamma process prior per factor loadings
                 a2_sigma_view=3.1,
                 alpha_joint = 3.0,
                 alpha_individual = 3.0,
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
        # self.p_l = None
        self.p_l_list = None
        
        self.dense = dense
        self.outcome = outcome
        
        # Hyperparams
        if dense:
            self.a_sigma_joint = a_sigma_joint
            self.b_sigma_joint = b_sigma_joint
            self.a_sigma_view = a_sigma_view
            self.b_sigma_view = b_sigma_view
        if not dense:
            self.a1_sigma_joint = a1_sigma_joint
            self.a2_sigma_joint = a2_sigma_joint
            self.a1_sigma_view = a1_sigma_view
            self.a2_sigma_view = a2_sigma_view
            self.alpha_joint = alpha_joint
            self.alpha_individual = alpha_individual
        
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
        
        self.params = None
        
    def forward(self, 
                X_list, 
                batch_idx,
                y = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.forward_dense(X_list, 
                batch_idx,
                y)
        else:
            return self.forward_mgp(X_list, 
                batch_idx,
                y)
        
        
    def forward_mgp(self, 
                X_list, 
                batch_idx,
                y = None,
                cens = None):
        """
        X_l = Z @ Lambda_l^T + Phi_l @ Gamma_l^T + E,   l = 1, ... , L
        y = Z @ beta + e
        With shrinkage of factor loadings
        """
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda^l_jk ~ N(0, (rho^l_{jk})^-1 * (tau^l_k)^-12)
        # sigma^l_k^2 ~ InvGamma(a_sigma, b_sigma))
        #
        # Gamma^l_jk ~ N(0, sigma_gamma^l_k^2)
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            # tau - global shrinkage
            delta_gamma = []
            for m in range(k_l):
                shape = self.a1_sigma_view if m == 0 else self.a2_sigma_view
                delta_l_m = pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), 
                                        dist.Gamma(shape, 1.0))
                delta_gamma.append(delta_l_m)
            tau_gamma_k_list = torch.cumprod(torch.stack(delta_gamma), dim = 0).squeeze()
                
            # rho - local shrinkage
            rho_gamma = pyro.sample(Sites.rho_gamma_l.format(l = l),
                                    dist.Gamma(self.alpha_individual / 2, self.alpha_individual / 2).expand([p_l, k_l]).to_event(2)).squeeze()
            
            # print('Gamma - tau shape', tau_gamma_k_list.shape)
            # print('Gamma - rho shape', rho_gamma.shape)
            # sigma_gamma_l = (rho_gamma * tau_gamma_k_list).pow_(-0.5)
            precision = rho_gamma * tau_gamma_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)  # Prevent extreme values
            sigma_gamma_l = precision.pow_(-0.5)
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(torch.zeros(p_l, k_l), sigma_gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            # tau - global shrinkage
            delta_lambda = []
            for m in range(self.k):
                shape = self.a1_sigma_joint if m == 0 else self.a2_sigma_joint
                delta_l_m = pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                                        dist.Gamma(shape, 1.0))
                delta_lambda.append(delta_l_m)
            tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim = 0).squeeze()
        
            # rho - local shrinkage
            rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l = l), 
                                dist.Gamma(self.alpha_joint / 2, self.alpha_joint / 2).expand([p_l, self.k]).to_event(2)).squeeze()
            
            # print('Lambda - tau shape', tau_lambda_k_list.shape)
            # print('Lambda - rho shape', rho_lambda.shape)
            # sigma_lambda_l = (rho_lambda * tau_lambda_k_list).pow_(-0.5)   
            precision = rho_lambda * tau_lambda_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)
            sigma_lambda_l = precision.pow_(-0.5)     
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l = l), 
                                   dist.Normal(torch.zeros(p_l, self.k), sigma_lambda_l).to_event(2))
            
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
        for l, p_l in enumerate(self.p_l_list):
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_sqrt = torch.sqrt(psi_l)
            psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.num_outcome_factors), sigma2_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        else:
            # outcome_concentration = pyro.sample("")
            raise NotImplementedError
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)
                
            
            # Compute structures
            joint_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l.format(l = l), 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l.format(l = l),
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                
                X_l = pyro.sample(Sites.X_l.format(l = l), 
                                  dist.Normal(total_structure_l, psi_sqrt_l_list[l]).to_event(1), 
                            obs = X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # Outcome model
            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                    torch.matmul(full_factors, beta))
            else:
                outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                    torch.matmul(Z, beta))
                # pyro.sample("y", dist.Normal(outcome_structure, 
                #                             sigma_y), #.to_event(1),
                #             obs = y)
            
            if self.outcome == "gaussian":
                pyro.sample(Sites.y, dist.Normal(outcome_structure, 
                                            sigma_y), #.to_event(1),
                            obs = y)
            elif self.outcome == "censored":
                weibull = dist.Weibull(outcome_structure, outcome_concentration)
                with mask(mask = (cens == 1)):
                    pyro.sample(Sites.y, weibull, obs = y)
                with mask(mask = (cens == 0)):
                    survival_prob = 1 - weibull.cdf(y)
                    pyro.sample(Sites.censored, dist.Bernoulli(survival_prob), obs = 1)
            else:
                raise NotImplementedError
    
    def predict_forward(self,
                        X_list, 
                        batch_idx,
                        y = None):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        
        """
        X_l = Z @ Lambda_l^T + Phi_l @ Gamma_l^T + E,   l = 1, ... , L
        y = Z @ beta + e
        With shrinkage of factor loadings
        """
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda^l_jk ~ N(0, (rho^l_{jk})^-1 * (tau^l_k)^-12)
        # sigma^l_k^2 ~ InvGamma(a_sigma, b_sigma))
        #
        # Gamma^l_jk ~ N(0, sigma_gamma^l_k^2)
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            # Can ignore shrinkage prior parameters, only need final distributions of loadings
            # tau - global shrinkage
            delta_gamma = []
            for m in range(k_l):
                # shape = self.a1_sigma_view if m == 0 else self.a2_sigma_view
                a_delta_gamma_l_k = self.params[Params.a_delta_gamma_l_k.format(l = l, m = m)]
                b_delta_gamma_l_k = self.params[Params.b_delta_gamma_l_k.format(l = l, m = m)]
                delta_l_m = pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), 
                                        dist.Gamma(a_delta_gamma_l_k, b_delta_gamma_l_k))
                delta_gamma.append(delta_l_m)
            # tau_gamma_k_list = torch.cumprod(torch.stack(delta_gamma), dim = 0).squeeze()
                
            # rho - local shrinkage
            a_rho_gamma_l = self.params[Params.a_rho_gamma_l]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l]
            rho_gamma = pyro.sample(Sites.rho_gamma_l.format(l = l),
                                    dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).expand([p_l, k_l]).to_event(2)).squeeze()
            
            # print('Gamma - tau shape', tau_gamma_k_list.shape)
            # print('Gamma - rho shape', rho_gamma.shape)
            # sigma_gamma_l = (rho_gamma * tau_gamma_k_list).pow_(-0.5)
            # precision = rho_gamma * tau_gamma_k_list
            # precision = torch.clamp(precision, min=1e-10, max=1e10)  # Prevent extreme values
            # sigma_gamma_l = precision.pow_(-0.5)
            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l = l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l = l)]
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            # tau - global shrinkage
            delta_lambda = []
            for m in range(self.k):
                # shape = self.a1_sigma_joint if m == 0 else self.a2_sigma_joint
                a_delta_lambda_l_k = self.params[Params.a_delta_lambda_l_k.format(l = l, m = m)]
                b_delta_lambda_l_k = self.params[Params.b_delta_lambda_l_k.format(l = l, m = m)]
                delta_l_m = pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                                        dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))
                delta_lambda.append(delta_l_m)
            # tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim = 0).squeeze()
        
            # rho - local shrinkage
            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l = l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l = l)]
            rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l = l), 
                                dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()
            
            # # print('Lambda - tau shape', tau_lambda_k_list.shape)
            # # print('Lambda - rho shape', rho_lambda.shape)
            # # sigma_lambda_l = (rho_lambda * tau_lambda_k_list).pow_(-0.5)   
            # precision = rho_lambda * tau_lambda_k_list
            # precision = torch.clamp(precision, min=1e-10, max=1e10)
            # sigma_lambda_l = precision.pow_(-0.5)     
            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l = l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l = l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l = l), 
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            
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
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l = l)]
            b_psi_l = self.params[Params.b_psi_l.format(l = l)]
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            psi_sqrt = torch.sqrt(psi_l)
            psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        
        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        a_sigma_y = self.params[Params.a_sigma_y]
        b_sigma_y = self.params[Params.b_sigma_y]
        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        else:
            # outcome_concentration = pyro.sample("")
            raise NotImplementedError
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)
                
            
            # Compute structures
            joint_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l.format(l = l), 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l.format(l = l),
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                
                X_l = pyro.sample(Sites.X_l.format(l = l), 
                                  dist.Normal(total_structure_l, psi_sqrt_l_list[l]).to_event(1), 
                            obs = X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # Outcome model
            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                    torch.matmul(full_factors, beta))
            else:
                outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                    torch.matmul(Z, beta))
                # pyro.sample("y", dist.Normal(outcome_structure, 
                #                             sigma_y), #.to_event(1),
                #             obs = y)
            
            if self.outcome == "gaussian":
                pyro.sample(Sites.y, dist.Normal(outcome_structure, 
                                            sigma_y), #.to_event(1),
                            obs = y)
            elif self.outcome == "censored":
                weibull = dist.Weibull(outcome_structure, outcome_concentration)
                with mask(mask = (cens == 1)):
                    pyro.sample(Sites.y, weibull, obs = y)
                with mask(mask = (cens == 0)):
                    survival_prob = 1 - weibull.cdf(y)
                    pyro.sample(Sites.censored, dist.Bernoulli(survival_prob), obs = 1)
            else:
                raise NotImplementedError
    
    def forward_dense(self, 
                X_list, 
                batch_idx,
                y = None):
        """
        X_l = Z @ Lambda_l^T + Phi_l @ Gamma_l^T + E,   l = 1, ... , L
        y = Z @ beta + e
        """
        m = len(batch_idx)
        p_l_list = [X_l.shape[1] for X_l in X_list]
        # print('p_l', p_l_list)
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
            # psi_l = pyro.sample(f"psi_l{l}", dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_l_inv = pyro.sample(f"psi_l{l}", dist.Gamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_l = psi_l_inv.pow_(-1)
            
            # print(f"psi_l{l} min: {psi_l.min().item()}, max: {psi_l.max().item()}")
            # print(f"psi_l{l} has NaN: {torch.isnan(psi_l).any()}")
            # print(f"psi_l{l} has negative: {(psi_l < 0).any()}")
            
            psi_sqrt = torch.sqrt(psi_l)
            
            # print(f"psi_sqrt_l{l} has NaN: {torch.isnan(psi_sqrt).any()}")
            
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
                
                X_l = pyro.sample(f"X_l{l}", dist.Normal(total_structure_l, 
                                                   psi_sqrt_l_list[l]).to_event(1), 
                            obs = X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                    print(torch.where(torch.isnan(X_l)))
                    print(X_l)
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")
                
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # Outcome model
            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic("outcome_structure",
                                                    torch.matmul(full_factors, beta))
                # pyro.sample("y", dist.Normal(outcome_structure, 
                #                             sigma_y), #.to_event(1),
                #             obs = y)
            else:
                outcome_structure = pyro.deterministic("outcome_structure",
                                                    torch.matmul(Z, beta))
                # pyro.sample("y", dist.Normal(outcome_structure, 
                #                             sigma_y), #.to_event(1),
                #             obs = y)
            if self.outcome == "gaussian":
                pyro.sample("y", dist.Normal(outcome_structure, 
                                            sigma_y), #.to_event(1),
                            obs = y)
            else:
                raise NotImplementedError
            # pyro.sample("y", dist.Normal(outcome_structure.index_select(0, batch_idx), 
            #                              sigma_y),
            #             obs = y.index_select(0, batch_idx))
            
            
    def guide(self, 
              X_list, 
              batch_idx,
              y = None,
              cens = None):
        if self.dense:
            raise NotImplementedError
        
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            for m in range(k_l):
                a_delta_gamma_l_k = pyro.param(Params.a_delta_gamma_l_k.format(l = l, m = m), 
                                               torch.tensor(self.a1_sigma_view),
                                               constraint = dist.constraints.positive)
                b_delta_gamma_l_k = pyro.param(Params.b_delta_gamma_l_k.format(l = l, m = m), 
                                               torch.tensor(1.0),
                                               constraint = dist.constraints.positive)
                pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), 
                            dist.Gamma(a_delta_gamma_l_k, b_delta_gamma_l_k))
                
            a_rho_gamma_l = pyro.param(Params.a_rho_gamma_l.format(l = l), 
                                            torch.tensor(self.alpha_individual / 2),
                                            constraint = dist.constraints.positive)
            b_rho_gamma_l = pyro.param(Params.b_rho_gamma_l.format(l = l), 
                                            torch.tensor(self.alpha_individual / 2),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.rho_gamma_l.format(l = l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).expand([p_l, k_l]).to_event(2)).squeeze()

            loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), torch.zeros(p_l, k_l))
            scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l = l), torch.ones(p_l, k_l),
                                       constraint = dist.constraints.positive)
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            for m in range(self.k):
                a_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                                                torch.tensor(self.a1_sigma_joint),
                                                constraint = dist.constraints.positive)
                b_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                                                torch.tensor(1.0),
                                                constraint = dist.constraints.positive)
                pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

            a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
                                            torch.tensor(self.alpha_joint / 2),
                                            constraint = dist.constraints.positive)
            b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
                                            torch.tensor(self.alpha_joint / 2),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.rho_lambda_l.format(l = l), 
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()
            
            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(p_l, self.k))
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(p_l, self.k),
                                        constraint = dist.constraints.positive)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l = l), 
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            
            Lambda_l_list.append(Lambda_l)
        
        
        ########################
        # ---- Observations --------
        # Outcome y can be dependent on just shared factors, or all factors
        ########################
        
        # Idiosyncratic error variance 
        psi_sqrt_l_list = []           
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = pyro.param(Params.a_psi_l.format(l = l), 
                                 torch.tensor(self.a_psi),
                                 constraint = dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l = l), 
                                 torch.tensor(self.b_psi),
                                 constraint = dist.constraints.positive)
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            # psi_sqrt = torch.sqrt(psi_l)
            # psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                  torch.tensor(self.a_sigma_beta),
                                  constraint = dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                  torch.tensor(self.b_sigma_beta),
                                  constraint = dist.constraints.positive)
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        
        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        scale_beta = pyro.param(Params.scale_beta, torch.ones(self.num_outcome_factors),
                                constraint = dist.constraints.positive)
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y, 
                                   torch.tensor(self.a_sigma_y),
                                   constraint = dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y, 
                                   torch.tensor(self.b_sigma_y),
                                   constraint = dist.constraints.positive)
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            # sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        else:
            # outcome_concentration = pyro.sample("")
            raise NotImplementedError
        
        # Local latent variables and observations
        loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        scale_Z = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
                                constraint = dist.constraints.positive)
        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l), torch.zeros(self.n, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l = l), torch.ones(self.n, k_l),
                                     constraint = dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)
            
        with pyro.plate("obs", self.n, subsample = batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            # Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                # loc_Z
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(loc_Phi, scale_Phi).to_event(1))
                # Phi_l_list.append(Phi_l)
                
            
            # # Compute structures
            # joint_structure_list = []
            # for l in range(len(X_list)):
            #     joint_structure_l = pyro.deterministic(f"joint_structure_l{l}", 
            #                                            torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
            #     view_structure_l = pyro.deterministic(f"view_structure_l{l}",
            #                                         torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
            #     total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
            #     joint_structure_list.append(joint_structure_l)
                
            #     X_l = pyro.sample(f"X_l{l}", dist.Normal(total_structure_l, 
            #                                        psi_sqrt_l_list[l]).to_event(1), 
            #                 obs = X_list[l])
            #     if torch.isnan(X_l).any():
            #         print("NaN values found in X_l_tensor!")
            #     if torch.isinf(X_l).any():
            #         print("Inf values found in X_l_tensor!")
            
            #     # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
            #     #                                    psi_sqrt_l_list[l]).to_event(1), 
            #     #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # # Outcome model
            # if self.include_view_factors:
            #     full_factors = torch.cat((Z, *Phi_l_list), 1)
            #     outcome_structure = pyro.deterministic("outcome_structure",
            #                                         torch.matmul(full_factors, beta))
            # else:
            #     outcome_structure = pyro.deterministic("outcome_structure",
            #                                         torch.matmul(Z, beta))
            #     # pyro.sample("y", dist.Normal(outcome_structure, 
            #     #                             sigma_y), #.to_event(1),
            #     #             obs = y)
            
            # if self.outcome == "gaussian":
            #     pyro.sample("y", dist.Normal(outcome_structure, 
            #                                 sigma_y), #.to_event(1),
            #                 obs = y)
            # elif self.outcome == "censored":
            #     raise NotImplementedError
            #     # weibull = dist.Weibull(outcome_structure, outcome_concentration)
            #     # with mask(mask = (cens == 1)):
            #     #     pyro.sample("y", weibull, obs = y)
            #     # with mask(mask = (cens == 0)):
            #     #     survival_prob = 1 - weibull.cdf(y)
            #     #     pyro.sample("censored", dist.Bernoulli(survival_prob), obs = 1)
            # else:
            #     raise NotImplementedError
            
    def predict_guide(self, 
              X_list, 
              batch_idx,
              y = None,
              cens = None):
        if self.dense:
            raise NotImplementedError
        
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            for m in range(k_l):
                # a_delta_gamma_l_m = pyro.param(Params.a_delta_gamma_l_k.format(l = l, m = m), 
                #                                torch.tensor(self.a1_sigma_view),
                #                                constraint = dist.constraints.positive)
                # b_delta_gamma_l_m = pyro.param(Params.b_delta_gamma_l_k.format(l = l, m = m), 
                #                                torch.tensor(1.0),
                #                                constraint = dist.constraints.positive)
                a_delta_gamma_l_k = self.params[Params.a_delta_gamma_l_k.format(l = l, m = m)]
                b_delta_gamma_l_k = self.params[Params.b_delta_gamma_l_k.format(l = l, m = m)]
                pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), dist.Gamma(a_delta_gamma_l_k, b_delta_gamma_l_k))
                
            # a_rho_gamma_l = pyro.param(Params.a_rho_gamma_l.format(l = l), 
            #                                 torch.tensor(self.alpha_individual / 2),
            #                                 constraint = dist.constraints.positive)
            # b_rho_gamma_l = pyro.param(Params.b_rho_gamma_l.format(l = l), 
            #                                 torch.tensor(self.alpha_individual / 2),
            #                                 constraint = dist.constraints.positive)
            a_rho_gamma_l = self.params[Params.a_rho_gamma_l]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l]
            pyro.sample(Sites.rho_gamma_l.format(l = l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).expand([p_l, k_l]).to_event(2)).squeeze()

            # loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), torch.zeros(p_l, k_l))
            # scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l = l), torch.ones(p_l, k_l),
            #                            constraint = dist.constraints.positive)
            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l = l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l = l)]
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            for m in range(self.k):
                # a_delta_lambda_l_m = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                #                                 torch.tensor(self.a1_sigma_joint),
                #                                 constraint = dist.constraints.positive)
                # b_delta_lambda_l_m = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                #                                 torch.tensor(1.0),
                #                                 constraint = dist.constraints.positive)
                a_delta_lambda_l_k = self.params[Params.a_delta_lambda_l_k.format(l = l, m = m)]
                b_delta_lambda_l_k = self.params[Params.b_delta_lambda_l_k.format(l = l, m = m)]
                pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

            # a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
            #                                 torch.tensor(self.alpha_joint / 2),
            #                                 constraint = dist.constraints.positive)
            # b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
            #                                 torch.tensor(self.alpha_joint / 2),
            #                                 constraint = dist.constraints.positive)
            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l = l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l = l)]
            pyro.sample(Sites.rho_lambda_l.format(l = l), 
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()
            
            # loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(p_l, self.k))
            # scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(p_l, self.k),
            #                             constraint = dist.constraints.positive)
            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l = l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l = l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l = l), 
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            
            Lambda_l_list.append(Lambda_l)
        
        
        ########################
        # ---- Observations --------
        # Outcome y can be dependent on just shared factors, or all factors
        ########################
        
        # Idiosyncratic error variance 
        psi_sqrt_l_list = []           
        for l, p_l in enumerate(self.p_l_list):
            # a_psi_l = pyro.param(Params.a_psi_l.format(l = l), 
            #                      torch.tensor(self.a_psi),
            #                      constraint = dist.constraints.positive)
            # b_psi_l = pyro.param(Params.b_psi_l.format(l = l), 
            #                      torch.tensor(self.b_psi),
            #                      constraint = dist.constraints.positive)
            a_psi_l = self.params[Params.a_psi_l.format(l = l)]
            b_psi_l = self.params[Params.b_psi_l.format(l = l)]
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            # psi_sqrt = torch.sqrt(psi_l)
            # psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        # a_sigma_beta = pyro.param(Params.a_sigma_beta, 
        #                           torch.tensor(self.a_sigma_beta),
        #                           constraint = dist.constraints.positive)
        # b_sigma_beta = pyro.param(Params.b_sigma_beta, 
        #                           torch.tensor(self.b_sigma_beta),
        #                           constraint = dist.constraints.positive)
        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]        
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        
        # loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        # scale_beta = pyro.param(Params.scale_beta, torch.ones(self.num_outcome_factors),
        #                         constraint = dist.constraints.positive)
        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        if self.outcome == "gaussian":
            # a_sigma_y = pyro.param(Params.a_sigma_y, 
            #                        torch.tensor(self.a_sigma_y),
            #                        constraint = dist.constraints.positive)
            # b_sigma_y = pyro.param(Params.b_sigma_y, 
            #                        torch.tensor(self.b_sigma_y),
            #                        constraint = dist.constraints.positive)
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            # sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        else:
            # outcome_concentration = pyro.sample("")
            raise NotImplementedError
        
        # Local latent variables and observations
        loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        scale_Z = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
                                constraint = dist.constraints.positive)
        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l), torch.zeros(self.n, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l = l), torch.ones(self.n, k_l),
                                     constraint = dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)
            
        with pyro.plate("obs", self.n, subsample = batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            # Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                # loc_Z
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(loc_Phi, scale_Phi).to_event(1))
##############
