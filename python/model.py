import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
# from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal
from pyro.poutine import mask

from constants import Sites, Params
from data_utils import align_tensor_shapes

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class SupMultiviewDecomp(PyroModule):
    def __init__(self, 
                 k, 
                 k_l_list,
                 ortho_penalty = 0.0,
                #  n,
                 include_view_factors = False,
                 dense = True,
                 outcome = "gaussian",
                 penalty_obj = None,
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
                 a_psi=3.0, # Hyperparams: per-view error: IG same for all views
                 b_psi=1.0,
                 a_sigma_y = 3.0, # Hyperparams: outcome error: IG
                 b_sigma_y = 1.0,
                 a_sigma_beta = 2.0, # Hyperparams: outcome coefficients: IG
                 b_sigma_beta = 2.0,
                 a_weibull = 2.0, # Hyperparams: Weibull shape prior: Gamma
                 b_weibull = 1.0,
                 joint_scores_init = None,
                 joint_loadings_list_init = None,
                 view_scores_list_init = None,
                 view_loadings_list_init = None
                 ):
        super().__init__()
        
        self.model_type = "SupMultiviewDecomp"
        # Model parameters
        self.n = None
        self.n_predict = None
        self.k = k
        self.k_l_list = k_l_list # assumes k_l > 0 for all l
        # self.p_l = None
        self.p_l_list = None
        
        self.ortho_penalty = ortho_penalty
        self.penalty_obj = penalty_obj
        
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
        self.a_weibull = a_weibull
        self.b_weibull = b_weibull
        
        # For testing: Can initialize location params of scores and loadings with given values
        self.joint_scores_init = joint_scores_init
        self.joint_loadings_list_init = joint_loadings_list_init
        self.view_scores_list_init = view_scores_list_init
        self.view_loadings_list_init = view_loadings_list_init
        
        # Should view-specific factors be included in the outcome model?
        self.include_view_factors = include_view_factors
        self.num_outcome_factors = k if not include_view_factors else sum((k, *k_l_list))
        
        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []
        
        self.local_epochs = 0
        self.local_total_iters = None
        self.local_loss_history = []
        self.local_var_param_convergence_history = []
        
        self.params = None
        
        self.inv_gamma_init_param = 2.1
        self.init_param = 0.1
    
        
    def forward(self,
                X_list,
                batch_idx,
                y = None,
                cens = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.forward_dense(X_list, batch_idx, y, cens)
        else:
            return self.forward_mgp(X_list, batch_idx, y, cens)

    def guide(self,
                X_list,
                batch_idx,
                y = None,
                cens = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.guide_dense(X_list, batch_idx, y, cens)
        else:
            return self.guide_mgp(X_list, batch_idx, y, cens)
        
        
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
        # n = X_list[0].shape[0]
        # if self.n is None:
        #     self.n = n
        
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
        elif self.outcome == "censored":
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(self.a_weibull, self.b_weibull))
        else:
            raise NotImplementedError

        # orthogonality_weight = pyro.sample("ortho_weight", dist.InverseGamma(1, 1))
        # orthogonality_weight = 100.
        
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
            view_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l.format(l = l), 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l.format(l = l),
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                view_structure_list.append(view_structure_l)
                
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
            
            # Orthogonality in structures
            # joint_structure_full = torch.cat(joint_structure_list, 1)
            # Calculate penalty terms
            # total_penalty = torch.tensor(0.0)
            # for l in range(len(X_list)):
            #     cross_prod = view_structure_list[l].T @ joint_structure_full
            #     total_penalty += torch.sum(cross_prod ** 2)
                
            # pyro.factor("orthogonality", 
            #             -orthogonality_weight * total_penalty)
            
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
                y_pred = pyro.sample(Sites.y, dist.Normal(outcome_structure,
                                            sigma_y),
                            obs = y)
                return y_pred
            elif self.outcome == "censored":
                scale = torch.exp(outcome_structure).clamp(min=1e-10)
                with mask(mask=(cens == 1)):
                    pyro.sample(Sites.y, dist.Weibull(scale, weibull_concentration), obs=y)
                log_surv = -torch.pow(y / scale, weibull_concentration)
                with mask(mask=(cens == 0)):
                    pyro.factor(Sites.censored, log_surv)
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
        # n = X_list[0].shape[0]
        # if self.n is None:
        #     self.n = n
        
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
            a_rho_gamma_l = self.params[Params.a_rho_gamma_l.format(l = l)]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l.format(l = l)]
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
        if self.outcome == "gaussian":
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError
        
        # Local latent variables and observations
        with pyro.plate("obs_pred", self.n_predict, subsample = batch_idx):
            Z = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(Sites.Phi_l_pred.format(l = l), 
                                    dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)
                
            
            # Compute structures
            joint_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l = l), 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l_pred.format(l = l),
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                
                X_l = pyro.sample(Sites.X_l_pred.format(l = l), 
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
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                    torch.matmul(full_factors, beta))
            else:
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                    torch.matmul(Z, beta))
                # pyro.sample("y", dist.Normal(outcome_structure, 
                #                             sigma_y), #.to_event(1),
                #             obs = y)
            
            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y),
                            obs = y)
                return y_pred
            elif self.outcome == "censored":
                scale = torch.exp(outcome_structure).clamp(min=1e-10)
                if cens is not None:
                    with mask(mask=(cens == 1)):
                        pyro.sample(Sites.y_pred, dist.Weibull(scale, weibull_concentration), obs=y)
                    log_surv = -torch.pow(y / scale, weibull_concentration)
                    with mask(mask=(cens == 0)):
                        pyro.factor(Sites.censored, log_surv)
                else:
                    y_pred = pyro.sample(Sites.y_pred, dist.Weibull(scale, weibull_concentration), obs=y)
                    return y_pred
            else:
                raise NotImplementedError

    def forward_dense(self,
                X_list,
                batch_idx,
                y = None,
                cens = None):
        """
        X_l = Z @ Lambda_l^T + Phi_l @ Gamma_l^T + E,   l = 1, ... , L
        y = Z @ beta + e
        """
        m = len(batch_idx)
        p_l_list = [X_l.shape[1] for X_l in X_list]
        n = X_list[0].shape[0]
        if self.n is None:
            self.n = n

        
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
            sigma2_gamma_l = pyro.sample(Sites.sigma2_gamma_l.format(l = l),
                                   dist.InverseGamma(self.a_sigma_view, self.b_sigma_view).expand([k_l]).to_event(1))
            sigma_gamma_l = torch.sqrt(sigma2_gamma_l)
        
            # sigma is broadcast across rows of Lambda
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), dist.Normal(torch.zeros(p_l, k_l), sigma_gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED
        Lambda_l_list = []
        for l, p_l in enumerate(p_l_list):
            sigma2_lambda_l = pyro.sample(Sites.sigma2_lambda_l.format(l = l),
                                   dist.InverseGamma(self.a_sigma_joint, self.b_sigma_joint).expand([self.k]).to_event(1))
            sigma_lambda_l = torch.sqrt(sigma2_lambda_l)
        
            # sigma is broadcast across rows of Lambda
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l = l), dist.Normal(torch.zeros(p_l, self.k), sigma_lambda_l).to_event(2))
            
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
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                    dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            # psi_l = psi_l_inv.pow_(-1)
            
            # print(f"psi_l{l} min: {psi_l.min().item()}, max: {psi_l.max().item()}")
            # print(f"psi_l{l} has NaN: {torch.isnan(psi_l).any()}")
            # print(f"psi_l{l} has negative: {(psi_l < 0).any()}")
            
            psi_sqrt = torch.sqrt(psi_l)
            
            # print(f"psi_sqrt_l{l} has NaN: {torch.isnan(psi_sqrt).any()}")
            
            psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.num_outcome_factors), sigma2_beta).to_event(1)).\
                               squeeze(0)
        
        # Outcome variances
        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(self.a_weibull, self.b_weibull))
        else:
            raise NotImplementedError
        
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
                pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y), obs=y)
            elif self.outcome == "censored":
                scale = torch.exp(outcome_structure).clamp(min=1e-10)
                with mask(mask=(cens == 1)):
                    pyro.sample(Sites.y, dist.Weibull(scale, weibull_concentration), obs=y)
                log_surv = -torch.pow(y / scale, weibull_concentration)
                with mask(mask=(cens == 0)):
                    pyro.factor(Sites.censored, log_surv)
            else:
                raise NotImplementedError
            # pyro.sample("y", dist.Normal(outcome_structure.index_select(0, batch_idx),
            #                              sigma_y),
            #             obs = y.index_select(0, batch_idx))
            
            
    def guide_mgp(self, 
              X_list, 
              batch_idx,
              y = None,
              cens = None):
        if self.dense:
            raise NotImplementedError
        
        # def projection_by_qr(tens):
        #     Q, _ = torch.linalg.qr(tens)
        #     return Q @ Q.T
        
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        # n = X_list[0].shape[0]
        
        ########################
        # ---- Loadings --------
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            for m in range(k_l):
                a_delta_gamma_l_k = pyro.param(Params.a_delta_gamma_l_k.format(l = l, m = m), 
                                            #    torch.tensor(self.a1_sigma_view),
                                               torch.tensor(self.init_param),
                                               constraint = dist.constraints.positive)
                b_delta_gamma_l_k = pyro.param(Params.b_delta_gamma_l_k.format(l = l, m = m), 
                                               torch.tensor(self.init_param),
                                               constraint = dist.constraints.positive)
                pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), 
                            dist.Gamma(a_delta_gamma_l_k, b_delta_gamma_l_k))
                # loc_delta_gamma_l_k = pyro.param(Params.a_delta_gamma_l_k.format(l = l, m = m), 
                #                                torch.tensor(0.0))
                # scale_delta_gamma_l_k = pyro.param(Params.b_delta_gamma_l_k.format(l = l, m = m), 
                #                                torch.tensor(1.0),
                #                                constraint = dist.constraints.positive)
                # pyro.sample(Sites.delta_gamma_l_k.format(l = l, m = m), 
                #             dist.LogNormal(loc_delta_gamma_l_k, scale_delta_gamma_l_k))
                
                
            a_rho_gamma_l = pyro.param(Params.a_rho_gamma_l.format(l = l), 
                                            # torch.tensor(self.alpha_individual / 2),
                                            torch.tensor(self.init_param),
                                            constraint = dist.constraints.positive)
            b_rho_gamma_l = pyro.param(Params.b_rho_gamma_l.format(l = l), 
                                            # torch.tensor(self.alpha_individual / 2),
                                            torch.tensor(self.init_param),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.rho_gamma_l.format(l = l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).expand([p_l, k_l]).to_event(2)).squeeze()
            # loc_rho_gamma_l = pyro.param(Params.a_rho_gamma_l.format(l = l), 
            #                                 torch.tensor(0.0))
            # scale_rho_gamma_l = pyro.param(Params.b_rho_gamma_l.format(l = l), 
            #                                 torch.tensor(1.0),
            #                                 constraint = dist.constraints.positive)
            # pyro.sample(Sites.rho_gamma_l.format(l = l),
            #             dist.LogNormal(loc_rho_gamma_l, scale_rho_gamma_l).expand([p_l, k_l]).to_event(2)).squeeze()

            if self.view_loadings_list_init is None:
                loc_Gamma_l = align_tensor_shapes(self.view_loadings_list_init[l], 
                                                additional = k_l - self.view_loadings_list_init[l].shape[1])
                loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), 
                                         loc_Gamma_l
                                         )
            else:
                loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), torch.zeros(p_l, k_l))
            scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l = l), 
                                    #    torch.ones(p_l, k_l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint = dist.constraints.positive)
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            for m in range(self.k):
                a_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                                                # torch.tensor(self.a1_sigma_joint),
                                                torch.tensor(self.init_param),
                                                constraint = dist.constraints.positive)
                b_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                                                torch.tensor(self.init_param),
                                                constraint = dist.constraints.positive)
                pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))
                # loc_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                #                                 torch.tensor(0.0))
                # scale_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                #                                 torch.tensor(1.0),
                #                                 constraint = dist.constraints.positive)
                # pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                #             dist.LogNormal(loc_delta_lambda_l_k, scale_delta_lambda_l_k))

            a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
                                            # torch.tensor(self.alpha_joint / 2),
                                            torch.tensor(self.init_param),
                                            constraint = dist.constraints.positive)
            b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
                                            # torch.tensor(self.alpha_joint / 2),
                                            torch.tensor(self.init_param),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.rho_lambda_l.format(l = l), 
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()
            # loc_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
            #                                 torch.tensor(0.0))
            # scale_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
            #                                 torch.tensor(1.0),
            #                                 # torch.tensor(1.0),
            #                                 constraint = dist.constraints.positive)
            # pyro.sample(Sites.rho_lambda_l.format(l = l), 
            #             dist.LogNormal(loc_rho_lambda_l, scale_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()
            
            if self.joint_loadings_list_init is None:
                loc_Lambda_l = align_tensor_shapes(self.joint_loadings_list_init[l], 
                                                additional = self.k - self.joint_loadings_list_init[l].shape[1])
                loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), 
                                          loc_Lambda_l
                                         )
            else:
                loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(p_l, self.k))
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), 
                                        # torch.ones(p_l, self.k),
                                        torch.full((p_l, self.k), self.init_param),
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
                                #  torch.tensor(self.a_psi),
                                 torch.tensor(self.inv_gamma_init_param),
                                 constraint = dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l = l), 
                                #  torch.tensor(self.b_psi),
                                torch.tensor(self.init_param),
                                 constraint = dist.constraints.positive)
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                # dist.Gamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            # psi_sqrt = torch.sqrt(psi_l)
            # psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                #   torch.tensor(2.1),
                                  torch.tensor(self.inv_gamma_init_param),
                                #   torch.tensor(self.a_sigma_beta),
                                  constraint = dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                #   torch.tensor(self.b_sigma_beta),
                                #   torch.tensor(0.1),
                                  torch.tensor(self.init_param),
                                  constraint = dist.constraints.positive)
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        
        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        scale_beta = pyro.param(Params.scale_beta, 
                                # torch.ones(self.num_outcome_factors),
                                torch.full((self.num_outcome_factors,), self.init_param),
                                constraint = dist.constraints.positive)
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                #    torch.tensor(self.a_sigma_y),
                                   torch.tensor(self.inv_gamma_init_param),
                                #    torch.tensor(2.1),
                                #    torch.tensor(2.5),
                                   constraint = dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                #    torch.tensor(self.b_sigma_y),
                                #    torch.tensor(0.1),
                                   torch.tensor(self.init_param),
                                   constraint = dist.constraints.positive)
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        elif self.outcome == "censored":
            a_weibull_c = pyro.param(Params.a_weibull_concentration,
                                     torch.tensor(self.a_weibull),
                                     constraint=dist.constraints.positive)
            b_weibull_c = pyro.param(Params.b_weibull_concentration,
                                     torch.tensor(self.b_weibull),
                                     constraint=dist.constraints.positive)
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        # Local latent variables and observations
        if self.joint_scores_init is None:
            loc_Z = align_tensor_shapes(self.joint_scores_init,
                                        additional = self.k - self.joint_scores_init)
            loc_Z = pyro.param(Params.loc_Z, 
                               loc_Z
            )
        else:
            loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        # scale_Z = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
        #                         constraint = dist.constraints.positive)
        scale_Z = pyro.param(Params.scale_Z, 
                             torch.full((self.n, self.k), self.init_param),
                             constraint = dist.constraints.positive)
        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            if self.view_scores_list_init is None:
                loc_Phi_l =  align_tensor_shapes(self.view_scores_list_init[l],
                                                additional = k_l - self.view_scores_list_init[l])
                loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l), 
                                      loc_Phi_l
                                       )
            else:
                loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l), torch.zeros(self.n, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l = l), 
                                    #  torch.ones(self.n, k_l),
                                     torch.full((self.n, k_l), self.init_param),
                                     constraint = dist.constraints.greater_than_eq(0.0)) #dist.constraints.positive
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)
        
        # PROJECTION OF VIEW-SPECIFIC STRUCTURE THROUGH SCORES
        # print('loc_z at beginning')
        # print(loc_Z)
         #@ torch.cat([Lambda.T for Lambda in Lambda_l_list],
                                        #   dim = 1)
        # with torch.no_grad():
        #     A = loc_Z
        #     P_A = projection_by_qr(A) # TODO make below faster
        #     P_A_perp = torch.sub(torch.eye(P_A.shape[0]), P_A)
        
        # # print(P_A_perp)
        # # Means
        # proj_locs_Phi_l_list = [torch.mm(P_A_perp, Phi_l) \
        #     for Phi_l in loc_Phi_list]
        # # Variances
        # proj_scales_Phi_l_list = []
        # for vars in scale_Phi_list:
        #     proj_scales = torch.stack(
        #         [torch.diag(P_A_perp * vars[:, k] @ P_A_perp.T) \
        #             for k in range(vars.shape[1])],
        #         dim = 1
        #     ) + 1e-3
        #     proj_scales_Phi_l_list.append(proj_scales)
            
            
            
            
        # ortho_a = pyro.param("ortho_a",
        #                         torch.tensor(1.),
        #                         constraint = dist.constraints.positive)
        # ortho_b = pyro.param("ortho_b", 
        #                         torch.tensor(1.),
        #                         constraint = dist.constraints.positive)
        # orthogonality_weight = pyro.sample("ortho_weight", 
        #                                     dist.InverseGamma(ortho_a, ortho_b))
        # orthogonality_weight = 10. #100.
            
        with pyro.plate("obs", self.n, subsample = batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            
            # print('loc_Z')
            # print(loc_Z_batch)
            # print('scale_Z')
            # print(scale_Z_batch)
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            # print('sampled_Z')
            # print(Z)
            # print(Z)
            
            loc_Phi_l_list = []
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                # loc_Z
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                
                # loc_Phi = proj_locs_Phi_l_list[l][batch_idx]
                # scale_Phi = proj_scales_Phi_l_list[l][batch_idx]
                
                # print('loc_Phi')
                # print(loc_Phi)
                # print(scale_Phi)
                
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(loc_Phi, scale_Phi).to_event(1))
                # print('sampled_Phi')
                # print(Phi_l)
                Phi_l_list.append(Phi_l)
                loc_Phi_l_list.append(loc_Phi)
                
            
            # Compute structures
            joint_structure_list = []
            view_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(f"joint_structure_l{l}", 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(f"view_structure_l{l}",
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                view_structure_list.append(view_structure_l)
                
            joint_structure_full = torch.cat(joint_structure_list, 1)
                
            # Calculate penalty terms
            # if (self.ortho_penalty != 0 and self.penalty_obj is None) or \
            #     (self.penalty_obj is not None and self.penalty_obj.weight != 0):
            #     total_penalty = torch.tensor(0.0)
            #     for l in range(len(X_list)):
            #         # cross_prod = view_structure_list[l].T @ joint_structure_full
            #         cross_prod = Phi_l_list[l].T @ Z
            #         # cross_prod = loc_Phi_l_list[l].T @ loc_Z_batch
                    
            #         total_penalty += torch.sum(cross_prod ** 2)
            #     if self.penalty_obj is None:
            #         pyro.factor("orthogonality", 
            #                     self.ortho_penalty * total_penalty * (len(batch_idx) / self.n),
            #                     has_rsample = True)
            #     else:
            #         pyro.factor("orthogonality", 
            #                 self.penalty_obj.weight * total_penalty * (len(batch_idx) / self.n),
            #                 has_rsample = True)
                
                # X_l = pyro.sample(f"X_l{l}", dist.Normal(total_structure_l, 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l])
                # if torch.isnan(X_l).any():
                #     print("NaN values found in X_l_tensor!")
                # if torch.isinf(X_l).any():
                #     print("Inf values found in X_l_tensor!")
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
            # Outcome model
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
            
            
    def guide_dense(self, 
              X_list, 
              batch_idx,
              y = None,
              cens = None):
        
        # def projection_by_qr(tens):
        #     Q, _ = torch.linalg.qr(tens)
        #     return Q @ Q.T
        
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        # n = X_list[0].shape[0]
        
        ########################
        # ---- Loadings --------
        ########################
        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            
            a_sigma_gamma = pyro.param(Params.a_sigma_gamma_l.format(l = l), 
                                       torch.tensor(self.inv_gamma_init_param))
            b_sigma_gamma = pyro.param(Params.b_sigma_gamma_l.format(l = l),
                                       torch.tensor(self.inv_gamma_init_param))
            sigma2_gamma_l = pyro.sample(Sites.sigma2_gamma_l.format(l = l),
                                        dist.InverseGamma(a_sigma_gamma, b_sigma_gamma).expand([k_l]).to_event(1))
            
            if self.view_loadings_list_init is None:
                loc_Gamma_l = align_tensor_shapes(self.view_loadings_list_init[l], 
                                                additional = k_l - self.view_loadings_list_init[l].shape[1])
                loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), 
                                         loc_Gamma_l
                                         )
            else:
                loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l = l), torch.zeros(p_l, k_l))
            scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l = l), 
                                    #    torch.ones(p_l, k_l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint = dist.constraints.positive)
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l = l), 
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            
            Gamma_l_list.append(Gamma_l)
        
        # SHARED        
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            
            a_sigma_lambda = pyro.param(Params.a_sigma_lambda_l.format(l = l), 
                                       torch.tensor(self.inv_gamma_init_param))
            b_sigma_lambda = pyro.param(Params.b_sigma_lambda_l.format(l = l),
                                       torch.tensor(self.inv_gamma_init_param))
            sigma2_lambda_l = pyro.sample(Sites.sigma2_lambda_l.format(l = l),
                                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).expand([self.k]).to_event(1))
            
            if self.joint_loadings_list_init is None:
                loc_Lambda_l = align_tensor_shapes(self.joint_loadings_list_init[l], 
                                                additional = self.k - self.joint_loadings_list_init[l].shape[1])
                loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), 
                                          loc_Lambda_l
                                         )
            else:
                loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(p_l, self.k))
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), 
                                        # torch.ones(p_l, self.k),
                                        torch.full((p_l, self.k), self.init_param),
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
                                #  torch.tensor(self.a_psi),
                                 torch.tensor(self.inv_gamma_init_param),
                                 constraint = dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l = l), 
                                #  torch.tensor(self.b_psi),
                                torch.tensor(self.init_param),
                                 constraint = dist.constraints.positive)
            psi_l = pyro.sample(Sites.psi_l.format(l = l), 
                                # dist.Gamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            # psi_sqrt = torch.sqrt(psi_l)
            # psi_sqrt_l_list.append(psi_sqrt)
            
        
        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta, 
                                #   torch.tensor(2.1),
                                  torch.tensor(self.inv_gamma_init_param),
                                #   torch.tensor(self.a_sigma_beta),
                                  constraint = dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta, 
                                #   torch.tensor(self.b_sigma_beta),
                                #   torch.tensor(0.1),
                                  torch.tensor(self.init_param),
                                  constraint = dist.constraints.positive)
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.num_outcome_factors]).to_event(1))
        
        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        scale_beta = pyro.param(Params.scale_beta, 
                                # torch.ones(self.num_outcome_factors),
                                torch.full((self.num_outcome_factors,), self.init_param),
                                constraint = dist.constraints.positive)
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).\
                               squeeze(0)
        
        ## Outcome error distribution parameters:
        # Continuous: Normal variances
        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                #    torch.tensor(self.a_sigma_y),
                                   torch.tensor(self.inv_gamma_init_param),
                                #    torch.tensor(2.1),
                                #    torch.tensor(2.5),
                                   constraint = dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                #    torch.tensor(self.b_sigma_y),
                                #    torch.tensor(0.1),
                                   torch.tensor(self.init_param),
                                   constraint = dist.constraints.positive)
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            # sigma_y = torch.sqrt(sigma2_y)
        # Right-censored: Weibull shape (Gamma)
        elif self.outcome == "censored":
            a_weibull_c = pyro.param(Params.a_weibull_concentration,
                                     torch.tensor(self.a_weibull),
                                     constraint=dist.constraints.positive)
            b_weibull_c = pyro.param(Params.b_weibull_concentration,
                                     torch.tensor(self.b_weibull),
                                     constraint=dist.constraints.positive)
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        # Local latent variables and observations
        if self.joint_scores_init is None:
            loc_Z = align_tensor_shapes(self.joint_scores_init,
                                        additional = self.k - self.joint_scores_init)
            loc_Z = pyro.param(Params.loc_Z,
                               loc_Z
            )
        else:
            loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        # scale_Z = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
        #                         constraint = dist.constraints.positive)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.full((self.n, self.k), self.init_param),
                             constraint = dist.constraints.positive)
        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            if self.view_scores_list_init is None:
                loc_Phi_l =  align_tensor_shapes(self.view_scores_list_init[l],
                                                additional = k_l - self.view_scores_list_init[l])
                loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l),
                                      loc_Phi_l
                                       )
            else:
                loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l = l), torch.zeros(self.n, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l = l),
                                    #  torch.ones(self.n, k_l),
                                     torch.full((self.n, k_l), self.init_param),
                                     constraint = dist.constraints.greater_than_eq(0.0)) #dist.constraints.positive
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)

        # PROJECTION OF VIEW-SPECIFIC STRUCTURE THROUGH SCORES
        # print('loc_z at beginning')
        # print(loc_Z)
         #@ torch.cat([Lambda.T for Lambda in Lambda_l_list],
                                        #   dim = 1)
        # with torch.no_grad():
        #     A = loc_Z
        #     P_A = projection_by_qr(A) # TODO make below faster
        #     P_A_perp = torch.sub(torch.eye(P_A.shape[0]), P_A)
        
        # # print(P_A_perp)
        # # Means
        # proj_locs_Phi_l_list = [torch.mm(P_A_perp, Phi_l) \
        #     for Phi_l in loc_Phi_list]
        # # Variances
        # proj_scales_Phi_l_list = []
        # for vars in scale_Phi_list:
        #     proj_scales = torch.stack(
        #         [torch.diag(P_A_perp * vars[:, k] @ P_A_perp.T) \
        #             for k in range(vars.shape[1])],
        #         dim = 1
        #     ) + 1e-3
        #     proj_scales_Phi_l_list.append(proj_scales)
            
            
            
            
        # ortho_a = pyro.param("ortho_a",
        #                         torch.tensor(1.),
        #                         constraint = dist.constraints.positive)
        # ortho_b = pyro.param("ortho_b", 
        #                         torch.tensor(1.),
        #                         constraint = dist.constraints.positive)
        # orthogonality_weight = pyro.sample("ortho_weight", 
        #                                     dist.InverseGamma(ortho_a, ortho_b))
        # orthogonality_weight = 10. #100.
            
        with pyro.plate("obs", self.n, subsample = batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            
            # print('loc_Z')
            # print(loc_Z_batch)
            # print('scale_Z')
            # print(scale_Z_batch)
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            # print('sampled_Z')
            # print(Z)
            # print(Z)
            
            loc_Phi_l_list = []
            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                # loc_Z
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                
                # loc_Phi = proj_locs_Phi_l_list[l][batch_idx]
                # scale_Phi = proj_scales_Phi_l_list[l][batch_idx]
                
                # print('loc_Phi')
                # print(loc_Phi)
                # print(scale_Phi)
                
                Phi_l = pyro.sample(Sites.Phi_l.format(l = l), 
                                    dist.Normal(loc_Phi, scale_Phi).to_event(1))
                # print('sampled_Phi')
                # print(Phi_l)
                Phi_l_list.append(Phi_l)
                loc_Phi_l_list.append(loc_Phi)
                
            
            # Compute structures
            joint_structure_list = []
            view_structure_list = []
            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(f"joint_structure_l{l}", 
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(f"view_structure_l{l}",
                                                    torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                
                total_structure_l = torch.add(joint_structure_l, view_structure_l)
                
                joint_structure_list.append(joint_structure_l)
                view_structure_list.append(view_structure_l)
                
            joint_structure_full = torch.cat(joint_structure_list, 1)
                
            # Calculate penalty terms
            # if (self.ortho_penalty != 0 and self.penalty_obj is None) or \
            #     (self.penalty_obj is not None and self.penalty_obj.weight != 0):
            #     total_penalty = torch.tensor(0.0)
            #     for l in range(len(X_list)):
            #         # cross_prod = view_structure_list[l].T @ joint_structure_full
            #         cross_prod = Phi_l_list[l].T @ Z
            #         # cross_prod = loc_Phi_l_list[l].T @ loc_Z_batch
                    
            #         total_penalty += torch.sum(cross_prod ** 2)
            #     if self.penalty_obj is None:
            #         pyro.factor("orthogonality", 
            #                     self.ortho_penalty * total_penalty * (len(batch_idx) / self.n),
            #                     has_rsample = True)
            #     else:
            #         pyro.factor("orthogonality", 
            #                 self.penalty_obj.weight * total_penalty * (len(batch_idx) / self.n),
            #                 has_rsample = True)
                
                # X_l = pyro.sample(f"X_l{l}", dist.Normal(total_structure_l, 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l])
                # if torch.isnan(X_l).any():
                #     print("NaN values found in X_l_tensor!")
                # if torch.isinf(X_l).any():
                #     print("Inf values found in X_l_tensor!")
            
                # pyro.sample(f"X_l{l}", dist.Normal(total_structure_l.index_select(0, batch_idx), 
                #                                    psi_sqrt_l_list[l]).to_event(1), 
                #             obs = X_list[l].index_select(0, batch_idx))
            
            
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
        # print(dir(self))
        assert self.params is not None, "Param dict is None"
        
        if self.dense:
            raise NotImplementedError
        
        m = len(batch_idx)
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        # n = X_list[0].shape[0]
        
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
            a_rho_gamma_l = self.params[Params.a_rho_gamma_l.format(l = l)]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l.format(l = l)]
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
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError
        
        # Local latent variables and observations
        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                                constraint = dist.constraints.positive)
        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            loc_Phi_l = pyro.param(Params.loc_Phi_l_pred.format(l = l), torch.zeros(self.n_predict, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l_pred.format(l = l), torch.ones(self.n_predict, k_l),
                                     constraint = dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)
            
        with pyro.plate("obs_pred", self.n_predict, subsample = batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            # Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                # loc_Z
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                Phi_l = pyro.sample(Sites.Phi_l_pred.format(l = l), 
                                    dist.Normal(loc_Phi, scale_Phi).to_event(1))
##############
