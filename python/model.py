import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
# from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal
from pyro.poutine import mask, scale

from constants import Sites, Params
from data_utils import align_tensor_shapes

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class SupMultiviewDecomp(PyroModule):
    def __init__(self,
                 k,
                 k_l_list,
                 ortho_penalty = 0.0,
                 include_view_factors = False,
                 dense = True,
                 outcome = "gaussian",
                 penalty_obj = None,
                 a_sigma_joint=2.0,
                 b_sigma_joint=2.0,
                 a_sigma_view=2.0,
                 b_sigma_view=2.0,
                 a1_sigma_joint=2.1,
                 a2_sigma_joint=3.1,
                 a1_sigma_view=2.1,
                 a2_sigma_view=3.1,
                 alpha_joint = 3.0,
                 alpha_individual = 3.0,
                 a_psi=3.0,
                 b_psi=1.0,
                 a_sigma_y = 3.0,
                 b_sigma_y = 1.0,
                 a_sigma_beta = 2.0,
                 b_sigma_beta = 2.0,
                 a_weibull = 2.0,
                 b_weibull = 1.0,
                 outcome_weight = 1.0,
                 joint_scores_init = None,
                 joint_loadings_list_init = None,
                 view_scores_list_init = None,
                 view_loadings_list_init = None,
                 device = "cpu"
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

        self.outcome_weight = outcome_weight

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
        self.init_param = 1.0

        self.device = torch.device(device) if isinstance(device, str) else device
    
        
    def _t(self, x):
        """Scalar float → device-placed tensor (avoids CPU/MPS log_prob mismatch)."""
        return torch.tensor(float(x), device=self.device)

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
            shape_vec = torch.cat([
                torch.tensor([self.a1_sigma_view]),
                torch.full((k_l - 1,), self.a2_sigma_view)
            ])
            delta_gamma = pyro.sample(Sites.delta_gamma_l.format(l=l),
                                      dist.Gamma(shape_vec, torch.ones(k_l)).to_event(1))
            tau_gamma_k_list = torch.cumprod(delta_gamma, dim=-1)

            rho_gamma = pyro.sample(Sites.rho_gamma_l.format(l=l),
                                    dist.Gamma(self.alpha_individual / 2, self.alpha_individual / 2).expand([p_l, k_l]).to_event(2)).squeeze()

            precision = rho_gamma * tau_gamma_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)
            sigma_gamma_l = precision.pow_(-0.5)
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l=l),
                                  dist.Normal(torch.zeros(p_l, k_l), sigma_gamma_l).to_event(2))

            Gamma_l_list.append(Gamma_l)

        # SHARED
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            shape_vec = torch.cat([
                torch.tensor([self.a1_sigma_joint]),
                torch.full((self.k - 1,), self.a2_sigma_joint)
            ])
            delta_lambda = pyro.sample(Sites.delta_lambda_l.format(l=l),
                                       dist.Gamma(shape_vec, torch.ones(self.k)).to_event(1))
            tau_lambda_k_list = torch.cumprod(delta_lambda, dim=-1)

            rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l=l),
                                     dist.Gamma(self.alpha_joint / 2, self.alpha_joint / 2).expand([p_l, self.k]).to_event(2)).squeeze()

            precision = rho_lambda * tau_lambda_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)
            sigma_lambda_l = precision.pow_(-0.5)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
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
            
            with scale(scale=self.outcome_weight):
                if self.outcome == "gaussian":
                    y_pred = pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y),
                                         obs=y)
                    return y_pred
                elif self.outcome == "censored":
                    weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                    with mask(mask=(cens == 1)):
                        pyro.sample(Sites.y, dist.Weibull(weibull_scale, weibull_concentration), obs=y)
                    log_surv = -torch.pow(y / weibull_scale, weibull_concentration)
                    with mask(mask=(cens == 0)):
                        pyro.factor(Sites.censored, log_surv)
                else:
                    raise NotImplementedError


    def predict_forward(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_forward_dense(X_list, batch_idx)
        else:
            return self.predict_forward_mgp(X_list, batch_idx)

    def predict_forward_mgp(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
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
            a_delta_gamma_l = self.params[Params.a_delta_gamma_l.format(l=l)]
            b_delta_gamma_l = self.params[Params.b_delta_gamma_l.format(l=l)]
            pyro.sample(Sites.delta_gamma_l.format(l=l),
                        dist.Gamma(a_delta_gamma_l, b_delta_gamma_l).to_event(1))

            a_rho_gamma_l = self.params[Params.a_rho_gamma_l.format(l=l)]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l.format(l=l)]
            pyro.sample(Sites.rho_gamma_l.format(l=l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).to_event(2)).squeeze()

            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l=l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l=l)]
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l=l),
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))
            Gamma_l_list.append(Gamma_l)

        # SHARED
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_delta_lambda_l = self.params[Params.a_delta_lambda_l.format(l=l)]
            b_delta_lambda_l = self.params[Params.b_delta_lambda_l.format(l=l)]
            pyro.sample(Sites.delta_lambda_l.format(l=l),
                        dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l=l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l=l)]
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).to_event(2)).squeeze()

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        psi_sqrt_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l=l)]
            b_psi_l = self.params[Params.b_psi_l.format(l=l)]
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(Sites.Phi_l_pred.format(l=l),
                                    dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l_pred.format(l=l),
                                                      torch.matmul(Phi_l_list[l], Gamma_l_list[l].squeeze(0).T))
                total_structure_l = torch.add(joint_structure_l, view_structure_l)

                X_l = pyro.sample(Sites.X_l_pred.format(l=l),
                                  dist.Normal(total_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(full_factors, beta))
            else:
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(Z, beta))

            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y))
                return y_pred
            elif self.outcome == "censored":
                weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                y_pred = pyro.sample(Sites.y_pred, dist.Weibull(weibull_scale, weibull_concentration))
                return y_pred
            else:
                raise NotImplementedError

    def predict_forward_dense(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        # VIEW SPECIFIC
        Gamma_l_list = []
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            a_sigma_gamma = self.params[Params.a_sigma_gamma_l.format(l=l)]
            b_sigma_gamma = self.params[Params.b_sigma_gamma_l.format(l=l)]
            pyro.sample(Sites.sigma2_gamma_l.format(l=l),
                        dist.InverseGamma(a_sigma_gamma, b_sigma_gamma).to_event(1))

            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l=l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l=l)]
            Gamma_l = pyro.sample(Sites.Gamma_l.format(l=l),
                                  dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2)).squeeze(0)
            Gamma_l_list.append(Gamma_l)

        # SHARED
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = self.params[Params.a_sigma_lambda_l.format(l=l)]
            b_sigma_lambda = self.params[Params.b_sigma_lambda_l.format(l=l)]
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2)).squeeze(0)
            Lambda_l_list.append(Lambda_l)

        psi_sqrt_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l=l)]
            b_psi_l = self.params[Params.b_psi_l.format(l=l)]
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            Phi_l_list = []
            for l, k_l in enumerate(self.k_l_list):
                Phi_l = pyro.sample(Sites.Phi_l_pred.format(l=l),
                                    dist.Normal(0., 1.).expand([k_l]).to_event(1))
                Phi_l_list.append(Phi_l)

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].T))
                view_structure_l = pyro.deterministic(Sites.view_structure_l_pred.format(l=l),
                                                      torch.matmul(Phi_l_list[l], Gamma_l_list[l].T))
                total_structure_l = torch.add(joint_structure_l, view_structure_l)

                X_l = pyro.sample(Sites.X_l_pred.format(l=l),
                                  dist.Normal(total_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            if self.include_view_factors:
                full_factors = torch.cat((Z, *Phi_l_list), 1)
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(full_factors, beta))
            else:
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(Z, beta))

            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y))
                return y_pred
            elif self.outcome == "censored":
                weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                y_pred = pyro.sample(Sites.y_pred, dist.Weibull(weibull_scale, weibull_concentration))
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
            with scale(scale=self.outcome_weight):
                if self.outcome == "gaussian":
                    pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y), obs=y)
                elif self.outcome == "censored":
                    weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                    with mask(mask=(cens == 1)):
                        pyro.sample(Sites.y, dist.Weibull(weibull_scale, weibull_concentration), obs=y)
                    log_surv = -torch.pow(y / weibull_scale, weibull_concentration)
                    with mask(mask=(cens == 0)):
                        pyro.factor(Sites.censored, log_surv)
                else:
                    raise NotImplementedError


    def guide_mgp(self,
              X_list,
              batch_idx,
              y = None,
              cens = None):
        if self.dense:
            raise NotImplementedError

        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        ########################
        # ---- Loadings --------
        ########################
        # VIEW SPECIFIC
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            a_delta_gamma_l = pyro.param(Params.a_delta_gamma_l.format(l=l),
                                         torch.full((k_l,), self.init_param),
                                         constraint=dist.constraints.positive)
            b_delta_gamma_l = pyro.param(Params.b_delta_gamma_l.format(l=l),
                                         torch.full((k_l,), self.init_param),
                                         constraint=dist.constraints.positive)
            pyro.sample(Sites.delta_gamma_l.format(l=l),
                        dist.Gamma(a_delta_gamma_l, b_delta_gamma_l).to_event(1))

            a_rho_gamma_l = pyro.param(Params.a_rho_gamma_l.format(l=l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint=dist.constraints.positive)
            b_rho_gamma_l = pyro.param(Params.b_rho_gamma_l.format(l=l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint=dist.constraints.positive)
            pyro.sample(Sites.rho_gamma_l.format(l=l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).to_event(2)).squeeze()

            _Gamma_default = (
                align_tensor_shapes(self.view_loadings_list_init[l],
                                    additional=k_l - self.view_loadings_list_init[l].shape[1])
                if self.view_loadings_list_init is not None
                else torch.zeros(p_l, k_l)
            )
            loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l=l), _Gamma_default)
            scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l=l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint=dist.constraints.positive)
            pyro.sample(Sites.Gamma_l.format(l=l),
                        dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))

        # SHARED
        for l, p_l in enumerate(self.p_l_list):
            a_delta_lambda_l = pyro.param(Params.a_delta_lambda_l.format(l=l),
                                          torch.full((self.k,), self.init_param),
                                          constraint=dist.constraints.positive)
            b_delta_lambda_l = pyro.param(Params.b_delta_lambda_l.format(l=l),
                                          torch.full((self.k,), self.init_param),
                                          constraint=dist.constraints.positive)
            pyro.sample(Sites.delta_lambda_l.format(l=l),
                        dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

            a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param),
                                        constraint=dist.constraints.positive)
            b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).to_event(2)).squeeze()

            _Lambda_default = (
                align_tensor_shapes(self.joint_loadings_list_init[l],
                                    additional=self.k - self.joint_loadings_list_init[l].shape[1])
                if self.joint_loadings_list_init is not None
                else torch.zeros(p_l, self.k)
            )
            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), _Lambda_default)
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))

        ########################
        # ---- Observations ----
        ########################
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = pyro.param(Params.a_psi_l.format(l=l),
                                 torch.full((p_l,), self.a_psi),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.full((p_l,), self.b_psi),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.full((self.num_outcome_factors,), self.a_sigma_beta),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.full((self.num_outcome_factors,), self.b_sigma_beta),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.full((self.num_outcome_factors,), self.init_param),
                                constraint=dist.constraints.positive)
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                   torch.tensor(self.inv_gamma_init_param),
                                   constraint=dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                   torch.tensor(self.init_param),
                                   constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
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

        _Z_default = (
            align_tensor_shapes(self.joint_scores_init,
                                additional=self.k - self.joint_scores_init.shape[1])
            if self.joint_scores_init is not None
            else torch.zeros(self.n, self.k)
        )
        loc_Z = pyro.param(Params.loc_Z, _Z_default)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.full((self.n, self.k), self.init_param),
                             constraint=dist.constraints.positive)

        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            _Phi_default = (
                align_tensor_shapes(self.view_scores_list_init[l],
                                    additional=k_l - self.view_scores_list_init[l].shape[1])
                if self.view_scores_list_init is not None
                else torch.zeros(self.n, k_l)
            )
            loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l=l), _Phi_default)
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l=l),
                                     torch.full((self.n, k_l), self.init_param),
                                     constraint=dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            for l, k_l in enumerate(self.k_l_list):
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                pyro.sample(Sites.Phi_l.format(l=l),
                            dist.Normal(loc_Phi, scale_Phi).to_event(1))


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
            
            a_sigma_gamma = pyro.param(Params.a_sigma_gamma_l.format(l=l),
                                       torch.full((k_l,), self.inv_gamma_init_param),
                                       constraint=dist.constraints.positive)
            b_sigma_gamma = pyro.param(Params.b_sigma_gamma_l.format(l=l),
                                       torch.full((k_l,), self.inv_gamma_init_param),
                                       constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_gamma_l.format(l=l),
                        dist.InverseGamma(a_sigma_gamma, b_sigma_gamma).to_event(1))

            _Gamma_default = (
                align_tensor_shapes(self.view_loadings_list_init[l],
                                    additional=k_l - self.view_loadings_list_init[l].shape[1])
                if self.view_loadings_list_init is not None
                else torch.zeros(p_l, k_l)
            )
            loc_Gamma_l = pyro.param(Params.loc_Gamma_l.format(l=l), _Gamma_default)
            scale_Gamma_l = pyro.param(Params.scale_Gamma_l.format(l=l),
                                       torch.full((p_l, k_l), self.init_param),
                                       constraint=dist.constraints.positive)
            pyro.sample(Sites.Gamma_l.format(l=l),
                        dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))

        # SHARED
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = pyro.param(Params.a_sigma_lambda_l.format(l=l),
                                        torch.full((self.k,), self.inv_gamma_init_param),
                                        constraint=dist.constraints.positive)
            b_sigma_lambda = pyro.param(Params.b_sigma_lambda_l.format(l=l),
                                        torch.full((self.k,), self.inv_gamma_init_param),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

            _Lambda_default = (
                align_tensor_shapes(self.joint_loadings_list_init[l],
                                    additional=self.k - self.joint_loadings_list_init[l].shape[1])
                if self.joint_loadings_list_init is not None
                else torch.zeros(p_l, self.k)
            )
            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), _Lambda_default)
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))

        ########################
        # ---- Observations ----
        ########################
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = pyro.param(Params.a_psi_l.format(l=l),
                                 torch.full((p_l,), self.a_psi),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.full((p_l,), self.b_psi),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.full((self.num_outcome_factors,), self.a_sigma_beta),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.full((self.num_outcome_factors,), self.b_sigma_beta),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.num_outcome_factors))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.full((self.num_outcome_factors,), self.init_param),
                                constraint=dist.constraints.positive)
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                   torch.tensor(self.inv_gamma_init_param),
                                   constraint=dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                   torch.tensor(self.init_param),
                                   constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
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

        _Z_default = (
            align_tensor_shapes(self.joint_scores_init,
                                additional=self.k - self.joint_scores_init.shape[1])
            if self.joint_scores_init is not None
            else torch.zeros(self.n, self.k)
        )
        loc_Z = pyro.param(Params.loc_Z, _Z_default)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.full((self.n, self.k), self.init_param),
                             constraint=dist.constraints.positive)

        loc_Phi_list = []
        scale_Phi_list = []
        for l, k_l in enumerate(self.k_l_list):
            _Phi_default = (
                align_tensor_shapes(self.view_scores_list_init[l],
                                    additional=k_l - self.view_scores_list_init[l].shape[1])
                if self.view_scores_list_init is not None
                else torch.zeros(self.n, k_l)
            )
            loc_Phi_l = pyro.param(Params.loc_Phi_l.format(l=l), _Phi_default)
            scale_Phi_l = pyro.param(Params.scale_Phi_l.format(l=l),
                                     torch.full((self.n, k_l), self.init_param),
                                     constraint=dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            for l, k_l in enumerate(self.k_l_list):
                loc_Phi = loc_Phi_list[l][batch_idx]
                scale_Phi = scale_Phi_list[l][batch_idx]
                pyro.sample(Sites.Phi_l.format(l=l),
                            dist.Normal(loc_Phi, scale_Phi).to_event(1))

    def predict_guide(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None"
        if self.dense:
            return self.predict_guide_dense(X_list, batch_idx)
        else:
            return self.predict_guide_mgp(X_list, batch_idx)

    def predict_guide_mgp(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        # VIEW SPECIFIC — global params frozen from training
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            a_delta_gamma_l = self.params[Params.a_delta_gamma_l.format(l=l)]
            b_delta_gamma_l = self.params[Params.b_delta_gamma_l.format(l=l)]
            pyro.sample(Sites.delta_gamma_l.format(l=l),
                        dist.Gamma(a_delta_gamma_l, b_delta_gamma_l).to_event(1))

            a_rho_gamma_l = self.params[Params.a_rho_gamma_l.format(l=l)]
            b_rho_gamma_l = self.params[Params.b_rho_gamma_l.format(l=l)]
            pyro.sample(Sites.rho_gamma_l.format(l=l),
                        dist.Gamma(a_rho_gamma_l, b_rho_gamma_l).to_event(2)).squeeze()

            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l=l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l=l)]
            pyro.sample(Sites.Gamma_l.format(l=l),
                        dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))

        # SHARED
        for l, p_l in enumerate(self.p_l_list):
            a_delta_lambda_l = self.params[Params.a_delta_lambda_l.format(l=l)]
            b_delta_lambda_l = self.params[Params.b_delta_lambda_l.format(l=l)]
            pyro.sample(Sites.delta_lambda_l.format(l=l),
                        dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l=l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l=l)]
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).to_event(2)).squeeze()

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))

        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l=l)]
            b_psi_l = self.params[Params.b_psi_l.format(l=l)]
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        # Local — fresh params optimized during local inference
        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)
        loc_Phi_list, scale_Phi_list = [], []
        for l, k_l in enumerate(self.k_l_list):
            loc_Phi_l = pyro.param(Params.loc_Phi_l_pred.format(l=l), torch.zeros(self.n_predict, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l_pred.format(l=l), torch.ones(self.n_predict, k_l),
                                     constraint=dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            pyro.sample(Sites.Z_pred,
                        dist.Normal(loc_Z[batch_idx], scale_Z[batch_idx]).to_event(1))
            for l, k_l in enumerate(self.k_l_list):
                pyro.sample(Sites.Phi_l_pred.format(l=l),
                            dist.Normal(loc_Phi_list[l][batch_idx],
                                        scale_Phi_list[l][batch_idx]).to_event(1))

    def predict_guide_dense(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        # VIEW SPECIFIC
        for l, (p_l, k_l) in enumerate(zip(self.p_l_list, self.k_l_list)):
            a_sigma_gamma = self.params[Params.a_sigma_gamma_l.format(l=l)]
            b_sigma_gamma = self.params[Params.b_sigma_gamma_l.format(l=l)]
            pyro.sample(Sites.sigma2_gamma_l.format(l=l),
                        dist.InverseGamma(a_sigma_gamma, b_sigma_gamma).to_event(1))

            loc_Gamma_l = self.params[Params.loc_Gamma_l.format(l=l)]
            scale_Gamma_l = self.params[Params.scale_Gamma_l.format(l=l)]
            pyro.sample(Sites.Gamma_l.format(l=l),
                        dist.Normal(loc_Gamma_l, scale_Gamma_l).to_event(2))

        # SHARED
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = self.params[Params.a_sigma_lambda_l.format(l=l)]
            b_sigma_lambda = self.params[Params.b_sigma_lambda_l.format(l=l)]
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))

        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l=l)]
            b_psi_l = self.params[Params.b_psi_l.format(l=l)]
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        a_sigma_beta = self.params[Params.a_sigma_beta]
        b_sigma_beta = self.params[Params.b_sigma_beta]
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = self.params[Params.loc_beta]
        scale_beta = self.params[Params.scale_beta]
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = self.params[Params.a_sigma_y]
            b_sigma_y = self.params[Params.b_sigma_y]
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
        elif self.outcome == "censored":
            a_weibull_c = self.params[Params.a_weibull_concentration]
            b_weibull_c = self.params[Params.b_weibull_concentration]
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)
        loc_Phi_list, scale_Phi_list = [], []
        for l, k_l in enumerate(self.k_l_list):
            loc_Phi_l = pyro.param(Params.loc_Phi_l_pred.format(l=l), torch.zeros(self.n_predict, k_l))
            scale_Phi_l = pyro.param(Params.scale_Phi_l_pred.format(l=l), torch.ones(self.n_predict, k_l),
                                     constraint=dist.constraints.positive)
            loc_Phi_list.append(loc_Phi_l)
            scale_Phi_list.append(scale_Phi_l)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            pyro.sample(Sites.Z_pred,
                        dist.Normal(loc_Z[batch_idx], scale_Z[batch_idx]).to_event(1))
            for l, k_l in enumerate(self.k_l_list):
                pyro.sample(Sites.Phi_l_pred.format(l=l),
                            dist.Normal(loc_Phi_list[l][batch_idx],
                                        scale_Phi_list[l][batch_idx]).to_event(1))

