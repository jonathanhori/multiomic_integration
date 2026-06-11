import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from pyro.poutine import mask, scale

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
                 outcome = "gaussian",
                 penalty_obj = None,
                 a_sigma=2.1,
                 b_sigma=2.0,
                 a1_sigma=2.1,
                 a2_sigma=3.1,
                 alpha=3.0,
                 a_psi=3.0,
                 b_psi=1.0,
                 a_weibull=2.0,
                 b_weibull=1.0,
                 outcome_weight=1.0,
                 scores_init = None,
                 loadings_init = None,
                 device="cpu"):
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
        self.outcome = outcome
        self.a_sigma_y = 3.0
        self.b_sigma_y = 2.0
        self.a_sigma_beta = 2.1
        self.b_sigma_beta = 2.0
        self.a_weibull = a_weibull
        self.b_weibull = b_weibull
        
        self.outcome_weight = outcome_weight

        self.inv_gamma_init_param = 2.1
        self.init_param = 1.0

        self.dense = dense

        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []

        self.scores_init = scores_init
        self.loadings_init = loadings_init

        self.params = None  # Set by ModelHandler after training
        self.n_predict = None
        self.local_epochs = 0
        self.local_total_iters = None
        self.local_loss_history = []
        self.local_var_param_convergence_history = []

        self.device = torch.device(device) if isinstance(device, str) else device
        
        
    def _t(self, x):
        """Scalar float → device-placed tensor (avoids CPU/MPS log_prob mismatch)."""
        return torch.tensor(float(x), device=self.device)

    def forward(self,
                X,
                batch_idx,
                y = None,
                cens = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.forward_dense(X, batch_idx, y, cens)
        else:
            return self.forward_mgp(X, batch_idx, y, cens)

    def guide(self,
            X,
            batch_idx,
            y = None,
            cens = None):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.guide_dense(X, batch_idx, y)
        else:
            return self.guide_mgp(X, batch_idx, y)
            
    #######
    # SPARSE MODEL - MGP PRIOR ON LAMBDA
    #######
    
    def forward_mgp(self, X, batch_idx, y = None, cens = None):
        l = 0
        # m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # MGP prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda_jk ~ N(0, (rho^l_{jk})^-1 * (tau^l_k)^-1)
        ########################
        # tau - global shrinkage
        shape_vec = torch.cat([
            torch.tensor([self.a1_sigma]),
            torch.full((self.k - 1,), self.a2_sigma)
        ])
        delta_lambda = pyro.sample(Sites.delta_lambda_l.format(l=l),
                                   dist.Gamma(shape_vec, torch.ones(self.k)).to_event(1))
        tau_lambda_k_list = torch.cumprod(delta_lambda, dim=-1)

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
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))

            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))

            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0])

            if self.supervised_model:
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta.squeeze(0)))
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


    def guide_mgp(self, X, batch_idx, y = None):
        l = 0
        # m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        ######
        # Loadings
        ######
        a_delta_lambda_l = pyro.param(Params.a_delta_lambda_l.format(l=l),
                                      torch.full((self.k,), self.init_param),
                                      constraint=dist.constraints.positive)
        b_delta_lambda_l = pyro.param(Params.b_delta_lambda_l.format(l=l),
                                      torch.full((self.k,), self.init_param),
                                      constraint=dist.constraints.positive)
        pyro.sample(Sites.delta_lambda_l.format(l=l),
                    dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

        a_rho_lambda = pyro.param(Params.a_rho_lambda_l.format(l=l),
                                  torch.full((self.p, self.k), self.init_param),
                                  constraint=dist.constraints.positive)
        b_rho_lambda = pyro.param(Params.b_rho_lambda_l.format(l=l),
                                  torch.full((self.p, self.k), self.init_param),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.rho_lambda_l.format(l=l),
                    dist.Gamma(a_rho_lambda, b_rho_lambda).to_event(2)).squeeze()

        loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l=l), torch.zeros(self.p, self.k))
        scale_Lambda = pyro.param(Params.scale_Lambda_l.format(l=l),
                                  torch.full((self.p, self.k), self.init_param),
                                  constraint=dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l=l),
                             dist.Normal(loc_Lambda, scale_Lambda).to_event(2))
        
        
        # Psi: Variational params a, b
        a_psi = pyro.param(Params.a_psi_l.format(l=l),
                           torch.full((self.p,), self.a_psi),
                           constraint=dist.constraints.positive)
        b_psi = pyro.param(Params.b_psi_l.format(l=l),
                           torch.full((self.p,), self.b_psi),
                           constraint=dist.constraints.positive)
        pyro.sample(Sites.psi_l.format(l=l), dist.InverseGamma(a_psi, b_psi).to_event(1))

        if self.supervised_model:
            # Outcome model coefficients
            a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                      torch.full((self.k,), self.a_sigma_beta),
                                      constraint=dist.constraints.positive)
            b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                      torch.full((self.k,), self.b_sigma_beta),
                                      constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

            loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k))
            scale_beta = pyro.param(Params.scale_beta,
                                    torch.full((self.k,), self.init_param),
                                    constraint=dist.constraints.positive)
            beta = pyro.sample(Sites.beta,
                               dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

            if self.outcome == "gaussian":
                a_sigma_y = pyro.param(Params.a_sigma_y,
                                       torch.tensor(self.a_sigma_y),
                                       constraint=dist.constraints.positive)
                b_sigma_y = pyro.param(Params.b_sigma_y,
                                       torch.tensor(self.b_sigma_y),
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

        # Local latent variables Z: Variational params loc, scale
        Z_loc = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        Z_scale = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
                             constraint=dist.constraints.positive)
        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
      
      
        
    ########################################################################################
    # DENSE MODEL - NORMAL-IG PRIOR ON LAMBDA
    ########################################################################################   
    
    def forward_dense(self, X, batch_idx, y = None, cens = None):
        l = 0
        # m = len(batch_idx)
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
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))

            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.squeeze(0).T))

            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0])

            if self.supervised_model:
                outcome_structure = pyro.deterministic("outcome_structure",
                                                        torch.matmul(Z_batch, beta.squeeze(0)))
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


    def guide_dense(self, X, batch_idx, y = None):
        l = 0
        # m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        ######
        # Loadings
        ######
        a_sigma_lambda = pyro.param("a_sigma_lambda",
                                    torch.full((self.k,), self.a_sigma),
                                    constraint=dist.constraints.positive)
        b_sigma_lambda = pyro.param("b_sigma_lambda",
                                    torch.full((self.k,), self.b_sigma),
                                    constraint=dist.constraints.positive)
        pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

        if self.loadings_init is not None:
            _Lambda_default = align_tensor_shapes(self.loadings_init,
                                                  additional=self.k - self.loadings_init.shape[1])
        else:
            _Lambda_default = torch.zeros(self.p, self.k)
        loc_Lambda = pyro.param(Params.loc_Lambda_l.format(l=l), _Lambda_default)
        scale_Lambda = pyro.param(Params.scale_Lambda_l.format(l=l),
                                  torch.full((self.p, self.k), self.init_param),
                                  constraint=dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l=l),
                             dist.Normal(loc_Lambda, scale_Lambda).to_event(2))

        # Psi: Variational params a, b
        a_psi = pyro.param(Params.a_psi_l.format(l=l),
                           torch.full((self.p,), self.a_psi),
                           constraint=dist.constraints.positive)
        b_psi = pyro.param(Params.b_psi_l.format(l=l),
                           torch.full((self.p,), self.b_psi),
                           constraint=dist.constraints.positive)
        pyro.sample(Sites.psi_l.format(l=l), dist.InverseGamma(a_psi, b_psi).to_event(1))

        if self.supervised_model:
            # Outcome model coefficients
            a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                      torch.full((self.k,), self.a_sigma_beta),
                                      constraint=dist.constraints.positive)
            b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                      torch.full((self.k,), self.b_sigma_beta),
                                      constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

            loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k))
            scale_beta = pyro.param(Params.scale_beta,
                                    torch.full((self.k,), self.init_param),
                                    constraint=dist.constraints.positive)
            beta = pyro.sample(Sites.beta,
                               dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

            if self.outcome == "gaussian":
                a_sigma_y = pyro.param(Params.a_sigma_y,
                                       torch.tensor(self.a_sigma_y),
                                       constraint=dist.constraints.positive)
                b_sigma_y = pyro.param(Params.b_sigma_y,
                                       torch.tensor(self.b_sigma_y),
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

        # Local latent variables Z: Variational params loc, scale
        if self.scores_init is not None:
            _Z_default = align_tensor_shapes(self.scores_init,
                                             additional=self.k - self.scores_init.shape[1])
        else:
            _Z_default = torch.zeros(self.n, self.k)
        Z_loc = pyro.param(Params.loc_Z, _Z_default)
        Z_scale = pyro.param(Params.scale_Z, torch.ones(self.n, self.k),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))

    ########################################################################################
    # PREDICTION (local inference for test observations)
    # Global variational parameters are frozen from the training param store.
    # Only local Z scores (loc_Z_pred, scale_Z_pred) are optimized.
    # Setting model == guide for global sites makes their KL contribution zero, so the
    # prediction ELBO is driven entirely by the test-set observation likelihood and KL(q(Z_pred)||p(Z_pred)).
    ########################################################################################

    def predict_forward(self, X, batch_idx):
        assert self.params is not None, "self.params is None: run training inference first."
        if self.dense:
            return self.predict_forward_dense(X, batch_idx)
        else:
            return self.predict_forward_mgp(X, batch_idx)

    def predict_guide(self, X, batch_idx):
        assert self.params is not None, "self.params is None: run training inference first."
        if self.dense:
            return self.predict_guide_dense(X, batch_idx)
        else:
            return self.predict_guide_mgp(X, batch_idx)

    def predict_forward_mgp(self, X, batch_idx):
        l = 0
        if self.p is None:
            self.p = X[0].shape[1]

        # Global shrinkage — sample from trained variational distributions (model == guide -> KL = 0)
        a_delta_lambda_l = self.params[Params.a_delta_lambda_l.format(l=l)]
        b_delta_lambda_l = self.params[Params.b_delta_lambda_l.format(l=l)]
        pyro.sample(Sites.delta_lambda_l.format(l=l),
                    dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

        a_rho = self.params[Params.a_rho_lambda_l.format(l=l)]
        b_rho = self.params[Params.b_rho_lambda_l.format(l=l)]
        pyro.sample(Sites.rho_lambda_l.format(l=l),
                    dist.Gamma(a_rho, b_rho).to_event(2)).squeeze()

        loc_Lambda = self.params[Params.loc_Lambda_l.format(l=l)]
        scale_Lambda = self.params[Params.scale_Lambda_l.format(l=l)]
        Lambda = pyro.sample(Sites.Lambda_l.format(l=l),
                             dist.Normal(loc_Lambda, scale_Lambda).to_event(2)).squeeze()

        a_psi = self.params[Params.a_psi_l.format(l=l)]
        b_psi = self.params[Params.b_psi_l.format(l=l)]
        psi = pyro.sample(Sites.psi_l.format(l=l),
                          dist.InverseGamma(a_psi, b_psi).to_event(1))
        psi_sqrt = torch.sqrt(psi)

        if self.supervised_model:
            a_sigma_beta = self.params[Params.a_sigma_beta]
            b_sigma_beta = self.params[Params.b_sigma_beta]
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

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
            Z_batch = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                               torch.matmul(Z_batch, Lambda.T))

            pyro.sample(Sites.X_l_pred.format(l=l),
                        dist.Normal(torch.matmul(Z_batch, Lambda.T), psi_sqrt).to_event(1),
                        obs=X[0])

            if self.supervised_model:
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(Z_batch, beta.squeeze(0)))
                if self.outcome == "gaussian":
                    pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y))
                elif self.outcome == "censored":
                    weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                    pyro.sample(Sites.y_pred, dist.Weibull(weibull_scale, weibull_concentration))
                else:
                    raise NotImplementedError

    def predict_guide_mgp(self, X, batch_idx):
        l = 0
        if self.p is None:
            self.p = X[0].shape[1]

        # Global params — frozen from training param store
        a_delta_lambda_l = self.params[Params.a_delta_lambda_l.format(l=l)]
        b_delta_lambda_l = self.params[Params.b_delta_lambda_l.format(l=l)]
        pyro.sample(Sites.delta_lambda_l.format(l=l),
                    dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

        a_rho = self.params[Params.a_rho_lambda_l.format(l=l)]
        b_rho = self.params[Params.b_rho_lambda_l.format(l=l)]
        pyro.sample(Sites.rho_lambda_l.format(l=l),
                    dist.Gamma(a_rho, b_rho).to_event(2)).squeeze()

        loc_Lambda = self.params[Params.loc_Lambda_l.format(l=l)]
        scale_Lambda = self.params[Params.scale_Lambda_l.format(l=l)]
        pyro.sample(Sites.Lambda_l.format(l=l),
                    dist.Normal(loc_Lambda, scale_Lambda).to_event(2))

        a_psi = self.params[Params.a_psi_l.format(l=l)]
        b_psi = self.params[Params.b_psi_l.format(l=l)]
        pyro.sample(Sites.psi_l.format(l=l),
                    dist.InverseGamma(a_psi, b_psi).to_event(1))

        if self.supervised_model:
            a_sigma_beta = self.params[Params.a_sigma_beta]
            b_sigma_beta = self.params[Params.b_sigma_beta]
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

            loc_beta = self.params[Params.loc_beta]
            scale_beta = self.params[Params.scale_beta]
            beta = pyro.sample(Sites.beta,
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

        # Local params — fresh variational parameters optimized during local inference
        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z_loc_batch = loc_Z[batch_idx]
            Z_scale_batch = scale_Z[batch_idx]
            Z_batch = pyro.sample(Sites.Z_pred,
                                  dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))

    def predict_forward_dense(self, X, batch_idx):
        l = 0
        if self.p is None:
            self.p = X[0].shape[1]

        # Global params — sample from trained variational distributions (model == guide -> KL = 0)
        a_sigma_lambda = self.params["a_sigma_lambda"]
        b_sigma_lambda = self.params["b_sigma_lambda"]
        sigma2_lambda = pyro.sample("sigma2_lambda",
                                    dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))
        sigma_lambda = torch.sqrt(sigma2_lambda)

        loc_Lambda = self.params[Params.loc_Lambda_l.format(l=l)]
        scale_Lambda = self.params[Params.scale_Lambda_l.format(l=l)]
        Lambda = pyro.sample(Sites.Lambda_l.format(l=l),
                             dist.Normal(loc_Lambda, scale_Lambda).to_event(2)).squeeze(0)

        a_psi = self.params[Params.a_psi_l.format(l = l)]
        b_psi = self.params[Params.b_psi_l.format(l = l)]
        psi = pyro.sample(Sites.psi_l.format(l=l),
                          dist.InverseGamma(a_psi, b_psi).to_event(1))
        psi_sqrt = torch.sqrt(psi)

        if self.supervised_model:
            a_sigma_beta = self.params[Params.a_sigma_beta]
            b_sigma_beta = self.params[Params.b_sigma_beta]
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

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
            Z_batch = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                               torch.matmul(Z_batch, Lambda.T))

            pyro.sample(Sites.X_l_pred.format(l=l),
                        dist.Normal(torch.matmul(Z_batch, Lambda.T), psi_sqrt).to_event(1),
                        obs=X[0])

            if self.supervised_model:
                outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                       torch.matmul(Z_batch, beta.squeeze(0)))
                if self.outcome == "gaussian":
                    pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y))
                elif self.outcome == "censored":
                    weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                    pyro.sample(Sites.y_pred, dist.Weibull(weibull_scale, weibull_concentration))
                else:
                    raise NotImplementedError

    def predict_guide_dense(self, X, batch_idx):
        l = 0
        if self.p is None:
            self.p = X[0].shape[1]

        # Global params — frozen from training param store
        a_sigma_lambda = self.params["a_sigma_lambda"]
        b_sigma_lambda = self.params["b_sigma_lambda"]
        pyro.sample("sigma2_lambda",
                    dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

        loc_Lambda = self.params[Params.loc_Lambda_l.format(l=l)]
        scale_Lambda = self.params[Params.scale_Lambda_l.format(l=l)]
        pyro.sample(Sites.Lambda_l.format(l=l),
                    dist.Normal(loc_Lambda, scale_Lambda).to_event(2))

        a_psi = self.params[Params.a_psi_l.format(l = l)]
        b_psi = self.params[Params.b_psi_l.format(l = l)]
        pyro.sample(Sites.psi_l.format(l=l),
                    dist.InverseGamma(a_psi, b_psi).to_event(1))

        if self.supervised_model:
            a_sigma_beta = self.params[Params.a_sigma_beta]
            b_sigma_beta = self.params[Params.b_sigma_beta]
            pyro.sample(Sites.sigma2_beta,
                        dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

            loc_beta = self.params[Params.loc_beta]
            scale_beta = self.params[Params.scale_beta]
            beta = pyro.sample(Sites.beta,
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

        # Local params — fresh variational parameters optimized during local inference
        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z_loc_batch = loc_Z[batch_idx]
            Z_scale_batch = scale_Z[batch_idx]
            Z_batch = pyro.sample(Sites.Z_pred,
                                  dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
