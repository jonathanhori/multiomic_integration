import os
import sys

from functools import partial

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.poutine import mask, scale

from constants import Sites, Params
from data_utils import align_tensor_shapes

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class SupMultiviewShared(PyroModule):
    """
    X_l = Z @ Lambda_l^T + E_l,   l = 1, ..., L
    y   = Z @ beta + e

    Z (n x k): shared scores across all views
    Lambda_l (p_l x k): view-specific loading matrices
    """
    def __init__(self,
                 k,
                 dense=True,
                 outcome="gaussian",
                 a_sigma_joint=2.0,
                 b_sigma_joint=2.0,
                 a1_sigma_joint=2.1,
                 a2_sigma_joint=3.1,
                 alpha_joint=3.0,
                 a_psi=3.0,
                 b_psi=1.0,
                 a_sigma_y=3.0,
                 b_sigma_y=1.0,
                 a_sigma_beta=3.0,
                 b_sigma_beta=1.0,
                 a_weibull=2.0,
                 b_weibull=1.0,
                 outcome_weight=1.0,
                 joint_scores_init=None,
                 joint_loadings_list_init=None,
                 device="cpu"):
        super().__init__()

        self.model_type = "SupMultiviewShared"
        self.n = None
        self.n_predict = None
        self.k = k
        self.p_l_list = None

        self.dense = dense
        self.outcome = outcome

        if dense:
            self.a_sigma_joint = a_sigma_joint
            self.b_sigma_joint = b_sigma_joint
        if not dense:
            self.a1_sigma_joint = a1_sigma_joint
            self.a2_sigma_joint = a2_sigma_joint
            self.alpha_joint = alpha_joint

        self.a_psi = a_psi
        self.b_psi = b_psi
        self.a_sigma_y = a_sigma_y
        self.b_sigma_y = b_sigma_y
        self.a_sigma_beta = a_sigma_beta
        self.b_sigma_beta = b_sigma_beta
        self.a_weibull = a_weibull
        self.b_weibull = b_weibull

        # Up-weighting factor for the outcome log-likelihood. The reconstruction
        # likelihood contributes L * p_l terms per observation while the outcome
        # contributes a single term, so the shared scores Z are estimated almost
        # entirely from X. Setting outcome_weight ~ L * p_l rebalances the ELBO so
        # the outcome informs Z comparably to the views. outcome_weight=1.0
        # recovers the original (reconstruction-dominated) objective.
        self.outcome_weight = outcome_weight

        self.joint_scores_init = joint_scores_init
        self.joint_loadings_list_init = joint_loadings_list_init

        self.ortho_penalty = 0.0
        self.penalty_obj = None

        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []

        self.local_epochs = 0
        self.local_total_iters = None
        self.local_loss_history = []
        self.local_var_param_convergence_history = []

        self.params = None

        self.device = torch.device(device) if isinstance(device, str) else device

        self.inv_gamma_init_param = 2.1
        self.init_param = 1.0

    def _t(self, x):
        """Scalar float → device-placed tensor (avoids CPU/MPS log_prob mismatch)."""
        return torch.tensor(float(x), device=self.device)


    def forward(self, X_list, batch_idx, y=None, cens=None):
        if self.dense:
            return self.forward_dense(X_list, batch_idx, y, cens)
        else:
            return self.forward_mgp(X_list, batch_idx, y, cens)

    def guide(self, X_list, batch_idx, y=None, cens=None):
        if self.dense:
            return self.guide_dense(X_list, batch_idx, y)
        else:
            return self.guide_mgp(X_list, batch_idx, y)


    def forward_mgp(self, X_list, batch_idx, y=None, cens=None):
        """
        X_l = Z @ Lambda_l^T + E_l,   l = 1, ..., L
        y   = Z @ beta + e
        With MGP shrinkage prior on loadings.
        """
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        n = X_list[0].shape[0]
        if self.n is None:
            self.n = n

        ########################
        # ---- Loadings --------
        # Lambda^l_jk ~ N(0, (rho^l_jk)^-1 * (tau^l_k)^-1)
        ########################
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            # tau - global shrinkage
            shape_vec = torch.cat([
                torch.tensor([self.a1_sigma_joint], device=self.device),
                torch.full((self.k - 1,), self.a2_sigma_joint, device=self.device)
            ])
            delta_lambda = pyro.sample(Sites.delta_lambda_l.format(l=l),
                                       dist.Gamma(shape_vec, torch.ones(self.k, device=self.device)).to_event(1))
            tau_lambda_k_list = torch.cumprod(delta_lambda, dim=-1)

            # rho - local shrinkage
            rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l=l),
                                     dist.Gamma(self._t(self.alpha_joint / 2), self._t(self.alpha_joint / 2)).expand([p_l, self.k]).to_event(2)).squeeze()

            precision = rho_lambda * tau_lambda_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)
            sigma_lambda_l = precision.pow_(-0.5)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(torch.zeros(p_l, self.k, device=self.device), sigma_lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        psi_sqrt_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(self._t(self.a_psi), self._t(self.b_psi)).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self._t(self.a_sigma_beta), self._t(self.b_sigma_beta)).expand([self.k]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.k, device=self.device), sigma2_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(self._t(self.a_sigma_y), self._t(self.b_sigma_y))).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(self._t(self.a_weibull), self._t(self.b_weibull)))
        else:
            raise NotImplementedError

        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(self._t(0.), self._t(1.)).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                X_l = pyro.sample(Sites.X_l.format(l=l),
                                  dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                   torch.matmul(Z, beta))
            # Up-weight the outcome likelihood so it informs Z comparably to the views.
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


    def forward_dense(self, X_list, batch_idx, y=None, cens=None):
        """
        X_l = Z @ Lambda_l^T + E_l,   l = 1, ..., L
        y   = Z @ beta + e
        """
        p_l_list = [X_l.shape[1] for X_l in X_list]
        n = X_list[0].shape[0]
        if self.n is None:
            self.n = n
        if self.p_l_list is None:
            self.p_l_list = p_l_list

        ########################
        # ---- Loadings --------
        # ARD prior: Lambda^l_jk ~ N(0, sigma^l_k^2)
        # sigma^l_k^2 ~ InvGamma(a_sigma, b_sigma)
        ########################
        Lambda_l_list = []
        for l, p_l in enumerate(p_l_list):
            sigma2_lambda_l = pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                                          dist.InverseGamma(self._t(self.a_sigma_joint), self._t(self.b_sigma_joint)).expand([self.k]).to_event(1))
            sigma_lambda_l = torch.sqrt(sigma2_lambda_l)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(torch.zeros(p_l, self.k, device=self.device), sigma_lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        psi_sqrt_l_list = []
        for l, p_l in enumerate(p_l_list):
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(self._t(self.a_psi), self._t(self.b_psi)).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self._t(self.a_sigma_beta), self._t(self.b_sigma_beta)).expand([self.k]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.k, device=self.device), sigma2_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(self._t(self.a_sigma_y), self._t(self.b_sigma_y))).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(self._t(self.a_weibull), self._t(self.b_weibull)))
        else:
            raise NotImplementedError

        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(self._t(0.), self._t(1.)).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                X_l = pyro.sample(Sites.X_l.format(l=l),
                                  dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            outcome_structure = pyro.deterministic(Sites.outcome_structure,
                                                   torch.matmul(Z, beta))
            # Up-weight the outcome likelihood so it informs Z comparably to the views.
            with scale(scale=self.outcome_weight):
                if self.outcome == "gaussian":
                    pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y),
                                obs=y)
                elif self.outcome == "censored":
                    weibull_scale = torch.exp(outcome_structure).clamp(min=1e-10)
                    with mask(mask=(cens == 1)):
                        pyro.sample(Sites.y, dist.Weibull(weibull_scale, weibull_concentration), obs=y)
                    log_surv = -torch.pow(y / weibull_scale, weibull_concentration)
                    with mask(mask=(cens == 0)):
                        pyro.factor(Sites.censored, log_surv)
                else:
                    raise NotImplementedError


    def guide_mgp(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        ########################
        # ---- Loadings --------
        ########################
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_delta_lambda_l = pyro.param(Params.a_delta_lambda_l.format(l=l),
                                          torch.full((self.k,), self.init_param, device=self.device),
                                          constraint=dist.constraints.positive)
            b_delta_lambda_l = pyro.param(Params.b_delta_lambda_l.format(l=l),
                                          torch.full((self.k,), self.init_param, device=self.device),
                                          constraint=dist.constraints.positive)
            pyro.sample(Sites.delta_lambda_l.format(l=l),
                        dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

            a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param, device=self.device),
                                        constraint=dist.constraints.positive)
            b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l=l),
                                        torch.full((p_l, self.k), self.init_param, device=self.device),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).to_event(2)).squeeze()

            _Lambda_default = (
                self.joint_loadings_list_init[l].to(self.device)
                if self.joint_loadings_list_init is not None
                else torch.zeros(p_l, self.k, device=self.device)
            )
            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), _Lambda_default)
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.ones(p_l, self.k, device=self.device),
                                        constraint=dist.constraints.positive)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = pyro.param(Params.a_psi_l.format(l=l),
                                 torch.full((p_l,), self.a_psi, device=self.device),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.full((p_l,), self.b_psi, device=self.device),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.full((self.k,), self.a_sigma_beta, device=self.device),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.full((self.k,), self.b_sigma_beta, device=self.device),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k, device=self.device))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.full((self.k,), self.init_param, device=self.device),
                                constraint=dist.constraints.positive)
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                   torch.tensor(self.a_sigma_y, device=self.device),
                                   constraint=dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                   torch.tensor(self.b_sigma_y, device=self.device),
                                   constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
        elif self.outcome == "censored":
            a_weibull_c = pyro.param(Params.a_weibull_concentration,
                                     torch.tensor(self.a_weibull, device=self.device),
                                     constraint=dist.constraints.positive)
            b_weibull_c = pyro.param(Params.b_weibull_concentration,
                                     torch.tensor(self.b_weibull, device=self.device),
                                     constraint=dist.constraints.positive)
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        # Local scores Z
        _Z_default = (
            self.joint_scores_init.to(self.device)
            if self.joint_scores_init is not None
            else torch.zeros(self.n, self.k, device=self.device)
        )
        loc_Z = pyro.param(Params.loc_Z, _Z_default)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.ones(self.n, self.k, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            # for l in range(len(X_list)):
            #     pyro.deterministic(Sites.joint_structure_l.format(l=l),
            #                        torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))


    def guide_dense(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        ########################
        # ---- Loadings --------
        ########################
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = pyro.param(Params.a_sigma_lambda_l.format(l=l),
                                        torch.full((self.k,), self.a_sigma_joint, device=self.device),
                                        constraint=dist.constraints.positive)
            b_sigma_lambda = pyro.param(Params.b_sigma_lambda_l.format(l=l),
                                        torch.full((self.k,), self.b_sigma_joint, device=self.device),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

            _Lambda_default = (
                self.joint_loadings_list_init[l].to(self.device)
                if self.joint_loadings_list_init is not None
                else torch.zeros(p_l, self.k, device=self.device)
            )
            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), _Lambda_default)
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.ones(p_l, self.k, device=self.device),
                                        constraint=dist.constraints.positive)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = pyro.param(Params.a_psi_l.format(l=l),
                                 torch.full((p_l,), self.a_psi, device=self.device),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.full((p_l,), self.b_psi, device=self.device),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))

        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.full((self.k,), self.a_sigma_beta, device=self.device),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.full((self.k,), self.b_sigma_beta, device=self.device),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k, device=self.device))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.full((self.k,), self.init_param, device=self.device),
                                constraint=dist.constraints.positive)
        pyro.sample(Sites.beta,
                    dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            a_sigma_y = pyro.param(Params.a_sigma_y,
                                   torch.tensor(self.a_sigma_y, device=self.device),
                                   constraint=dist.constraints.positive)
            b_sigma_y = pyro.param(Params.b_sigma_y,
                                   torch.tensor(self.b_sigma_y, device=self.device),
                                   constraint=dist.constraints.positive)
            pyro.sample(Sites.sigma2_y,
                        dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
        elif self.outcome == "censored":
            a_weibull_c = pyro.param(Params.a_weibull_concentration,
                                     torch.tensor(self.a_weibull, device=self.device),
                                     constraint=dist.constraints.positive)
            b_weibull_c = pyro.param(Params.b_weibull_concentration,
                                     torch.tensor(self.b_weibull, device=self.device),
                                     constraint=dist.constraints.positive)
            pyro.sample(Sites.weibull_concentration,
                        dist.Gamma(a_weibull_c, b_weibull_c))
        else:
            raise NotImplementedError

        # Local scores Z
        _Z_default = (
            self.joint_scores_init.to(self.device)
            if self.joint_scores_init is not None
            else torch.zeros(self.n, self.k, device=self.device)
        )
        loc_Z = pyro.param(Params.loc_Z, _Z_default)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.ones(self.n, self.k, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            # for l in range(len(X_list)):
            #     pyro.deterministic(Sites.joint_structure_l.format(l=l),
            #                        torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))


    def predict_forward(self, X_list, batch_idx, y=None, cens=None):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_forward_dense(X_list, batch_idx)
        else:
            return self.predict_forward_mgp(X_list, batch_idx)

    def predict_guide(self, X_list, batch_idx, y=None, cens=None):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_guide_dense(X_list, batch_idx, y)
        else:
            return self.predict_guide_mgp(X_list, batch_idx, y)


    def predict_forward_mgp(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

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
            Z = pyro.sample(Sites.Z_pred, dist.Normal(self._t(0.), self._t(1.)).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                pyro.sample(Sites.X_l_pred.format(l=l),
                            dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                            obs=X_list[l])

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


    def predict_guide_mgp(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

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

        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k, device=self.device))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            # for l in range(len(X_list)):
            #     pyro.deterministic(Sites.joint_structure_l.format(l=l),
            #                        torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))

            # loc_beta = self.params[Params.loc_beta]
            # pyro.deterministic(Sites.outcome_structure_pred,
            #                    torch.matmul(Z, loc_beta.squeeze(0)))


    def predict_forward_dense(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

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
            Z = pyro.sample(Sites.Z_pred, dist.Normal(self._t(0.), self._t(1.)).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].T))
                pyro.sample(Sites.X_l_pred.format(l=l),
                            dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                            obs=X_list[l])

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


    def predict_guide_dense(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = self.params[Params.a_sigma_lambda_l.format(l=l)]
            b_sigma_lambda = self.params[Params.b_sigma_lambda_l.format(l=l)]
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

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

        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k, device=self.device))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            # for l in range(len(X_list)):
            #     pyro.deterministic(Sites.joint_structure_l.format(l=l),
            #                        torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))

            # loc_beta = self.params[Params.loc_beta]
            # pyro.deterministic(Sites.outcome_structure_pred,
            #                    torch.matmul(Z, loc_beta.squeeze(0)))
