import os
import sys

from functools import partial

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.poutine import mask

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
                 a_sigma_beta=2.0,
                 b_sigma_beta=2.0,
                 joint_scores_init=None,
                 joint_loadings_list_init=None):
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

        self.inv_gamma_init_param = 2.1
        self.init_param = 0.1


    def forward(self, X_list, batch_idx, y=None):
        if self.dense:
            return self.forward_dense(X_list, batch_idx, y)
        else:
            return self.forward_mgp(X_list, batch_idx, y)

    def guide(self, X_list, batch_idx, y=None):
        if self.dense:
            return self.guide_dense(X_list, batch_idx, y)
        else:
            return self.guide_mgp(X_list, batch_idx, y)


    def forward_mgp(self, X_list, batch_idx, y=None):
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
            delta_lambda = []
            for m in range(self.k):
                shape = self.a1_sigma_joint if m == 0 else self.a2_sigma_joint
                delta_l_m = pyro.sample(Sites.delta_lambda_l_k.format(l=l, m=m),
                                        dist.Gamma(shape, 1.0))
                delta_lambda.append(delta_l_m)
            tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim=0).squeeze()

            # rho - local shrinkage
            rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l=l),
                                     dist.Gamma(self.alpha_joint / 2, self.alpha_joint / 2).expand([p_l, self.k]).to_event(2)).squeeze()

            precision = rho_lambda * tau_lambda_k_list
            precision = torch.clamp(precision, min=1e-10, max=1e10)
            sigma_lambda_l = precision.pow_(-0.5)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(torch.zeros(p_l, self.k), sigma_lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        psi_sqrt_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.k]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.k), sigma2_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        else:
            raise NotImplementedError

        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(0., 1.).expand([self.k]).to_event(1))

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
            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y),
                                     obs=y)
                return y_pred
            else:
                raise NotImplementedError


    def forward_dense(self, X_list, batch_idx, y=None):
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
                                          dist.InverseGamma(self.a_sigma_joint, self.b_sigma_joint).expand([self.k]).to_event(1))
            sigma_lambda_l = torch.sqrt(sigma2_lambda_l)
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                                   dist.Normal(torch.zeros(p_l, self.k), sigma_lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        psi_sqrt_l_list = []
        for l, p_l in enumerate(p_l_list):
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(self.a_psi, self.b_psi).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        # Outcome model coefficients
        sigma2_beta = pyro.sample(Sites.sigma2_beta,
                                  dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta).expand([self.k]).to_event(1))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(torch.zeros(self.k), sigma2_beta).to_event(1)).squeeze(0)

        sigma2_y = pyro.sample(Sites.sigma2_y,
                               dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)).squeeze(0)
        sigma_y = torch.sqrt(sigma2_y)

        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z = pyro.sample(Sites.Z, dist.Normal(0., 1.).expand([self.k]).to_event(1))

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
            if self.outcome == "gaussian":
                pyro.sample(Sites.y, dist.Normal(outcome_structure, sigma_y),
                            obs=y)
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
            for m in range(self.k):
                a_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l=l, m=m),
                                                torch.tensor(self.init_param),
                                                constraint=dist.constraints.positive)
                b_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l=l, m=m),
                                                torch.tensor(self.init_param),
                                                constraint=dist.constraints.positive)
                pyro.sample(Sites.delta_lambda_l_k.format(l=l, m=m),
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

            a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l=l),
                                        torch.tensor(self.init_param),
                                        constraint=dist.constraints.positive)
            b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l=l),
                                        torch.tensor(self.init_param),
                                        constraint=dist.constraints.positive)
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()

            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), torch.zeros(p_l, self.k))
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.tensor(self.init_param).expand([p_l, self.k]),
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
                                 torch.tensor(self.inv_gamma_init_param),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.tensor(self.init_param),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))

        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.tensor(self.inv_gamma_init_param),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.tensor(self.init_param),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.tensor(self.init_param).expand([self.k]),
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
        else:
            raise NotImplementedError

        # Local scores Z
        loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        scale_Z = pyro.param(Params.scale_Z,
                             torch.tensor(self.init_param).expand([self.n, self.k]),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            for l in range(len(X_list)):
                pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                   torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))


    def guide_dense(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        ########################
        # ---- Loadings --------
        ########################
        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            a_sigma_lambda = pyro.param(Params.a_sigma_lambda_l.format(l=l),
                                        torch.tensor(self.inv_gamma_init_param))
            b_sigma_lambda = pyro.param(Params.b_sigma_lambda_l.format(l=l),
                                        torch.tensor(self.inv_gamma_init_param))
            pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                        dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).expand([self.k]).to_event(1))

            loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l=l), torch.zeros(p_l, self.k))
            scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l=l),
                                        torch.tensor(self.init_param).expand([p_l, self.k]),
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
                                 torch.tensor(self.inv_gamma_init_param),
                                 constraint=dist.constraints.positive)
            b_psi_l = pyro.param(Params.b_psi_l.format(l=l),
                                 torch.tensor(self.init_param),
                                 constraint=dist.constraints.positive)
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))

        # Outcome model coefficients
        a_sigma_beta = pyro.param(Params.a_sigma_beta,
                                  torch.tensor(self.inv_gamma_init_param),
                                  constraint=dist.constraints.positive)
        b_sigma_beta = pyro.param(Params.b_sigma_beta,
                                  torch.tensor(self.init_param),
                                  constraint=dist.constraints.positive)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).expand([self.k]).to_event(1))

        loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k))
        scale_beta = pyro.param(Params.scale_beta,
                                torch.tensor(self.init_param).expand([self.k]),
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
        else:
            raise NotImplementedError

        # Local scores Z
        loc_Z = pyro.param(Params.loc_Z, torch.zeros(self.n, self.k))
        scale_Z = pyro.param(Params.scale_Z,
                             torch.tensor(self.init_param).expand([self.n, self.k]),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))

            for l in range(len(X_list)):
                pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                   torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))


    def predict_forward(self, X_list, batch_idx, y=None):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_forward_dense(X_list, batch_idx, y)
        else:
            return self.predict_forward_mgp(X_list, batch_idx, y)

    def predict_guide(self, X_list, batch_idx, y=None):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_guide_dense(X_list, batch_idx, y)
        else:
            return self.predict_guide_mgp(X_list, batch_idx, y)


    def predict_forward_mgp(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            for m in range(self.k):
                a_delta_lambda_l_k = self.params[Params.a_delta_lambda_l_k.format(l=l, m=m)]
                b_delta_lambda_l_k = self.params[Params.b_delta_lambda_l_k.format(l=l, m=m)]
                pyro.sample(Sites.delta_lambda_l_k.format(l=l, m=m),
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l=l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l=l)]
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()

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
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

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
        else:
            raise NotImplementedError

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))
                X_l = pyro.sample(Sites.X_l_pred.format(l=l),
                                  dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                   torch.matmul(Z, beta))
            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y),
                                     obs=y)
                return y_pred
            else:
                raise NotImplementedError


    def predict_guide_mgp(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]

        Lambda_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            for m in range(self.k):
                a_delta_lambda_l_k = self.params[Params.a_delta_lambda_l_k.format(l=l, m=m)]
                b_delta_lambda_l_k = self.params[Params.b_delta_lambda_l_k.format(l=l, m=m)]
                pyro.sample(Sites.delta_lambda_l_k.format(l=l, m=m),
                            dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

            a_rho_lambda_l = self.params[Params.a_rho_lambda_l.format(l=l)]
            b_rho_lambda_l = self.params[Params.b_rho_lambda_l.format(l=l)]
            pyro.sample(Sites.rho_lambda_l.format(l=l),
                        dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([p_l, self.k]).to_event(2)).squeeze()

            loc_Lambda_l = self.params[Params.loc_Lambda_l.format(l=l)]
            scale_Lambda_l = self.params[Params.scale_Lambda_l.format(l=l)]
            Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                        dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
            Lambda_l_list.append(Lambda_l)

        for l, p_l in enumerate(self.p_l_list):
            a_psi_l = self.params[Params.a_psi_l.format(l=l)]
            b_psi_l = self.params[Params.b_psi_l.format(l=l)]
            pyro.sample(Sites.psi_l.format(l=l),
                        dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))

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
        else:
            raise NotImplementedError

        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            for l in range(len(X_list)):
                pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                   torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))

            loc_beta = self.params[Params.loc_beta]
            pyro.deterministic(Sites.outcome_structure_pred,
                               torch.matmul(Z, loc_beta.squeeze(0)))


    def predict_forward_dense(self, X_list, batch_idx, y=None):
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
                                dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

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
        else:
            raise NotImplementedError

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            Z = pyro.sample(Sites.Z_pred, dist.Normal(0., 1.).expand([self.k]).to_event(1))

            for l in range(len(X_list)):
                joint_structure_l = pyro.deterministic(Sites.joint_structure_l_pred.format(l=l),
                                                       torch.matmul(Z, Lambda_l_list[l].T))
                X_l = pyro.sample(Sites.X_l_pred.format(l=l),
                                  dist.Normal(joint_structure_l, psi_sqrt_l_list[l]).to_event(1),
                                  obs=X_list[l])
                if torch.isnan(X_l).any():
                    print("NaN values found in X_l_tensor!")
                if torch.isinf(X_l).any():
                    print("Inf values found in X_l_tensor!")

            outcome_structure = pyro.deterministic(Sites.outcome_structure_pred,
                                                   torch.matmul(Z, beta))
            if self.outcome == "gaussian":
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y),
                                     obs=y)
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
                        dist.InverseGamma(a_psi_l, b_psi_l).expand([p_l]).to_event(1))

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
        else:
            raise NotImplementedError

        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            Z = pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
            
            for l in range(len(X_list)):
                pyro.deterministic(Sites.joint_structure_l.format(l=l),
                                   torch.matmul(Z, Lambda_l_list[l].squeeze(0).T))

            loc_beta = self.params[Params.loc_beta]
            pyro.deterministic(Sites.outcome_structure_pred,
                               torch.matmul(Z, loc_beta.squeeze(0)))
