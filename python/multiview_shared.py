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
    y   = Z @ beta + e                                   outcome="gaussian"
    log T = beta_0 + Z @ beta + sigma_y e, right-censored  outcome="lognormal_aft"
            (y = standardised log follow-up time, event = 1 observed / 0 censored)
    T ~ Weibull(exp(beta_0 + Z @ beta), k), right-censored outcome="censored"

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
                 init_scale_Lambda=1.0,
                 init_scale_Z=1.0,
                 init_param=1.0,
                 conjugate_shapes=True,
                 sigma_intercept=10.0,
                 intercept_init=0.0,
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
        # Censored outcomes ("censored" Weibull AFT, "lognormal_aft") only: intercept
        # beta_0 ~ N(0, sigma_intercept^2)
        # in the log scale, log lambda_i = beta_0 + z_i' beta. intercept_init is the
        # initial guide mean (log of a typical survival time is a good choice).
        self.sigma_intercept = sigma_intercept
        self.intercept_init = intercept_init

        # Weight on the outcome log-likelihood (a tempered posterior that counts
        # each y_i outcome_weight times). Keep the default 1.0, the correctly
        # specified posterior. Weights above 1 lift a factor that sits just below
        # the detection threshold sqrt(P/n), but they narrow the beta intervals
        # by about sqrt(w) and let training scores absorb y, so test-time 95%
        # coverage falls to about 0.74 at w ~ 2 and 0.5 at w ~ 5, with no reliable
        # gain in CV MSE (docs/supervision_regimes.md, section 6). Do not scale
        # it with L * p_l.
        self.outcome_weight = outcome_weight

        self.joint_scores_init = joint_scores_init
        self.joint_loadings_list_init = joint_loadings_list_init
        # Initial guide standard deviations of Lambda_l and Z (training guide only).
        # The defaults (1) give sampled reconstructions ~k units of noise variance
        # on unit-variance data; smaller values start the guide at low variance.
        self.init_scale_Lambda = init_scale_Lambda
        self.init_scale_Z = init_scale_Z

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
        # Initial shape and rate of the MGP Gamma guides (delta, rho) and initial
        # guide sd of beta. 0.1 was the value before 27 May 2026 (commit 49d239a).
        self.init_param = init_param
        # MGP guide only. True (default since November 2026): the shapes of the five
        # conjugate guide factors q(delta), q(rho), q(psi), q(sigma2_y), q(sigma2_beta)
        # are set to their closed-form mean-field optima and never trained (they are
        # kept in the param store but used detached, so they get no gradient); their
        # rates start so that the expectations the other factors see equal those of
        # the old start (E[delta] = E[rho] = 1, E[1/s] = a_prior / b_prior for the
        # inverse-gamma variances) and are trained. Holding these shapes fixed leaves
        # the ELBO's stationary points unchanged; SVI otherwise barely moves them, and
        # q(delta) started at Gamma(1, 1) let the global precision tau collapse at large
        # k. Derivation: docs/conjugate_guide_shapes.tex. False: the old start (Gamma
        # guides at init_param, inverse-gamma guides at the prior, all trained).
        self.conjugate_shapes = conjugate_shapes

    def _conj_param(self, name_a, name_b, a_closed, b_closed, a_old, b_old):
        """Shape and rate of a conjugate guide factor (see conjugate_shapes)."""
        pos = dist.constraints.positive
        if not self.conjugate_shapes:
            return (pyro.param(name_a, a_old, constraint=pos), pyro.param(name_b, b_old, constraint=pos))
        a = pyro.param(name_a, a_closed, constraint=pos).detach()
        return a, pyro.param(name_b, b_closed, constraint=pos)

    def _t(self, x):
        """Scalar float → device-placed tensor (avoids CPU/MPS log_prob mismatch)."""
        return torch.tensor(float(x), device=self.device)


    # ------------------------------------------------------------------------------
    # Structure (November 2026 refactor; no change to the model or the guide):
    #
    #   model, training      forward_{dense,mgp}         = loadings prior + _model_globals_and_obs
    #   guide, training      guide_{dense,mgp}           = _guide_globals(fitted=False) + _guide_scores
    #   model, prediction    predict_forward_{dense,mgp} = _guide_globals(fitted=True) + _predict_obs
    #   guide, prediction    predict_guide_{dense,mgp}   = _guide_globals(fitted=True) + _guide_scores
    #
    # The prior on the loadings (dense / MGP) is in _loadings_prior_{dense,mgp} and its
    # guide in _loadings_guide_{dense,mgp}; everything else is shared. Sample sites are
    # drawn in the same order as before the refactor (it sets the random-number stream);
    # python/tests/test_shared_refactor_equivalence.py checks bit-identical results
    # against the pre-refactor copy in python/tests/legacy_multiview_shared.py.
    # ------------------------------------------------------------------------------

    def forward(self, X_list, batch_idx, y=None, event=None):
        if self.dense:
            return self.forward_dense(X_list, batch_idx, y, event)
        else:
            return self.forward_mgp(X_list, batch_idx, y, event)

    def guide(self, X_list, batch_idx, y=None, event=None):
        if self.dense:
            return self.guide_dense(X_list, batch_idx, y)
        else:
            return self.guide_mgp(X_list, batch_idx, y)


    def forward_mgp(self, X_list, batch_idx, y=None, event=None):
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

        Lambda_l_list = [self._loadings_prior_mgp(l, p_l) for l, p_l in enumerate(self.p_l_list)]
        return self._model_globals_and_obs(Lambda_l_list, self.p_l_list, X_list, batch_idx, y, event)


    def forward_dense(self, X_list, batch_idx, y=None, event=None):
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

        Lambda_l_list = [self._loadings_prior_dense(l, p_l) for l, p_l in enumerate(p_l_list)]
        self._model_globals_and_obs(Lambda_l_list, p_l_list, X_list, batch_idx, y, event)


    # ---- Model: prior on the loadings --------------------------------------------

    def _loadings_prior_mgp(self, l, p_l):
        ########################
        # ---- Loadings --------
        # Lambda^l_jk ~ N(0, (rho^l_jk)^-1 * (tau^l_k)^-1)
        ########################
        # tau - global shrinkage
        shape_vec = torch.cat([
            torch.tensor([self.a1_sigma_joint], device=self.device),
            torch.full((self.k - 1,), self.a2_sigma_joint, device=self.device)
        ])
        delta_lambda = pyro.sample(Sites.delta_lambda_l.format(l=l),
                                   dist.Gamma(shape_vec, torch.ones(self.k, device=self.device)).to_event(1))
        # tau_k = prod_{h<=k} delta_h, computed in log space: with large k the
        # product under- or overflows in float32 and the gradients become NaN.
        log_tau_lambda = torch.cumsum(torch.log(delta_lambda.clamp(min=1e-30)), dim=-1)

        # rho - local shrinkage
        rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l=l),
                                 dist.Gamma(self._t(self.alpha_joint / 2), self._t(self.alpha_joint / 2)).expand([p_l, self.k]).to_event(2)).squeeze()

        # precision = rho * tau, clamped to [1e-10, 1e10]
        log_precision = torch.log(rho_lambda.clamp(min=1e-30)) + log_tau_lambda
        log_precision = torch.clamp(log_precision, min=-23.03, max=23.03)
        sigma_lambda_l = torch.exp(-0.5 * log_precision)
        Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                               dist.Normal(torch.zeros(p_l, self.k, device=self.device), sigma_lambda_l).to_event(2))
        return Lambda_l

    def _loadings_prior_dense(self, l, p_l):
        ########################
        # ---- Loadings --------
        # ARD prior: Lambda^l_jk ~ N(0, sigma^l_k^2)
        # sigma^l_k^2 ~ InvGamma(a_sigma, b_sigma)
        ########################
        sigma2_lambda_l = pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                                      dist.InverseGamma(self._t(self.a_sigma_joint), self._t(self.b_sigma_joint)).expand([self.k]).to_event(1))
        sigma_lambda_l = torch.sqrt(sigma2_lambda_l)
        Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                               dist.Normal(torch.zeros(p_l, self.k, device=self.device), sigma_lambda_l).to_event(2))
        return Lambda_l


    # ---- Model: other global priors, scores and likelihood (training) ---------------

    def _model_globals_and_obs(self, Lambda_l_list, p_l_list, X_list, batch_idx, y, event):
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
                           dist.Normal(torch.zeros(self.k, device=self.device), torch.sqrt(sigma2_beta)).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(self._t(self.a_sigma_y), self._t(self.b_sigma_y))).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.Gamma(self._t(self.a_weibull), self._t(self.b_weibull)))
            intercept = pyro.sample(Sites.intercept,
                                    dist.Normal(self._t(0.), self._t(self.sigma_intercept)))
        elif self.outcome == "lognormal_aft":
            # log T = beta_0 + z'beta + sigma_y e: the Gaussian model on log time, plus an
            # intercept, with right-censored samples. y is the (standardised) log time.
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(self._t(self.a_sigma_y), self._t(self.b_sigma_y))).squeeze(0)
            sigma_y = torch.sqrt(sigma2_y)
            intercept = pyro.sample(Sites.intercept,
                                    dist.Normal(self._t(0.), self._t(self.sigma_intercept)))
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
                    weibull_scale = torch.exp(intercept + outcome_structure).clamp(min=1e-10)
                    with mask(mask=(event == 1)):
                        pyro.sample(Sites.y, dist.Weibull(weibull_scale, weibull_concentration), obs=y)
                    log_surv = -torch.pow(y / weibull_scale, weibull_concentration)
                    with mask(mask=(event == 0)):
                        pyro.factor(Sites.censored, log_surv)
                elif self.outcome == "lognormal_aft":
                    # events: log density of the log time; censored: log P(log T > log y)
                    # (the -log y Jacobian of the events is constant and left out)
                    mu = intercept + outcome_structure
                    with mask(mask=(event == 1)):
                        pyro.sample(Sites.y, dist.Normal(mu, sigma_y), obs=y)
                    with mask(mask=(event == 0)):
                        pyro.factor(Sites.censored, torch.special.log_ndtr(-(y - mu) / sigma_y))
                else:
                    raise NotImplementedError


    # ---- Guide (training) ------------------------------------------------------------

    def guide_mgp(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        self._guide_globals(dense=False, fitted=False)
        self._guide_scores_train(batch_idx)

    def guide_dense(self, X_list, batch_idx, y=None):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        self._guide_globals(dense=True, fitted=False)
        self._guide_scores_train(batch_idx)

    def _guide_scores_train(self, batch_idx):
        # Local scores Z
        _Z_default = (
            self.joint_scores_init.to(self.device)
            if self.joint_scores_init is not None
            else torch.zeros(self.n, self.k, device=self.device)
        )
        loc_Z = pyro.param(Params.loc_Z, _Z_default)
        scale_Z = pyro.param(Params.scale_Z,
                             torch.full((self.n, self.k), self.init_scale_Z, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs", self.n, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            pyro.sample(Sites.Z, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))


    # ---- Guide factors of the global parameters --------------------------------------
    # fitted=False: training guide, the variational parameters are pyro.params (MGP:
    #   conjugate factors through _conj_param; dense: all trained from the prior values).
    # fitted=True: prediction, the fitted values are read from self.params (frozen).

    def _shape_rate(self, name_a, name_b, fitted, conj, inits):
        """Shape and rate of a Gamma / inverse-gamma guide factor.
        inits() -> (a_closed, b_closed, a_old, b_old), evaluated only when training."""
        if fitted:
            return self.params[name_a], self.params[name_b]
        a_closed, b_closed, a_old, b_old = inits()
        if conj:
            return self._conj_param(name_a, name_b, a_closed, b_closed, a_old, b_old)
        pos = dist.constraints.positive
        return (pyro.param(name_a, a_old, constraint=pos), pyro.param(name_b, b_old, constraint=pos))

    def _loc_scale(self, name_loc, name_scale, fitted, loc_init, scale_init):
        """Location and scale of a Normal guide factor (inits are callables)."""
        if fitted:
            return self.params[name_loc], self.params[name_scale]
        return (pyro.param(name_loc, loc_init()),
                pyro.param(name_scale, scale_init(), constraint=dist.constraints.positive))

    def _loadings_guide_mgp(self, l, p_l, fitted):
        # q(delta_h): closed-form shape a_prior_h + p_l (k - h + 1) / 2, rate = shape
        def delta_inits():
            a0 = torch.full((self.k,), self.a2_sigma_joint, device=self.device)
            a0[0] = self.a1_sigma_joint
            a_cf = a0 + p_l * (self.k - torch.arange(self.k, device=self.device, dtype=a0.dtype)) / 2
            old = torch.full((self.k,), self.init_param, device=self.device)
            return a_cf, a_cf.clone(), old, old.clone()
        a_delta_lambda_l, b_delta_lambda_l = self._shape_rate(
            Params.a_delta_lambda_l.format(l=l), Params.b_delta_lambda_l.format(l=l),
            fitted, True, delta_inits)
        pyro.sample(Sites.delta_lambda_l.format(l=l),
                    dist.Gamma(a_delta_lambda_l, b_delta_lambda_l).to_event(1))

        # q(rho_jk): closed-form shape alpha / 2 + 1 / 2, rate = shape
        def rho_inits():
            a_cf = torch.full((p_l, self.k), self.alpha_joint / 2 + 0.5, device=self.device)
            old = torch.full((p_l, self.k), self.init_param, device=self.device)
            return a_cf, a_cf.clone(), old, old.clone()
        a_rho_lambda_l, b_rho_lambda_l = self._shape_rate(
            Params.a_rho_lambda_l.format(l=l), Params.b_rho_lambda_l.format(l=l),
            fitted, True, rho_inits)
        pyro.sample(Sites.rho_lambda_l.format(l=l),
                    dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).to_event(2)).squeeze()

        loc_Lambda_l, scale_Lambda_l = self._loc_scale(
            Params.loc_Lambda_l.format(l=l), Params.scale_Lambda_l.format(l=l), fitted,
            lambda: self._Lambda_init(l, p_l),
            lambda: torch.full((p_l, self.k), self.init_scale_Lambda, device=self.device))
        return pyro.sample(Sites.Lambda_l.format(l=l),
                           dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))

    def _loadings_guide_dense(self, l, p_l, fitted):
        a_sigma_lambda, b_sigma_lambda = self._shape_rate(
            Params.a_sigma_lambda_l.format(l=l), Params.b_sigma_lambda_l.format(l=l),
            fitted, False,
            lambda: (None, None,
                     torch.full((self.k,), self.a_sigma_joint, device=self.device),
                     torch.full((self.k,), self.b_sigma_joint, device=self.device)))
        pyro.sample(Sites.sigma2_lambda_l.format(l=l),
                    dist.InverseGamma(a_sigma_lambda, b_sigma_lambda).to_event(1))

        loc_Lambda_l, scale_Lambda_l = self._loc_scale(
            Params.loc_Lambda_l.format(l=l), Params.scale_Lambda_l.format(l=l), fitted,
            lambda: self._Lambda_init(l, p_l),
            lambda: torch.full((p_l, self.k), self.init_scale_Lambda, device=self.device))
        Lambda_l = pyro.sample(Sites.Lambda_l.format(l=l),
                               dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
        # The dense prediction model squeezed the sampled loadings (a no-op unless p_l = 1).
        return Lambda_l.squeeze(0) if fitted else Lambda_l

    def _Lambda_init(self, l, p_l):
        return (
            self.joint_loadings_list_init[l].to(self.device)
            if self.joint_loadings_list_init is not None
            else torch.zeros(p_l, self.k, device=self.device)
        )

    def _guide_globals(self, dense, fitted):
        """Guide factors of all global parameters, in the model's site order.
        Returns (Lambda_l_list, psi_sqrt_l_list, beta, outcome_param), where
        outcome_param is sigma_y (gaussian), (Weibull concentration, intercept) (censored)
        or (sigma_y, intercept) (lognormal_aft)."""
        conj = not dense   # the MGP guide uses closed-form conjugate shapes; dense does not

        ########################
        # ---- Loadings --------
        ########################
        loadings_guide = self._loadings_guide_dense if dense else self._loadings_guide_mgp
        Lambda_l_list = [loadings_guide(l, p_l, fitted) for l, p_l in enumerate(self.p_l_list)]

        ########################
        # ---- Observations ----
        ########################

        # Idiosyncratic error variance
        psi_sqrt_l_list = []
        for l, p_l in enumerate(self.p_l_list):
            # q(psi_j): closed-form shape a_psi + n / 2; rate keeps E[1/psi] = a_psi / b_psi
            def psi_inits(p_l=p_l):
                a_cf = torch.full((p_l,), self.a_psi + self.n / 2, device=self.device)
                return (a_cf, a_cf * self.b_psi / self.a_psi,
                        torch.full((p_l,), self.a_psi, device=self.device),
                        torch.full((p_l,), self.b_psi, device=self.device))
            a_psi_l, b_psi_l = self._shape_rate(Params.a_psi_l.format(l=l), Params.b_psi_l.format(l=l),
                                                fitted, conj, psi_inits)
            psi_l = pyro.sample(Sites.psi_l.format(l=l),
                                dist.InverseGamma(a_psi_l, b_psi_l).to_event(1))
            psi_sqrt_l_list.append(torch.sqrt(psi_l))

        # Outcome model coefficients
        # q(sigma2_beta_k): closed-form shape a_sigma_beta + 1 / 2; rate keeps E[1/sigma2_beta]
        def sigma_beta_inits():
            a_cf = torch.full((self.k,), self.a_sigma_beta + 0.5, device=self.device)
            return (a_cf, a_cf * self.b_sigma_beta / self.a_sigma_beta,
                    torch.full((self.k,), self.a_sigma_beta, device=self.device),
                    torch.full((self.k,), self.b_sigma_beta, device=self.device))
        a_sigma_beta, b_sigma_beta = self._shape_rate(Params.a_sigma_beta, Params.b_sigma_beta,
                                                      fitted, conj, sigma_beta_inits)
        pyro.sample(Sites.sigma2_beta,
                    dist.InverseGamma(a_sigma_beta, b_sigma_beta).to_event(1))

        loc_beta, scale_beta = self._loc_scale(
            Params.loc_beta, Params.scale_beta, fitted,
            lambda: torch.zeros(self.k, device=self.device),
            lambda: torch.full((self.k,), self.init_param, device=self.device))
        beta = pyro.sample(Sites.beta,
                           dist.Normal(loc_beta, scale_beta).to_event(1)).squeeze(0)

        if self.outcome == "gaussian":
            # q(sigma2_y): closed-form shape a_sigma_y + w n / 2; rate keeps E[1/sigma2_y]
            def sigma_y_inits():
                a_cf = torch.tensor(self.a_sigma_y + self.outcome_weight * self.n / 2, device=self.device)
                return (a_cf, a_cf * self.b_sigma_y / self.a_sigma_y,
                        torch.tensor(self.a_sigma_y, device=self.device),
                        torch.tensor(self.b_sigma_y, device=self.device))
            a_sigma_y, b_sigma_y = self._shape_rate(Params.a_sigma_y, Params.b_sigma_y,
                                                    fitted, conj, sigma_y_inits)
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            outcome_param = torch.sqrt(sigma2_y)
        elif self.outcome == "censored":
            # q(k): LogNormal, started narrow at k = 1. A Gamma(a_weibull, b_weibull)
            # guide start is so wide that early draws of large k give huge
            # (y / scale)^k gradients (NaN under Adam, a frozen AdagradRMSProp
            # accumulator); the model prior stays Gamma(a_weibull, b_weibull).
            loc_log_k, scale_log_k = self._loc_scale(
                Params.loc_log_weibull_concentration, Params.scale_log_weibull_concentration, fitted,
                lambda: torch.tensor(0., device=self.device),
                lambda: torch.tensor(0.1, device=self.device))
            weibull_concentration = pyro.sample(Sites.weibull_concentration,
                                                dist.LogNormal(loc_log_k, scale_log_k))
            loc_intercept, scale_intercept = self._loc_scale(
                Params.loc_intercept, Params.scale_intercept, fitted,
                lambda: torch.tensor(float(self.intercept_init), device=self.device),
                lambda: torch.tensor(0.1, device=self.device))
            intercept = pyro.sample(Sites.intercept, dist.Normal(loc_intercept, scale_intercept))
            outcome_param = (weibull_concentration, intercept)
        elif self.outcome == "lognormal_aft":
            # q(sigma2_y): with censoring there is no closed-form shape, so shape and rate
            # are both trained (also under the MGP prior); they start at the uncensored
            # closed-form values a_sigma_y + w n / 2 and a * b_sigma_y / a_sigma_y.
            def sigma_y_inits():
                a_cf = torch.tensor(self.a_sigma_y + self.outcome_weight * self.n / 2, device=self.device)
                return None, None, a_cf, a_cf * self.b_sigma_y / self.a_sigma_y
            a_sigma_y, b_sigma_y = self._shape_rate(Params.a_sigma_y, Params.b_sigma_y,
                                                    fitted, False, sigma_y_inits)
            sigma2_y = pyro.sample(Sites.sigma2_y,
                                   dist.InverseGamma(a_sigma_y, b_sigma_y)).squeeze(0)
            loc_intercept, scale_intercept = self._loc_scale(
                Params.loc_intercept, Params.scale_intercept, fitted,
                lambda: torch.tensor(float(self.intercept_init), device=self.device),
                lambda: torch.tensor(0.1, device=self.device))
            intercept = pyro.sample(Sites.intercept, dist.Normal(loc_intercept, scale_intercept))
            outcome_param = (torch.sqrt(sigma2_y), intercept)
        else:
            raise NotImplementedError

        return Lambda_l_list, psi_sqrt_l_list, beta, outcome_param


    # ---- Prediction (local inference for new observations) ---------------------------

    def predict_forward(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_forward_dense(X_list, batch_idx)
        else:
            return self.predict_forward_mgp(X_list, batch_idx)

    def predict_guide(self, X_list, batch_idx):
        assert self.params is not None, "Param dict is None: has training inference been run first?"
        if self.dense:
            return self.predict_guide_dense(X_list, batch_idx)
        else:
            return self.predict_guide_mgp(X_list, batch_idx)

    def predict_forward_mgp(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        g = self._guide_globals(dense=False, fitted=True)
        return self._predict_obs(*g, X_list, batch_idx)

    def predict_forward_dense(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        g = self._guide_globals(dense=True, fitted=True)
        return self._predict_obs(*g, X_list, batch_idx)

    def predict_guide_mgp(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        self._guide_globals(dense=False, fitted=True)
        self._guide_scores_predict(batch_idx)

    def predict_guide_dense(self, X_list, batch_idx):
        if self.p_l_list is None:
            self.p_l_list = [X_l.shape[1] for X_l in X_list]
        self._guide_globals(dense=True, fitted=True)
        self._guide_scores_predict(batch_idx)

    def _predict_obs(self, Lambda_l_list, psi_sqrt_l_list, beta, outcome_param, X_list, batch_idx):
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
                sigma_y = outcome_param
                y_pred = pyro.sample(Sites.y_pred, dist.Normal(outcome_structure, sigma_y))
                return y_pred
            elif self.outcome == "censored":
                # The outcome is not part of local inference: Z_pred is inferred from the
                # test views alone. A latent pyro.sample of y_pred would add its
                # log-density to the local ELBO; for a reparameterised Weibull draw that
                # term is -log(scale) + const, which pushes z'beta to -inf (scale and
                # y_pred underflow, NaN scores). So y_pred is drawn outside the ELBO and
                # recorded with pyro.deterministic (log-density 0, no gradient), which
                # Predictive still returns as a posterior predictive draw.
                weibull_concentration, intercept = outcome_param
                log_scale = intercept + outcome_structure
                with torch.no_grad():
                    y_draw = dist.Weibull(torch.exp(log_scale).clamp(min=1e-10),
                                          weibull_concentration).sample()
                return pyro.deterministic(Sites.y_pred, y_draw)
            elif self.outcome == "lognormal_aft":
                # predictive standardised log time; drawn outside the local ELBO as for
                # "censored", so the test scores come from the views alone
                sigma_y, intercept = outcome_param
                with torch.no_grad():
                    y_draw = dist.Normal(intercept + outcome_structure, sigma_y).sample()
                return pyro.deterministic(Sites.y_pred, y_draw)
            else:
                raise NotImplementedError

    def _guide_scores_predict(self, batch_idx):
        loc_Z = pyro.param(Params.loc_Z_pred, torch.zeros(self.n_predict, self.k, device=self.device))
        scale_Z = pyro.param(Params.scale_Z_pred, torch.ones(self.n_predict, self.k, device=self.device),
                             constraint=dist.constraints.positive)

        with pyro.plate("obs_pred", self.n_predict, subsample=batch_idx):
            loc_Z_batch = loc_Z[batch_idx]
            scale_Z_batch = scale_Z[batch_idx]
            pyro.sample(Sites.Z_pred, dist.Normal(loc_Z_batch, scale_Z_batch).to_event(1))
