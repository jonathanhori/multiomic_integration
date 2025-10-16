# # !/usr/bin/env python

import os
import sys
import pyreadr

import torch
import pyro
import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# from pyro.infer.autoguide import AutoMultivariateNormal
# from pyro.optim import Adam
# from torch.utils.data import TensorDataset, DataLoader

pyro.enable_validation(True)
pyro.set_rng_seed(0)

def supervised_factor_model(X, y, k, 
                            a_sigma=2.0, # Hyperparams: ARD prior per factor: IG
                            b_sigma=2.0,
                            a_psi=2.0, # Hyperparams: per-view error: IG
                            b_psi=2.0,
                            sigma_beta=1.0, # Hyperparams: outcome model coef variance: N
                            a_y = 2.0, # Hyperparams: outcome model error : IG
                            b_y = 2.0):
    """
    Pyro model for supervised factor analysis with ARD shrinkage on columns of Lambda.
    X: (n, p) torch.tensor
    y: (n,) torch.tensor (continuous)
    k: number of latent factors
    """
    n, p = X.shape

    # ---- priors on factor-specific scales (ARD) ----
    with pyro.plate("factors", k):
        sigma2_h = pyro.sample("sigma2_h", dist.InverseGamma(a_sigma, b_sigma))
        sigma_h = torch.sqrt(sigma2_h)  # standard deviation per-factor (shape: k)

    # ---- loadings Lambda: shape (p, k) ----
    # We sample columns of Lambda using the corresponding sigma_h
    with pyro.plate("features", p):
        # For vectorized broadcasting: sample a k-dim normal with per-dim std sigma_h
        lambda_j = pyro.sample("Lambda_row",
                                dist.Normal(torch.zeros(k), sigma_h).to_event(1))
        # Lambda_row has shape (p, k) across the plate

    # ---- idiosyncratic variances psi_j for each feature ----
    with pyro.plate("features_psi", p):
        psi_j = pyro.sample("psi_j", dist.InverseGamma(a_psi, b_psi))

    # ---- regression coefficients for y ----
    beta = pyro.sample("beta", dist.Normal(torch.zeros(k), sigma_beta * torch.ones(k)).to_event(1))
    sigma2_y = pyro.sample("sigma2_y", dist.InverseGamma(a_y, b_y))
    sigma_y = torch.sqrt(sigma2_y)

    # ---- latent factors Z for each individual ----
    # We'll treat Z as local latent variables that we sample per minibatch below
    # To support minibatching we will use a Plate over data points when we observe X and y

    # Note: do not sample Z globally here; we'll do it inside the plate for observed data
    Lambda = pyro.deterministic("Lambda", lambda_j)   # shape (p, k)
    Psi = pyro.deterministic("Psi", psi_j)           # shape (p,)

    # Observations: plate over individuals
    with pyro.plate("data", n, dim=-2):
        # latent factor scores for each individual
        Z = pyro.sample("Z", dist.Normal(torch.zeros(k), torch.ones(k)).to_event(1))  # (n, k)

        # observation model for X_i: X_i ~ N(Z_i @ Lambda.T, diag(Psi))
        mu_X = (Z.unsqueeze(1) @ Lambda.unsqueeze(0).transpose(-1, -2)).squeeze(1)  # (n, p)
        # Pyro expects event_dim=1 for multivariate normal across p dims
        pyro.sample("X_obs", dist.Normal(mu_X, torch.sqrt(Psi)).to_event(1), obs=X)

        # response model y_i ~ N(Z_i @ beta, sigma_y^2)
        mu_y = (Z @ beta)
        pyro.sample("y_obs", dist.Normal(mu_y, sigma_y), obs=y)
        
        
sim_data = pyreadr.read_r('~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/data/n500p100_snr1.1/sim_data_ywithview_rep1.rds')

import pandas as pd

print(sim_data.keys())
sim_data = sim_data[None]

# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# readRDS = robjects.r['readRDS']
# sim_data = readRDS('~/Library/Mobile Documents/com~apple~CloudDocs/Projects/multiomic_integration/sim/data/n500p100_snr1.1/sim_data_ywithview_rep1.rds')
        
dir(sim_data)

sim_data.names

# Outcome
y = sim_data.rx2("y")
y = torch.tensor(y)

# Multiview data


[print(torch.tensor(X_l).shape) for X_l in sim_data.rx2("X_l")]
[X_l for X_l in sim_data.rx2("X_l")]

X_l_array = sim_data.rx2("X_l")

dir(X_l_array[1])
X_l_array

[print(name) for name in sim_data.names]

py_list = [pandas2ri.rpy2py(df) for df in sim_data]
py_list

for i, j in sim_data.items():
    if i == "X_l":
        print(i, j)
    
sim_data.items()

pyro.render_model(supervised_factor_model, render_distributions=True, render_params=True)