import os
import sys
# import pyreadr

import torch
import pyro
import pyro.distributions as dist

pyro.enable_validation(True)
pyro.set_rng_seed(0)

def supervised_factor_model(X, # Input
                            k, # Latent dimension
                            a_sigma=2.0, # Hyperparams: ARD prior per factor: IG
                            b_sigma=2.0,
                            a_psi=2.0, # Hyperparams: per-view error: IG
                            b_psi=2.0):
    """
    Bayesian matrix decomposition with ARD prior
    Inference performed using SVI with minibatching
    
    X = Z @ Lambda^T + E
    X_i ~ N_p(Lambda @ Z_i, diag(psi))
    """
    
    n, p = X.shape
    
    ########################
    # ---- Loadings --------
    # Loadings Lambda: sample rows across features (p, k)
    # ARD prior
    # Lambda_j ~ N_k(0, sigma^2 I)
    # sigma^2 ~ InvGamma(a_sigma, b_sigma))
    ########################
    
    with pyro.plate("")
    
    with pyro.plate("features", p):
        # sample a k-dim row for each feature; .to_event(1) marks k dims as event
        lambda_row = pyro.sample("Lambda_row", dist.Normal(torch.zeros(k), sigma).to_event(1))
    # lambda_row has shape (p, k)