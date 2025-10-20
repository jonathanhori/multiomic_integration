import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal

from torch.utils.data import TensorDataset, DataLoader

pyro.enable_validation(True)
pyro.set_rng_seed(0)

def bayesian_factor_model(X, # Input
                            k, # Latent dimension
                            n, # Total number of data observations
                            a_sigma=2.0, # Hyperparams: ARD prior per factor: IG
                            b_sigma=2.0,
                            a_psi=2.0, # Hyperparams: per-view error: IG
                            b_psi=2.0):
    """
    Bayesian matrix decomposition with ARD prior on factor loadings
    Inference performed using SVI with minibatching
    
    X = Z @ Lambda^T + E
    X_i ~ N_p(Lambda @ Z_i, diag(psi))
    """
    # X is TensorDataset
    # m = len(X)
    # p = X[0][0].shape[0]
    m, p = X.shape # Working with minibatches of size m
    
    ########################
    # ---- Loadings --------
    # Loadings Lambda: sample rows across features (p, k)
    # ARD prior
    # Lambda_j ~ N_k(0, sigma_k^2 I)
    #   ==> Lambda_jk ~ N(0, sigma_k^2)
    # sigma_k^2 ~ InvGamma(a_sigma, b_sigma))
    ########################
    
    # Variance of each loading
    # with pyro.plate("loading_var", k):
    #     sigma2 = pyro.sample("sigma2", dist.InverseGamma(a_sigma, b_sigma))    
    # sigma = torch.sqrt(sigma2)
    
    # with pyro.plate("loadings", p):
    #     # sample a k-dim row for each feature
    #     # .to_event(1) marks k dims as event (k-dimensional sample ("batch" in pyro))
    #     Lambda = pyro.sample("Lambda", dist.Normal(torch.zeros(p, k), sigma).to_event(2))
    
    # No row-wise conditional independence of loadings. We remove to_event() for now. Possibly add back depending on guides during optimization
    sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma, b_sigma).expand([k]).to_event(1))
    sigma = torch.sqrt(sigma2)
    
    # sigma is broadcast across rows of Lambda
    Lambda = pyro.sample("Lambda", dist.Normal(torch.zeros(p, k), sigma).to_event(2))
    
    
    ########################
    # ---- Observations --------
    # Working with minibatches of data (subsamples of rows of X)
    # The plate statement defines conditional independence over each observation
    # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
    #   function to perform minibatching. 
    ########################
    
    # Idiosyncratic error variance
    # with pyro.plate("psi", p):
    #     psi = pyro.sample("psi_j", dist.InverseGamma(a_psi, b_psi))
        
    psi = pyro.sample("psi", dist.InverseGamma(a_psi, b_psi).expand([p]).to_event(1))
    psi_sqrt = torch.sqrt(psi)
    
    # Local latent variables and observations
    with pyro.plate("obs", n, subsample = X): # X is a minibatch, can pass directly into subsample
        # Latent scores Z_i are local
        Z_batch = pyro.sample("Z_batch", dist.Normal(torch.zeros(m, k), torch.ones(k)).to_event(1))
        
        # Compute structure
        structure = torch.matmul(Z_batch, Lambda.T)
        
        X_batch = pyro.sample("X_batch", dist.Normal(structure, psi_sqrt).to_event(1), obs = X)
        
    
#TODO define guide
def bayesian_factor_guide(X, k, n):
    pass

def train_decomp(X, k, 
                 epochs = 20,
                 minibatch_size = 32,
                 tol = 1e-3,
                 device = "cpu"):
    # X = X.to(device)
    n, p = X.shape
    
    
    ########################
    # Handle minibatching of dataset
    ########################
    X_mod = TensorDataset(X)
    loader = DataLoader(X_mod, batch_size = minibatch_size, shuffle = True)
    
    ########################
    # CURRENTLY: model takes multiple arguments. User helper functions to load data and arguments 
    #   into model and guide. Essentially partial functions
    # TODO: create class to store hyperparams and arguments
    ########################
    def model_batch(batch):
        # return partial(bayesian_factor_model, k = k, n = n)
        return bayesian_factor_model(batch, k = k, n = n)
    # def guide_batch(batch):
    #     # return bayesian_factor_guide(batch, k, n)
    #     # return AutoContinuous(bayesian_factor_model) #
    #     return AutoContinuous(model_batch)
    #     # return AutoContinuous
    
    ########################
    # Initialize instances for optimization
    ########################
    # mod = bayesian_factor_model(X_mod, k, n)
    # guide = AutoNormal(model_batch)
    guide = AutoNormal(lambda batch: model_batch(batch))
    # mod = partial(bayesian_factor_model, k = k, n = n)
    # guide = guide_batch(mod)
    # guide = bayesian_factor_guide()
    opt = Adam({"lr": 0.001})
    elbo = Trace_ELBO()
    # svi = SVI(model_batch, guide_batch, opt, loss = elbo)
    svi = SVI(model_batch, guide, opt, loss = elbo)
    # svi = SVI(mod, guide, opt, loss = elbo)
    
    
    ########################
    # Train
    ########################
    prev_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.
        for batch, in loader:
            # print(batch.shape)
            loss = svi.step(batch)
            # print(loss)
            epoch_loss += loss
        print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / n:.4f}")
        
        if prev_loss is not None and abs(epoch_loss - prev_loss) < tol:
            break
            
        # print(f"delta: {epoch_loss / n - (prev_loss / n if prev_loss is not None else epoch_loss / n):+.6f}")
        
        # for name in pyro.get_param_store().get_all_param_names():
        #     p = pyro.param(name)
        #     print(name, p.shape, getattr(p, "grad", None) is not None)
        
        prev_loss = loss
    # for epoch in range(epochs):
    #     epoch_loss = 0.
    #     loss = svi.step(X_mod) #batch[0])
    #     print(loss)
    #     epoch_loss += loss
    #     # for batch in loader:
    #     #     loss = svi.step(batch[0])
    #     #     epoch_loss += loss
    #     print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss:.4f}")
        
    return svi#, guide
    # return svi, guide