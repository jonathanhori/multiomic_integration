import os
import sys
# import pyreadr

from functools import partial

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO
from pyro.infer.autoguide import AutoContinuous, AutoNormal #AutoMultivariateNormal

from torch.utils.data import TensorDataset, DataLoader

pyro.enable_validation(True)
pyro.set_rng_seed(0)


class FactorModelARD(PyroModule):
    def __init__(self, k, n,
                 a_sigma=2.0, # Hyperparams: ARD prior per factor: IG
                 b_sigma=2.0,
                 a_psi=2.0, # Hyperparams: per-view error: IG
                 b_psi=2.0):
        super().__init__()
        # Model parameters
        self.k = k
        self.n = n
        self.p = None
        
        # Hyperparams
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.a_psi = a_psi
        self.b_psi = b_psi
        
        self.loss_history = []
        
    
    def forward(self, X, batch_idx):
        if torch.is_tensor(X):
            m, p = X.shape # Working with minibatches of size m
        elif isinstance(X, TensorDataset): # working with full dataset directly
            # print("is tensordataset")
            m = len(X)
            p = X[0][0].shape[0]
        # self.p = 
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # ARD prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda_jk ~ N(0, sigma_k^2)
        # sigma_k^2 ~ InvGamma(a_sigma, b_sigma))
        ########################
        # No row-wise conditional independence of loadings.
        sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1))
        sigma = torch.sqrt(sigma2)
        
        # sigma is broadcast across rows of Lambda
        Lambda = pyro.sample("Lambda", dist.Normal(torch.zeros(p, self.k), sigma).to_event(2))
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample("psi", dist.InverseGamma(self.a_psi, self.b_psi).expand([p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx): # X is a minibatch, can pass directly into subsample
            # Latent scores Z_i are local
            # Z_batch = pyro.sample("Z", dist.Normal(torch.zeros(m, self.k), torch.ones(self.k)).to_event(1))
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X)
            
            
    def guide(self, X, batch_idx):
        if torch.is_tensor(X):
            m, p = X.shape # Working with minibatches of size m
        elif isinstance(X, TensorDataset): # working with full dataset directly
            # print("is tensordataset")
            m = len(X)
            p = X[0][0].shape[0]
            
        ######
        # Loadings
        # The model specifies 
        ######
        # Sigma2: Variational parameters a, b
        a_sigma_q = pyro.param("a_sigma_q", 
                               lambda: torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        b_sigma_q = pyro.param("b_sigma_q", 
                               lambda: torch.ones((self.k)),
                               constraint = dist.constraints.positive)
        
        sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_q, b_sigma_q).to_event(1))
        # sigma = torch.sqrt(sigma2)
        
        # Lambda: Variational parameters loc, scale
        Lambda_loc = pyro.param("Lambda_loc", lambda: torch.zeros(p, self.k))
        Lambda_scale = pyro.param("Lambda_scale", lambda: torch.ones(p, self.k),
                                  constraint = dist.constraints.positive)
        Lambda = pyro.sample("Lambda", dist.Normal(Lambda_loc, Lambda_scale).to_event(2))
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi_q = pyro.param("a_psi_q",
                             lambda: torch.ones((p)),
                             constraint = dist.constraints.positive)     
        b_psi_q = pyro.param("b_psi_q",
                             lambda: torch.ones((p)),
                             constraint = dist.constraints.positive)  
        psi = pyro.sample("psi", dist.InverseGamma(a_psi_q, b_psi_q).to_event(1))
        # psi_sqrt = torch.sqrt(psi)
        
        # Local latent variables Z: Variational params loc, scale
        Z_loc = pyro.param("Z_loc", lambda: torch.zeros(self.n, self.k))
        Z_scale = pyro.param("Z_scale", lambda: torch.ones(self.n, self.k),
                                constraint = dist.constraints.positive)
        with pyro.plate("obs", self.n, subsample = batch_idx):
            Z_loc_batch = Z_loc[batch_idx]
            Z_scale_batch = Z_scale[batch_idx]
            Z_batch = pyro.sample("Z", dist.Normal(Z_loc_batch, Z_scale_batch).to_event(1))
            
            pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            
    
##############
def do_inference(X, model, guide, opt = "Adam",
                    epochs = 20,
                    max_iter = 20000,
                    minibatch_flag = False,
                    minibatch_size = 32,
                    tol = 1e-4,
                    opt_args = {"lr": 0.001},
                    device = "cpu"):
    
    # if minibatch_flag == False, then do not minibatch, data loader not necessary
    if not minibatch_flag:
        minibatch_size = model.n
        
    ########################
    # Handle minibatching of dataset
    ########################
    X_mod = TensorDataset(torch.arange(X.shape[0]), X)
    loader = DataLoader(X_mod, batch_size = minibatch_size, shuffle = True, drop_last=True)
    
    ########################
    # Initialize instances for optimization
    ########################
    # guide = self.guide
    if opt == "Adam":
        opt = Adam(opt_args)
    elif opt == "ClippedAdam":
        opt = ClippedAdam(opt_args)
    elbo = Trace_ELBO(num_particles = 10)
    # elbo = TraceGraph_ELBO()
    svi = SVI(model, guide, opt, loss = elbo)
    
    ########################
    # Train
    # If minibatching: pass batch from loader
    # If not minibatching: pass original data as tensor
    ########################
    if minibatch_flag:
        prev_loss = None
        for epoch in range(epochs):
            epoch_loss = 0.
            for batch_idx, batch, in loader:
                # print(batch.shape)
                loss = svi.step(batch, batch_idx)
                # print(loss)
                epoch_loss += loss
            print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / model.n:.4f}")
            print(f"Loss at epoch {epoch+1}: {loss / model.n}")
            
            model.loss_history.append(epoch_loss)
            
            if prev_loss is not None and abs(epoch_loss - prev_loss) / model.n < tol:
                break
                
            # print(f"delta: {epoch_loss / n - (prev_loss / n if prev_loss is not None else epoch_loss / n):+.6f}")
            
            # for name in pyro.get_param_store().get_all_param_names():
            #     p = pyro.param(name)
            #     print(name, p.shape, getattr(p, "grad", None) is not None)
            
            prev_loss = loss
    else:
        # total_loss = 0.
        prev_loss = None
        for iter in range(max_iter):
            # total_loss = 0.
            loss = svi.step(X, torch.arange(X.shape[0])) #batch[0])
            # print(loss)
            # total_loss += loss
            # for batch in loader:
            #     loss = svi.step(batch[0])
            #     epoch_loss += loss
            if iter % 100 == 0:
                print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
                model.loss_history.append(loss)
                # print(f"Epoch {iter+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss:.4f}")
                
            # if prev_loss is not None and abs((loss- prev_loss) / prev_loss) < tol:
            if prev_loss is not None and abs((loss- prev_loss)) < tol:
                # print(f"Stopping: {abs((loss- prev_loss) / prev_loss)} < {tol}")
                print(f"Iteration {iter-1}  loss: {prev_loss / model.n:.4f}")
                print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
                break
            prev_loss = loss
            
        
    return svi, opt
        

    
# def predict_factor_model(model,
#                          guide,
#                          num_samples,
#                          data):
    
#     num_samples = 1000
#     predictive = Predictive(model, guide=guide, num_samples=num_samples)
#     return predictive(**data)
#     # svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
#     #             for k, v in predictive(data).items()
#     #             if k != "obs"}