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

from constants import Sites, Params
from data_utils import align_tensor_shapes

pyro.enable_validation(True)
pyro.set_rng_seed(0)
 

class MatrixDecomp(PyroModule):
    def __init__(self, k,
                 loading_target = None,
                 dense = False,
                 a_sigma=2.0, # Hyperparams: Conjugate prior per factor: IG
                 b_sigma=2.0,
                 a1_sigma=2.1, # Hyperparams: Conjugate prior per factor: IG
                 a2_sigma=3.1,
                 alpha=3.0,
                 a_psi=2.0, # Hyperparams: per-view error: IG
                 b_psi=2.0):
        super().__init__()
        
        self.model_type = "MatrixDecomp"
        # Model parameters
        self.k = k
        self.n = None
        self.p = None
        
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
        
        self.dense = False
        # self.loss_history = []
        
        self.total_epochs = 0
        self.total_iters = None
        self.loss_history = []
        self.var_param_convergence_history = []
        
        
    def forward(self, 
                X, 
                batch_idx):
        """
        Should we run model with sparsity prior or not?
        """
        if self.dense:
            return self.forward_dense(X, 
                batch_idx)
        else:
            return self.forward_mgp(X, 
                batch_idx)
    
    def forward_mgp(self, X, batch_idx):
        l = 0
        
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        # self.p = 
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
        # if self.p_l_list is None:
        #     self.p_l_list = [X_l.shape[1] for X_l in X_list]
        
        ########################
        # ---- Loadings --------
        # Loadings Lambda: sample rows across features (p, k)
        # MGP prior
        # Lambda_j ~ N_k(0, sigma_k^2 I)
        #   ==> Lambda_jk ~ N(0, N(0, (rho^l_{jk})^-1 * (tau^l_k)^-12))
        ########################
        # tau - global shrinkage
        delta_lambda = []
        for m in range(self.k):
            shape = self.a1_sigma if m == 0 else self.a2_sigma
            delta_l_m = pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                                    dist.Gamma(shape, 1.0))
            delta_lambda.append(delta_l_m)
        tau_lambda_k_list = torch.cumprod(torch.stack(delta_lambda), dim = 0).squeeze()
    
        # rho - local shrinkage
        rho_lambda = pyro.sample(Sites.rho_lambda_l.format(l = l), 
                            dist.Gamma(self.alpha / 2, self.alpha / 2).expand([self.p, self.k]).to_event(2)).squeeze()
        
        # print('Lambda - tau shape', tau_lambda_k_list.shape)
        # print('Lambda - rho shape', rho_lambda.shape)
        # sigma_lambda_l = (rho_lambda * tau_lambda_k_list).pow_(-0.5)   
        precision = rho_lambda * tau_lambda_k_list
        precision = torch.clamp(precision, min=1e-10, max=1e10)
        sigma_lambda_l = precision.pow_(-0.5)     
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(torch.zeros(self.p, self.k), sigma_lambda_l).to_event(2)).squeeze()
        
        # sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1))
        # sigma = torch.sqrt(sigma2)
        
        # # sigma is broadcast across rows of Lambda
        # Lambda = pyro.sample("Lambda", dist.Normal(torch.zeros(p, self.k), sigma).to_event(2))
        
        ########################
        # ---- Observations --------
        # Working with minibatches of data (subsamples of rows of X)
        # The plate statement defines conditional independence over each observation
        # We assume the full dataset cannot fit in memory, so a data loader is used OUTSIDE this
        #   function to perform minibatching. 
        ########################
        
        # Idiosyncratic error variance            
        psi = pyro.sample("psi", dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1))
        psi_sqrt = torch.sqrt(psi)
        
        # Local latent variables and observations
        with pyro.plate("obs", self.n, subsample = batch_idx): # X is a minibatch, can pass directly into subsample
            # Latent scores Z_i are local
            # Z_batch = pyro.sample("Z", dist.Normal(torch.zeros(m, self.k), torch.ones(self.k)).to_event(1))
            Z_batch = pyro.sample("Z", dist.Normal(0., 1.).expand([self.k]).to_event(1))
            
            # Compute structure
            structure = pyro.deterministic("structure", torch.matmul(Z_batch, Lambda.T))
            # if torch.isnan(structure).any():
            #         print("NaN values found in X_l_tensor!")
            # if torch.isinf(structure).any():
            #     print("Inf values found in X_l_tensor!")
                
            # print("finite structure?") 
            # print(torch.isfinite(structure).all())
            # print("finite sd?")
            # print(torch.isfinite(psi_sqrt).all())
            # print("geq 0 sd?") 
            # print((psi_sqrt > 0).all())
            # print("X finite:", torch.isfinite(X).all())
            # print("NaNs:", torch.isnan(X).sum())
            # print("Infs:", torch.isinf(X).sum())

            
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0]) # list of 1 element 
        
        

    
    def forward_dense(self, X, batch_idx):
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
            
            pyro.sample("X", dist.Normal(structure, psi_sqrt).to_event(1), obs = X[0]) # list of 1 element 
            
            
    def guide(self, X, batch_idx):
        l = 0
        # if torch.is_tensor(X):
        #     m, p = X.shape # Working with minibatches of size m
        # elif isinstance(X, TensorDataset): # working with full dataset directly
        #     # print("is tensordataset")
        #     m = len(X)
        #     p = X[0][0].shape[0]
        m = len(batch_idx)
        if self.p is None:
            self.p = X[0].shape[1]
            
        # print(X)
        # print(len(X))
        # print(X[0].shape)
        # print(X[1])
            
        ######
        # Loadings
        # The model specifies 
        ######
        for m in range(self.k):
            a_delta_lambda_l_k = pyro.param(Params.a_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(self.a1_sigma),
                                            constraint = dist.constraints.positive)
            b_delta_lambda_l_k = pyro.param(Params.b_delta_lambda_l_k.format(l = l, m = m), 
                                            torch.tensor(1.0),
                                            constraint = dist.constraints.positive)
            pyro.sample(Sites.delta_lambda_l_k.format(l = l, m = m), 
                        dist.Gamma(a_delta_lambda_l_k, b_delta_lambda_l_k))

        a_rho_lambda_l = pyro.param(Params.a_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2),
                                        constraint = dist.constraints.positive)
        b_rho_lambda_l = pyro.param(Params.b_rho_lambda_l.format(l = l), 
                                        torch.tensor(self.alpha / 2),
                                        constraint = dist.constraints.positive)
        pyro.sample(Sites.rho_lambda_l.format(l = l), 
                    dist.Gamma(a_rho_lambda_l, b_rho_lambda_l).expand([self.p, self.k]).to_event(2)).squeeze()
        
        loc_Lambda_l = pyro.param(Params.loc_Lambda_l.format(l = l), torch.zeros(self.p, self.k))
        scale_Lambda_l = pyro.param(Params.scale_Lambda_l.format(l = l), torch.ones(self.p, self.k),
                                    constraint = dist.constraints.positive)
        Lambda = pyro.sample(Sites.Lambda_l.format(l = l), 
                                dist.Normal(loc_Lambda_l, scale_Lambda_l).to_event(2))
        
        
        # Penalty to encourage loadings toward certain matrix
        orthogonality_weight = 100.
        if self.loading_target is not None:
            matrix_diff = Lambda - self.loading_target
            
            total_penalty = torch.sum(matrix_diff ** 2)
                
            pyro.factor("target_penalty", 
                        orthogonality_weight * total_penalty,
                        has_rsample = True)
        
        # # Sigma2: Variational parameters a, b
        # a_sigma_q = pyro.param("a_sigma_q", 
        #                        lambda: torch.ones((self.k)),
        #                        constraint = dist.constraints.positive)
        # b_sigma_q = pyro.param("b_sigma_q", 
        #                        lambda: torch.ones((self.k)),
        #                        constraint = dist.constraints.positive)
        
        # sigma2 = pyro.sample("sigma2_lambda", dist.InverseGamma(a_sigma_q, b_sigma_q).to_event(1))
        # # sigma = torch.sqrt(sigma2)
        
        # # Lambda: Variational parameters loc, scale
        # Lambda_loc = pyro.param("Lambda_loc", lambda: torch.zeros(p, self.k))
        # Lambda_scale = pyro.param("Lambda_scale", lambda: torch.ones(p, self.k),
        #                           constraint = dist.constraints.positive)
        # Lambda = pyro.sample("Lambda", dist.Normal(Lambda_loc, Lambda_scale).to_event(2))
        
        ########################
        # Scores
        ########################
        # Psi: Variational params a, b
        a_psi_q = pyro.param("a_psi_q",
                             lambda: torch.ones((self.p)),
                             constraint = dist.constraints.positive)     
        b_psi_q = pyro.param("b_psi_q",
                             lambda: torch.ones((self.p)),
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
# def do_inference(X, model, guide, opt = "Adam",
#                     epochs = 20,
#                     max_iter = 20000,
#                     minibatch_flag = False,
#                     minibatch_size = 32,
#                     tol = 1e-4,
#                     opt_args = {"lr": 0.001},
#                     device = "cpu"):
    
#     # if minibatch_flag == False, then do not minibatch, data loader not necessary
#     if not minibatch_flag:
#         minibatch_size = model.n
        
#     ########################
#     # Handle minibatching of dataset
#     ########################
#     X_mod = TensorDataset(torch.arange(X.shape[0]), X)
#     loader = DataLoader(X_mod, batch_size = minibatch_size, shuffle = True, drop_last=True)
    
#     ########################
#     # Initialize instances for optimization
#     ########################
#     # guide = self.guide
#     if opt == "Adam":
#         opt = Adam(opt_args)
#     elif opt == "ClippedAdam":
#         opt = ClippedAdam(opt_args)
#     elbo = Trace_ELBO(num_particles = 10)
#     # elbo = TraceGraph_ELBO()
#     svi = SVI(model, guide, opt, loss = elbo)
    
#     ########################
#     # Train
#     # If minibatching: pass batch from loader
#     # If not minibatching: pass original data as tensor
#     ########################
#     if minibatch_flag:
#         prev_loss = None
#         for epoch in range(epochs):
#             epoch_loss = 0.
#             for batch_idx, batch, in loader:
#                 # print(batch.shape)
#                 loss = svi.step(batch, batch_idx)
#                 # print(loss)
#                 epoch_loss += loss
#             print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / model.n:.4f}")
#             print(f"Loss at epoch {epoch+1}: {loss / model.n}")
            
#             model.loss_history.append(epoch_loss)
            
#             if prev_loss is not None and abs(epoch_loss - prev_loss) / model.n < tol:
#                 break
                
#             # print(f"delta: {epoch_loss / n - (prev_loss / n if prev_loss is not None else epoch_loss / n):+.6f}")
            
#             # for name in pyro.get_param_store().get_all_param_names():
#             #     p = pyro.param(name)
#             #     print(name, p.shape, getattr(p, "grad", None) is not None)
            
#             prev_loss = loss
#     else:
#         # total_loss = 0.
#         prev_loss = None
#         for iter in range(max_iter):
#             # total_loss = 0.
#             loss = svi.step(X, torch.arange(X.shape[0])) #batch[0])
#             # print(loss)
#             # total_loss += loss
#             # for batch in loader:
#             #     loss = svi.step(batch[0])
#             #     epoch_loss += loss
#             if iter % 100 == 0:
#                 print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
#                 model.loss_history.append(loss)
#                 # print(f"Epoch {iter+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss:.4f}")
                
#             # if prev_loss is not None and abs((loss- prev_loss) / prev_loss) < tol:
#             if prev_loss is not None and abs((loss- prev_loss)) < tol:
#                 # print(f"Stopping: {abs((loss- prev_loss) / prev_loss)} < {tol}")
#                 print(f"Iteration {iter-1}  loss: {prev_loss / model.n:.4f}")
#                 print(f"Iteration {iter}  loss: {loss / model.n:.4f}")
#                 break
#             prev_loss = loss
            
        
#     return svi, opt
        

    
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