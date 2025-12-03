
import math

import torch
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO

from torch.utils.data import DataLoader

class ModelHandler:
    def __init__(self):
        pass 
    
    def do_inference(self,
                     train_dataset,
                    model, 
                    guide, 
                    opt = Adam({"lr": 0.001}), #"Adam",
                    elbo = Trace_ELBO(),
                    inference = "svi",
                    min_epochs = 10,
                    epochs = 20,
                    max_iter = 20000,
                    minibatch_flag = True,
                    minibatch_size = 32,
                    tol = 1e-4,
                    variational_tol = 1e-4,
                    device = "cpu",
                    verbose = False):
        
        # min_epochs = 10
        
        # if minibatch_flag == False, then do not minibatch, data loader not necessary
        if not minibatch_flag:
            minibatch_size = model.n
            
        ########################
        # Handle minibatching of dataset
        ########################
        loader = DataLoader(train_dataset, batch_size = minibatch_size, shuffle = True, drop_last=True)
        
        ########################
        # Initialize instances for optimization
        ########################
        # guide = self.guide
        # if opt == "Adam":
        #     opt = Adam(opt_args)
        # elif opt == "ClippedAdam":
        #     opt = ClippedAdam(opt_args)
        # elbo = Trace_ELBO(num_particles = 1)
        # elbo = TraceGraph_ELBO()
        svi = SVI(model, guide, opt, loss = elbo)
        
        ########################
        # Train
        # If minibatching: pass batch from loader
        # If not minibatching: pass original data as tensor
        ########################
        if minibatch_flag:
            # prev_loss = None
            min_loss = math.inf
            epoch_at_min_loss = 0
            params_epoch_last = None
            params_epoch_curr = None
            for epoch in range(epochs):
                epoch_loss = 0.
                for batch in loader:
                    # batch is subsampled [idx, X_l_list, y]
                    batch_idx = batch.pop(0)
                    y_batch = batch.pop(-1).squeeze()
                    # print(batch.shape)
                    loss = svi.step(batch, batch_idx, y_batch)
                    # print(loss)
                    epoch_loss += loss
                
                model.loss_history.append(epoch_loss)
                
                # Determine the epoch when the min loss is obtained
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    epoch_at_min_loss = epoch
                
                model.total_epochs += 1
                param_store_curr = pyro.get_param_store()
                params_epoch_curr = {k: v.detach().clone() for k, v in param_store_curr.items()}
                if epoch == 0: 
                    params_epoch_last = params_epoch_curr
                # Check Euclidean norm of difference in variational params for convergence
                param_diff_norm_dict = {k: torch.norm(params_epoch_last[k] - params_epoch_curr[k]).item() for k in pyro.get_param_store()}
                model.var_param_convergence_history.append(max(param_diff_norm_dict.values()))
                # print(params_epoch_curr.items())
                params_converged = all(n < variational_tol for n in param_diff_norm_dict.values())
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}  avg neg-ELBO per datum: {epoch_loss / model.n:.4f}")
                    print(f"Loss at epoch {epoch+1}: {loss / model.n}")
                    print(f"Max variational parameter difference: {model.var_param_convergence_history[-1]}")
                
                # Converged?
                if epoch > min_epochs and epoch - epoch_at_min_loss > math.sqrt(epoch)\
                    and params_epoch_last is not None and params_converged:
                # if prev_loss is not None and abs(epoch_loss - prev_loss) / model.n < tol:
                    # model.total_epochs = epoch
                    break
                
                params_epoch_last = params_epoch_curr
                
                # prev_loss = epoch_loss
        else:
            raise NotImplementedError            
            
        return svi, opt