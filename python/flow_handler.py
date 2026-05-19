import numpy as np
import torch
import pyro
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO
from torch.utils.data import DataLoader


class FlowModelHandler:
    """
    Minimal SVI training loop for MatrixDecompFlow.

    Uses ClippedAdam, which automatically includes both the Pyro variational
    parameters (Lambda, psi, etc.) and the Zuko flow's neural network weights
    — the latter are registered in the Pyro param store via `pyro.module` inside
    the guide.

    Convergence is detected via ELBO SNR: training stops when the
    signal-to-noise ratio of ELBO improvement over a sliding window falls
    below `snr_threshold` (same criterion as ModelHandler's "elbo_snr" mode).

    Args:
        model:        a MatrixDecompFlow instance
        lr:           ClippedAdam learning rate
        clip_norm:    gradient clipping norm for ClippedAdam
        num_particles: number of ELBO samples per SVI step (1 is usually fine)
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        clip_norm: float = 10.0,
        num_particles: int = 1,
    ):
        self.model = model
        self.lr = lr
        self.clip_norm = clip_norm
        self.num_particles = num_particles

    def do_inference(
        self,
        train_dataset,
        epochs: int = 500,
        min_epochs: int = 50,
        minibatch_size: int = 32,
        window: int = 50,
        snr_threshold: float = 0.1,
        verbose: bool = False,
    ) -> SVI:
        """
        Run SVI training.

        Args:
            train_dataset:  TensorDataset of (idx, X [, y])
            epochs:         maximum number of training epochs
            min_epochs:     minimum epochs before convergence check activates
            minibatch_size: DataLoader batch size
            window:         sliding window length for ELBO SNR convergence check
            snr_threshold:  stop when ELBO SNR < this value
            verbose:        print loss every 10 epochs

        Returns:
            svi: the trained SVI object
        """
        self.model.n = len(train_dataset)
        loader = DataLoader(
            train_dataset, batch_size=minibatch_size, shuffle=True, drop_last=True
        )

        opt = ClippedAdam({"lr": self.lr, "clip_norm": self.clip_norm})
        loss_fn = Trace_ELBO(num_particles=self.num_particles)
        svi = SVI(self.model.forward, self.model.guide, opt, loss=loss_fn)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch_idx = batch[0]
                X_list = [batch[1]]
                y_batch = batch[2].squeeze() if len(batch) > 2 else None
                epoch_loss += svi.step(X_list, batch_idx, y_batch)

            self.model.loss_history.append(epoch_loss)
            self.model.total_epochs += 1

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1:>4d}/{epochs}  "
                    f"neg-ELBO: {epoch_loss / self.model.n:.4f} per datum"
                )

            # ELBO SNR convergence: stop when improvement becomes noise
            if epoch >= min_epochs and len(self.model.loss_history) >= 2 * window:
                recent = self.model.loss_history[-window:]
                prev = self.model.loss_history[-2 * window : -window]
                std_r = np.std(recent)
                if std_r == 0.0:
                    if verbose:
                        print(f"Converged at epoch {epoch + 1} (zero ELBO variance).")
                    break
                snr = abs(np.mean(recent) - np.mean(prev)) / std_r
                if snr < snr_threshold:
                    if verbose:
                        print(
                            f"Converged at epoch {epoch + 1} "
                            f"(ELBO SNR = {snr:.4f} < {snr_threshold})."
                        )
                    break

        self.model.params = pyro.get_param_store()
        return svi
