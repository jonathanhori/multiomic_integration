import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.contrib.zuko import ZukoToPyro
from zuko.flows import NSF, MAF

from constants import Sites, Params


class MatrixDecompFlow(PyroModule):
    """
    Single-view supervised/unsupervised matrix factorization with a Zuko
    normalizing flow variational guide for the latent scores Z.

    Model: dense (ARD) prior on Lambda, InverseGamma on idiosyncratic noise psi.
    Guide: mean-field (Normal / InverseGamma) for all global variables;
           amortized NSF/MAF flow  q(Z_i | X_i)  for the local scores.

    The flow takes X_i as context through a conditioner network, so trained
    flow parameters can be applied directly to held-out observations without
    any additional per-observation optimization (amortized inference).

    Args:
        k:                   number of latent factors
        p:                   observed feature dimension — must be known at init
                             to build the flow conditioner network
        flow_type:           "NSF" (Neural Spline Flow, recommended) or
                             "MAF" (Masked Autoregressive Flow)
        n_flow_transforms:   number of coupling/autoregressive layers
        flow_hidden_features: width of each layer in the conditioner MLP
        supervised:          include outcome regression  y ~ N(Z @ beta, sigma_y)
        a_sigma, b_sigma:    InverseGamma hyperparameters for ARD column variances
        a_psi, b_psi:        InverseGamma hyperparameters for observation noise
    """

    def __init__(
        self,
        k: int,
        p: int,
        flow_type: str = "NSF",
        n_flow_transforms: int = 3,
        flow_hidden_features: tuple = (64, 64),
        supervised: bool = False,
        a_sigma: float = 2.1,
        b_sigma: float = 2.0,
        a_psi: float = 3.0,
        b_psi: float = 1.0,
    ):
        super().__init__()
        self.model_type = "MatrixDecompFlow"
        self.k = k
        self.p = p
        self.n = None
        self.supervised_model = supervised

        # Dense (ARD) prior hyperparams
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.a_psi = a_psi
        self.b_psi = b_psi

        # Outcome model hyperparams
        self.a_sigma_y = 3.0
        self.b_sigma_y = 2.0
        self.a_sigma_beta = 2.1
        self.b_sigma_beta = 2.0

        # Zuko normalizing flow: q(Z_i | X_i)
        # NSF — Neural Spline Flow: expressive, fast to sample, good for VI
        # MAF — Masked Autoregressive Flow: fast log_prob but slow sampling
        flow_cls = {"NSF": NSF, "MAF": MAF}[flow_type]
        self.z_flow = flow_cls(
            features=k,
            context=p,
            transforms=n_flow_transforms,
            hidden_features=list(flow_hidden_features),
        )

        self.total_epochs = 0
        self.loss_history = []
        self.params = None

    # ------------------------------------------------------------------
    # Generative model
    # ------------------------------------------------------------------

    def forward(self, X, batch_idx, y=None):
        """
        p(X, Z, Lambda, psi [, beta, sigma_y]):
          sigma2_lambda_k ~ InvGamma(a_sigma, b_sigma)   k = 1..K  (ARD)
          Lambda          ~ N(0, diag(sigma2_lambda))    (p x K)
          psi             ~ InvGamma(a_psi, b_psi)       (p,)
          Z_i             ~ N(0, I)                      (K,)
          X_i             ~ N(Z_i @ Lambda^T, diag(psi)) (p,)
          [supervised]
          sigma2_beta     ~ InvGamma(...)                (K,)
          beta            ~ N(0, diag(sigma2_beta))      (K,)
          sigma2_y        ~ InvGamma(...)
          y_i             ~ N(Z_i @ beta, sqrt(sigma2_y))
        """
        l = 0

        sigma2_lambda = pyro.sample(
            "sigma2_lambda",
            dist.InverseGamma(self.a_sigma, self.b_sigma).expand([self.k]).to_event(1),
        )
        Lambda = pyro.sample(
            Sites.Lambda_l.format(l=l),
            dist.Normal(torch.zeros(self.p, self.k), sigma2_lambda.sqrt()).to_event(2),
        )
        psi = pyro.sample(
            Sites.psi_l.format(l=l),
            dist.InverseGamma(self.a_psi, self.b_psi).expand([self.p]).to_event(1),
        )

        if self.supervised_model:
            sigma2_beta = pyro.sample(
                Sites.sigma2_beta,
                dist.InverseGamma(self.a_sigma_beta, self.b_sigma_beta)
                .expand([self.k])
                .to_event(1),
            )
            beta = pyro.sample(
                Sites.beta,
                dist.Normal(torch.zeros(self.k), sigma2_beta.sqrt()).to_event(1),
            )
            sigma2_y = pyro.sample(
                Sites.sigma2_y, dist.InverseGamma(self.a_sigma_y, self.b_sigma_y)
            )

        with pyro.plate("obs", self.n, subsample=batch_idx):
            Z_batch = pyro.sample(
                "Z", dist.Normal(0.0, 1.0).expand([self.k]).to_event(1)
            )
            pyro.sample(
                "X",
                dist.Normal(Z_batch @ Lambda.T, psi.sqrt()).to_event(1),
                obs=X[0],
            )
            if self.supervised_model:
                pyro.sample(
                    "y",
                    dist.Normal(Z_batch @ beta, sigma2_y.sqrt()),
                    obs=y,
                )

    # ------------------------------------------------------------------
    # Variational guide
    # ------------------------------------------------------------------

    def guide(self, X, batch_idx, y=None):
        """
        q(Lambda, psi [, beta, sigma_y]): mean-field Normal / InverseGamma
        q(Z_i | X_i):                     amortized Zuko flow
        """
        l = 0

        # --- sigma2_lambda (ARD column variances) ---
        a_sl = pyro.param(
            "a_sigma_lambda",
            self.a_sigma * torch.ones(self.k),
            constraint=dist.constraints.positive,
        )
        b_sl = pyro.param(
            "b_sigma_lambda",
            self.b_sigma * torch.ones(self.k),
            constraint=dist.constraints.positive,
        )
        pyro.sample("sigma2_lambda", dist.InverseGamma(a_sl, b_sl).to_event(1))

        # --- Lambda ---
        loc_Lambda = pyro.param(
            Params.loc_Lambda_l.format(l=l), torch.zeros(self.p, self.k)
        )
        scale_Lambda = pyro.param(
            Params.scale_Lambda_l.format(l=l),
            0.1 * torch.ones(self.p, self.k),
            constraint=dist.constraints.positive,
        )
        pyro.sample(
            Sites.Lambda_l.format(l=l),
            dist.Normal(loc_Lambda, scale_Lambda).to_event(2),
        )

        # --- psi ---
        a_psi = pyro.param(
            Params.a_psi_l.format(l=l),
            self.a_psi * torch.ones(self.p),
            constraint=dist.constraints.positive,
        )
        b_psi = pyro.param(
            Params.b_psi_l.format(l=l),
            self.b_psi * torch.ones(self.p),
            constraint=dist.constraints.positive,
        )
        pyro.sample(Sites.psi_l.format(l=l), dist.InverseGamma(a_psi, b_psi).to_event(1))

        if self.supervised_model:
            a_sb = pyro.param(
                Params.a_sigma_beta,
                self.a_sigma_beta * torch.ones(self.k),
                constraint=dist.constraints.positive,
            )
            b_sb = pyro.param(
                Params.b_sigma_beta,
                self.b_sigma_beta * torch.ones(self.k),
                constraint=dist.constraints.positive,
            )
            pyro.sample(
                Sites.sigma2_beta, dist.InverseGamma(a_sb, b_sb).to_event(1)
            )

            loc_beta = pyro.param(Params.loc_beta, torch.zeros(self.k))
            scale_beta = pyro.param(
                Params.scale_beta,
                torch.ones(self.k),
                constraint=dist.constraints.positive,
            )
            pyro.sample(
                Sites.beta, dist.Normal(loc_beta, scale_beta).to_event(1)
            )

            a_sy = pyro.param(
                Params.a_sigma_y,
                torch.tensor(self.a_sigma_y),
                constraint=dist.constraints.positive,
            )
            b_sy = pyro.param(
                Params.b_sigma_y,
                torch.tensor(self.b_sigma_y),
                constraint=dist.constraints.positive,
            )
            pyro.sample(Sites.sigma2_y, dist.InverseGamma(a_sy, b_sy))

        # Register the flow's neural network weights in Pyro's param store so
        # the SVI optimizer (ClippedAdam) includes them alongside Pyro params.
        pyro.module("z_flow", self.z_flow)

        # --- Z: amortized flow q(Z_i | X_i) ---
        # X[0] has shape (batch_size, p) — already the minibatch from DataLoader.
        # self.z_flow(X[0]) returns a NormalizingFlow distribution with
        #   batch_shape = (batch_size,)  and  event_shape = (k,)
        # which maps correctly into the plate over obs.
        with pyro.plate("obs", self.n, subsample=batch_idx):
            pyro.sample("Z", ZukoToPyro(self.z_flow(X[0])))

    # ------------------------------------------------------------------
    # Amortized test-set inference
    # ------------------------------------------------------------------

    def sample_z(self, X_tensor: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """
        Draw posterior samples of Z for new observations using the trained flow.
        Because the guide is amortized, this requires no additional optimization.

        Args:
            X_tensor:    (n_test, p) observed feature matrix for test data
            num_samples: number of posterior samples per observation

        Returns:
            z_samples: (num_samples, n_test, k) tensor
        """
        assert self.params is not None, "Run do_inference before calling sample_z."
        with torch.no_grad():
            z_dist = self.z_flow(X_tensor)
            return z_dist.rsample((num_samples,))

    def predict(self, X_test: torch.Tensor, num_samples: int = 100) -> dict:
        """
        Posterior predictive samples for test observations.

        Samples Lambda from its mean-field guide and Z from the flow, then
        computes reconstructed X and (if supervised) predicted y.

        Returns dict with:
            "Z_samples":     (num_samples, n_test, k)
            "X_pred_mean":   (n_test, p) — posterior predictive mean of X
            "y_pred_mean":   (n_test,)  — posterior predictive mean of y (supervised only)
        """
        assert self.params is not None, "Run do_inference before calling predict."
        store = pyro.get_param_store()
        loc_Lambda = store[Params.loc_Lambda_l.format(l=0)]  # (p, k)

        with torch.no_grad():
            Z_samples = self.sample_z(X_test, num_samples)  # (S, n_test, k)
            X_pred_mean = (Z_samples.mean(0) @ loc_Lambda.T)  # (n_test, p)

        result = {"Z_samples": Z_samples, "X_pred_mean": X_pred_mean}

        if self.supervised_model:
            loc_beta = store[Params.loc_beta]  # (k,)
            with torch.no_grad():
                result["y_pred_mean"] = Z_samples.mean(0) @ loc_beta  # (n_test,)

        return result
