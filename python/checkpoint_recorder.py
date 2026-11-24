"""
Record a VIPER-Shared fit during training without changing it, so that stopping rules
and test-time scoring can be evaluated afterwards at any checkpoint.

The avg_param_change stopping rule calls handler.AveragedParamChange.update once per
epoch. record_checkpoints() wraps that method and keeps:
    - every epoch:   loc_beta (one K-vector per epoch);
    - every `every` epochs (and at the epoch the rule fires):
        the full param-store state (copy of pyro.get_param_store().get_state(): all
        global and training-local variational parameters), so local inference of test
        scores can be run later from that checkpoint (see load_checkpoint);
        score_fn(param_store, epoch) if given (e.g. cheap closed-form test scores),
        stored as a row together with that epoch's rule statistics;
    - every completed block: the rule's block means of each tracked group (Lambda
      stacked over views, beta), so the rule can be recomputed afterwards with other
      tolerances, window lengths (multiples of the block) or statistics.
Scoring runs under torch.no_grad() and the torch RNG state is restored afterwards, so the
training trajectory (and the stop epoch) is the same as without recording.

    rec = record_checkpoints(25, score_fn)
    train_model(...)
    rec.restore()
    rec.save(path, loss=model.loss_history)     # one torch file per fit
    rows = rec.rows                              # list of dicts
"""
import copy

import torch


class _Recorder:
    def __init__(self, every, score_fn):
        self.every, self.score_fn = int(every), score_fn
        self.rows, self.state_epochs, self.states = [], [], []
        self.block_epochs, self.block_means = [], {}
        self.beta_epochs, self.beta = [], []
        self._orig = None

    def restore(self):
        import handler
        if self._orig is not None:
            handler.AveragedParamChange.update = self._orig
            self._orig = None

    def save(self, path, **extra):
        torch.save({"state_epochs": self.state_epochs, "states": self.states,
                    "block_epochs": self.block_epochs,
                    "block_means": {g: torch.stack(v) for g, v in self.block_means.items()},
                    "beta_epochs": self.beta_epochs, "beta": torch.stack(self.beta) if self.beta else None,
                    "every": self.every, **extra}, path)


def record_checkpoints(every, score_fn=None):
    import pyro
    import handler
    rec = _Recorder(every, score_fn)
    orig = handler.AveragedParamChange.update
    rec._orig = orig

    def update(self, store, epoch):
        n_before = len(self.means[next(iter(self.groups))])
        cnt_before = self.count
        met = orig(self, store, epoch)
        ps = pyro.get_param_store()
        rec.beta_epochs.append(epoch)
        rec.beta.append(ps["loc_beta"].detach().reshape(-1).to("cpu", torch.float32).clone())
        if cnt_before + 1 == self.block:                         # a block completed this epoch
            rec.block_epochs.append(epoch)
            for g in self.groups:
                rec.block_means.setdefault(g, []).append(self.means[g][-1].clone())
        if epoch % rec.every == 0 or met:
            rng = torch.get_rng_state()
            row = {}
            if rec.score_fn is not None:
                with torch.no_grad():
                    row = dict(rec.score_fn(ps, epoch))
            torch.set_rng_state(rng)
            rec.state_epochs.append(epoch)
            rec.states.append(copy.deepcopy(ps.get_state()))
            stats = self.history[-1][1] if self.history and self.history[-1][0] == epoch else {}
            row.update(epoch=epoch, rule_met=bool(met), **{f"stat_{k}": float(v) for k, v in stats.items()})
            rec.rows.append(row)
        return met

    handler.AveragedParamChange.update = update
    return rec


def load_checkpoint(path, epoch):
    """Param-store state saved at `epoch` (the last checkpoint at or before it)."""
    d = torch.load(path, weights_only=False)
    i = max(j for j, e in enumerate(d["state_epochs"]) if e <= epoch)
    return d["state_epochs"][i], d["states"][i]
