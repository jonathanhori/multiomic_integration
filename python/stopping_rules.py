"""
Offline replay of training stopping rules on saved loss trajectories.

A stopping rule only decides *when* to stop; it does not change any step before
that. So for a fit trained without early stopping (controlled_training_length.py),
the epoch at which a rule would have stopped, and the fit at that point, are
exactly what a run using that rule would have produced (same seed and data).
Limits:
  - a rule that has not fired by the last recorded epoch is right-censored
    ("did not stop within max_epochs");
  - the fit is scored only at checkpoints (every 100 epochs), so the risk at a
    stopping epoch e is taken from the last checkpoint <= e (floor);
  - rules on the predictor coefficients can only be checked at checkpoints.

Epoch conventions follow ModelHandler.do_inference: epochs are 0-based inside the
loop, a rule can fire only when epoch > min_epochs, and stopping at loop index e
means e + 1 epochs were run (Phase 2: elbo_snr, window 50, threshold 5,
min_epochs 200 stops after 202 epochs).

Each rule takes the per-epoch loss history (and for coef_change the checkpoint
table) and returns the number of epochs run at the stop, or None if censored.
"""
import math

import numpy as np


def elbo_snr(loss, min_epochs=200, window=50, snr_threshold=5.0, spike_tol=0.1):
    """ModelHandler's elbo_snr: |mean(last w) - mean(previous w)| / sd(last w) < threshold,
    and the current epoch is not a spike (loss <= (1 + spike_tol) x median of last w)."""
    for e in range(len(loss)):
        if e <= min_epochs or e + 1 < 2 * window:
            continue
        recent, prev = loss[e + 1 - window:e + 1], loss[e + 1 - 2 * window:e + 1 - window]
        sd = np.std(recent)
        if sd == 0:
            return e + 1
        snr = abs(np.mean(recent) - np.mean(prev)) / sd
        if snr < snr_threshold and loss[e] <= np.median(recent) * (1 + spike_tol):
            return e + 1
    return None


def elbo_min(loss, min_epochs=200):
    """ModelHandler's elbo_min: epochs since the lowest loss exceed sqrt(current epoch)."""
    best, at = math.inf, 0
    for e in range(len(loss)):
        if loss[e] < best:
            best, at = loss[e], e
        if e > min_epochs and e - at > math.sqrt(e):
            return e + 1
    return None


def rel_loss_change(loss, min_epochs=200, window=100, tol=1e-3):
    """Relative change of the windowed mean loss: |mean(last w) - mean(previous w)| / |mean(last w)| < tol."""
    for e in range(len(loss)):
        if e <= min_epochs or e + 1 < 2 * window:
            continue
        m1 = np.mean(loss[e + 1 - window:e + 1])
        m0 = np.mean(loss[e + 1 - 2 * window:e + 1 - window])
        if abs(m1 - m0) / abs(m1) < tol:
            return e + 1
    return None


def coef_change(checkpoints, min_epochs=200, tol=0.01, patience=1):
    """Relative change of the model's own predictor coefficients between consecutive
    checkpoints (column coef_rel_change) below tol for `patience` checkpoints in a row.
    checkpoints: rows for one dataset with columns epoch, coef_rel_change (any variant)."""
    cp = checkpoints.drop_duplicates("epoch").sort_values("epoch")
    run = 0
    for ep, ch in zip(cp.epoch, cp.coef_rel_change):
        if ep <= min_epochs or not np.isfinite(ch):
            continue
        run = run + 1 if ch < tol else 0
        if run >= patience:
            return int(ep)
    return None


def risk_at(checkpoints, epochs_run, variant="own", col="risk_rel"):
    """Metric at the last checkpoint <= epochs_run (the final checkpoint if censored)."""
    cp = checkpoints[checkpoints.variant == variant].sort_values("epoch")
    if epochs_run is None:
        return float(cp[col].iloc[-1])
    sel = cp[cp.epoch <= epochs_run]
    return float(sel[col].iloc[-1]) if len(sel) else float("nan")


# ---------------------------------------------------------------------------
# Spike-robust, effect-size rules. elbo_snr asks whether the change between two
# windows is *detectable* against epoch-to-epoch noise; a slow steady drift never
# is in a short window, so it fires early. These rules instead ask whether the
# *size* of the improvement is small, using medians / robust slopes so that
# occasional loss spikes do not trigger (or block) stopping. They are evaluated
# every `every` epochs; `patience` consecutive checks must pass.
# ---------------------------------------------------------------------------

def _first_pass(checks, patience):
    run = 0
    for e, ok in checks:
        run = run + 1 if ok else 0
        if run >= patience:
            return e
    return None


def robust_rel_improvement(loss, min_epochs=200, window=100, tol=1e-3, every=10, patience=1):
    """(median(previous window) - median(last window)) / |median(last window)| < tol."""
    def checks():
        for e in range(max(min_epochs + 1, 2 * window - 1), len(loss), every):
            m1 = np.median(loss[e + 1 - window:e + 1])
            m0 = np.median(loss[e + 1 - 2 * window:e + 1 - window])
            yield e + 1, (m0 - m1) / abs(m1) < tol
    return _first_pass(checks(), patience)


def robust_slope(loss, min_epochs=200, window=200, horizon=None, tol=1e-3, every=10, patience=1):
    """Theil-Sen slope of the loss over the last `window` epochs, projected over
    `horizon` epochs (default: window) relative to the current median loss:
    -slope * horizon / |median| < tol, i.e. the expected further relative
    improvement over the next horizon epochs is below tol."""
    from scipy.stats import theilslopes
    horizon = horizon or window
    x = np.arange(window)
    def checks():
        for e in range(max(min_epochs + 1, window - 1), len(loss), every):
            seg = loss[e + 1 - window:e + 1]
            slope = theilslopes(seg, x)[0]
            yield e + 1, -slope * horizon / abs(np.median(seg)) < tol
    return _first_pass(checks(), patience)


def elbo_snr_robust(loss, min_epochs=200, window=50, snr_threshold=0.5, every=1, patience=1):
    """elbo_snr with medians and MAD: |median(last w) - median(previous w)| / (1.4826 MAD(last w))."""
    def checks():
        for e in range(max(min_epochs + 1, 2 * window - 1), len(loss), every):
            rec, prev = loss[e + 1 - window:e + 1], loss[e + 1 - 2 * window:e + 1 - window]
            mad = 1.4826 * np.median(np.abs(rec - np.median(rec)))
            yield e + 1, mad == 0 or abs(np.median(rec) - np.median(prev)) / mad < snr_threshold
    return _first_pass(checks(), patience)


# ---------------------------------------------------------------------------
# Rules from Tran, Nguyen & Dao (2021), "A practical tutorial on Variational
# Bayes" (arXiv:2103.01327), Section 3.1 / Algorithms 4-7. Our loss is the
# negative ELBO, so "the lower bound increases" = "the loss decreases".
# ---------------------------------------------------------------------------

def ma_patience(loss, min_epochs=0, window=50, patience=50):
    """Moving-average lower bound with patience: smooth the loss over the last
    `window` epochs; keep the best (lowest) smoothed value so far; the patience
    counter resets when the smoothed loss sets a new best and otherwise grows by
    one; stop when it reaches `patience` (and epoch > min_epochs). The paper
    returns the iterate with the best smoothed bound; here the fit at the stop is
    used (iterates are not saved)."""
    best, count = math.inf, 0
    csum = np.cumsum(np.insert(np.asarray(loss, float), 0, 0.0))
    for e in range(window - 1, len(loss)):
        ma = (csum[e + 1] - csum[e + 1 - window]) / window
        if ma < best:
            best, count = ma, 0
        else:
            count += 1
        if count >= patience and e > min_epochs:
            return e + 1
    return None


def fixed_budget(loss, epochs):
    """Stop after a fixed number of epochs (reference points for the time/risk trade-off)."""
    return epochs if len(loss) >= epochs else None


def val_patience(val_loss, epochs, patience=5, min_epochs=0):
    """Validation-loss rule (Tran, Nguyen & Dao 2021, Alg. 9): with a held-out loss
    recorded at checkpoints (val_loss at the given epochs), stop at the first
    checkpoint after min_epochs where the held-out loss has not reached a new
    minimum for `patience` consecutive checks. Returns (stop_epoch, best_epoch):
    the checkpoint where training stops and the one with the lowest held-out loss
    up to then (the iterate the paper returns). (None, best) if it never fires."""
    best, best_ep, count = np.inf, None, 0
    for v, e in zip(val_loss, epochs):
        if v < best:
            best, best_ep, count = v, e, 0
        else:
            count += 1
        if count >= patience and e > min_epochs:
            return e, best_ep
    return None, best_ep
