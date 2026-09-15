"""Vectorized per-(point, variable) online linear model for walk-forward
weather forecasting.

Rather than one heavyweight model per grid point, every (point, variable)
pair gets a compact linear model — a couple of Fourier seasonal terms plus
two short-lag autoregressive terms — updated one small SGD step at a time
as each new day's observation arrives. Every point and variable in the
grid is updated together in a single vectorized numpy pass per calendar
day, which is what keeps this cheap enough for a small droplet: the full
CONUS-grid model is only a few MB, and one day's update is a handful of
elementwise array operations, not a training run.

Targets and lag features are tracked in a running-normalized space (an
exponential moving mean/std per point/variable) so one global learning
rate behaves sensibly across variables on very different physical scales
(temperature in degC vs. surface pressure in hPa vs. precipitation in mm).
"""
from __future__ import annotations

import os

import numpy as np

N_LAGS = 7  # lag-1 and lag-7 features are read from a 7-slot ring buffer
N_FEATURES = 7  # bias, sin1, cos1, sin2, cos2, lag1, lag7

LEARNING_RATE = 0.02
NORM_EMA_BETA = 0.001  # running mean/std adapt slowly -- climate drift, not day-to-day noise
RESID_EMA_BETA = 0.01  # residual variance can adapt a bit faster
MIN_STD = 1e-3


def doy_fourier(doy) -> tuple:
    """First two Fourier harmonics of day-of-year, used as the model's
    seasonal features. Works on scalars or arrays."""
    angle = 2 * np.pi * np.asarray(doy) / 365.25
    return np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)


class OnlineModel:
    """All learnable state for the full grid: per-(point, variable) linear
    weights, running normalization stats, residual variance (the Monte
    Carlo noise scale), and a short lag-value history buffer.
    """

    def __init__(self, n_points: int, variables: list[str]):
        self.n_points = n_points
        self.variables = list(variables)
        n_vars = len(self.variables)
        self.weights = np.zeros((n_points, n_vars, N_FEATURES), dtype=np.float64)
        self.run_mean = np.zeros((n_points, n_vars), dtype=np.float64)
        self.run_var = np.ones((n_points, n_vars), dtype=np.float64)
        self.resid_var = np.ones((n_points, n_vars), dtype=np.float64)  # normalized-space residual variance
        self.history = np.zeros((n_points, n_vars, N_LAGS), dtype=np.float64)  # normalized-space lag buffer
        self.initialized = np.zeros((n_points, n_vars), dtype=bool)  # seen >=1 real observation yet
        self.n_updates = np.zeros((n_points, n_vars), dtype=np.int64)

    def var_index(self, variable: str) -> int:
        return self.variables.index(variable)

    # -- feature construction ------------------------------------------------

    def _features(self, doy) -> np.ndarray:
        s1, c1, s2, c2 = doy_fourier(doy)
        n_vars = len(self.variables)
        X = np.empty((self.n_points, n_vars, N_FEATURES), dtype=np.float64)
        X[..., 0] = 1.0
        X[..., 1] = s1
        X[..., 2] = c1
        X[..., 3] = s2
        X[..., 4] = c2
        X[..., 5] = self.history[..., -1]  # lag-1 (normalized)
        X[..., 6] = self.history[..., 0]  # lag-7 (normalized)
        return X

    # -- online step -----------------------------------------------------

    def step(self, doy, actual_raw: np.ndarray) -> dict:
        """Advance the model by one calendar day.

        actual_raw: (n_points, n_vars) array of that day's real values,
        NaN where a point/variable has no observation for this day.
        Returns a compact metrics dict (MAE/RMSE overall, plus a
        per-variable MAE breakdown) computed only over points that had
        real data today.
        """
        mask = ~np.isnan(actual_raw)

        # Running mean/variance use a count-based effective beta: 1/(n+1)
        # for the first ~1000 observations of a (point, variable) pair
        # (an exact incremental average, so it converges from nothing in a
        # handful of samples instead of crawling there at the slow EMA
        # rate) and NORM_EMA_BETA after that (a slow long-run drift rate).
        # This matters more than it might look: seeding variance from a
        # single early sample is fragile -- an unlucky first value near
        # the variable's mean gives a near-zero std, which turns the next
        # real deviation into a huge normalized target and a runaway
        # weight update. The count-based ramp avoids ever trusting a
        # single sample's variance.
        self.initialized |= mask
        n_seen = self.n_updates.astype(np.float64)
        eff_beta = np.maximum(NORM_EMA_BETA, 1.0 / (n_seen + 1.0))

        delta = actual_raw - self.run_mean
        self.run_mean = np.where(mask, self.run_mean + eff_beta * delta, self.run_mean)
        self.run_var = np.where(
            mask, (1 - eff_beta) * self.run_var + eff_beta * delta ** 2, self.run_var
        )
        std = np.sqrt(np.maximum(self.run_var, MIN_STD ** 2))

        # Clipped defensively: even with the ramped-up variance estimate, a
        # single volatile observation could otherwise produce an outsized
        # normalized target (and, via the lag features, keep re-injecting
        # itself into later days) that destabilizes an otherwise
        # well-behaved online fit.
        actual_norm = np.where(mask, np.clip((actual_raw - self.run_mean) / std, -15.0, 15.0), 0.0)

        X = self._features(doy)
        pred_norm = np.einsum("pvf,pvf->pv", X, self.weights)
        error_norm = np.where(mask, np.clip(actual_norm - pred_norm, -12.0, 12.0), 0.0)

        self.weights += LEARNING_RATE * error_norm[..., None] * X
        self.weights = np.clip(self.weights, -50.0, 50.0)  # hard ceiling: belt-and-suspenders against runaway drift
        self.resid_var = np.where(
            mask, (1 - RESID_EMA_BETA) * self.resid_var + RESID_EMA_BETA * error_norm ** 2, self.resid_var
        )
        self.n_updates += mask.astype(np.int64)

        # Advance the lag-history ring buffer with today's normalized
        # value; where there's no observation, carry the last known value
        # forward instead of poisoning future lag features with a gap.
        next_val = np.where(mask, actual_norm, self.history[..., -1])
        self.history = np.roll(self.history, -1, axis=-1)
        self.history[..., -1] = next_val

        error_raw = error_norm * std  # de-normalized, for human-readable metrics
        abs_err = np.abs(error_raw)
        metrics = {
            "mae": float(np.mean(abs_err[mask])) if mask.any() else None,
            "rmse": float(np.sqrt(np.mean(error_raw[mask] ** 2))) if mask.any() else None,
            "n_obs": int(mask.sum()),
            "per_variable_mae": {
                var: float(np.mean(abs_err[mask[:, vi], vi]))
                for vi, var in enumerate(self.variables)
                if mask[:, vi].any()
            },
        }
        return metrics

    # -- checkpoint I/O -----------------------------------------------------

    def save(self, path: str) -> None:
        # Write through an explicit file handle (rather than a bare string
        # path, which np.savez_compressed would silently suffix with
        # .npz) and rename into place atomically, so a crash mid-write
        # never leaves a truncated checkpoint that a later resume would load.
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(
                fh,
                weights=self.weights,
                run_mean=self.run_mean,
                run_var=self.run_var,
                resid_var=self.resid_var,
                history=self.history,
                initialized=self.initialized,
                n_updates=self.n_updates,
                variables=np.array(self.variables),
            )
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "OnlineModel":
        with np.load(path, allow_pickle=False) as data:
            variables = [str(v) for v in data["variables"]]
            n_points = data["weights"].shape[0]
            model = cls(n_points, variables)
            model.weights = data["weights"]
            model.run_mean = data["run_mean"]
            model.run_var = data["run_var"]
            model.resid_var = data["resid_var"]
            model.history = data["history"]
            model.initialized = data["initialized"]
            model.n_updates = data["n_updates"]
        return model

    # -- forecasting inputs (used by Monte Carlo simulation) -----------------

    def std_for(self, point_idx: int) -> np.ndarray:
        """Residual std, in the model's normalized space, per variable at
        one point -- the noise scale Monte Carlo sampling draws from."""
        return np.sqrt(np.maximum(self.resid_var[point_idx], 1e-6))

    def mean_std_for(self, point_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """De-normalization (mean, std) per variable at one point."""
        return self.run_mean[point_idx], np.sqrt(np.maximum(self.run_var[point_idx], MIN_STD ** 2))
