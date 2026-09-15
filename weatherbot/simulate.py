"""Monte Carlo ensemble forecasting from a trained OnlineModel checkpoint.

Given one grid point and a forecast horizon, simulates many independent
future paths day by day: each path's next value is the model's
seasonal+lag prediction (the deterministic component) plus Gaussian noise
drawn from that point/variable's tracked residual variance, and that noisy
value is fed back in as the path's own lag-1 for the following day -- so
paths diverge over the horizon the way real forecast uncertainty compounds,
rather than all reusing the same central forecast. The result is
summarized into an ensemble mean and percentile bands per forecast day,
the same "fan chart" shape used for probabilistic financial forecasts.

This is intentionally a small, on-demand computation (one point, a couple
of hundred paths, a couple of weeks) rather than something run for the
whole grid -- it's meant to be cheap enough to re-run interactively from
the dashboard on a small droplet.

Implemented as a generator so a caller (the web dashboard) can stream one
frame per simulated day -- for watching the ensemble build up live -- and
can cancel a run in progress via should_stop().
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from .model import N_FEATURES, OnlineModel, doy_fourier

DEFAULT_N_PATHS = 200
MAX_N_PATHS = 500  # resource ceiling for a small droplet -- see README
DEFAULT_HORIZON_DAYS = 14
MAX_HORIZON_DAYS = 60
PERCENTILES = (5, 25, 50, 75, 95)
MIN_OBSERVATIONS_FOR_CONFIDENCE = 30  # below this, results carry a "still warming up" warning


@dataclass
class SimFrame:
    day_offset: int  # 1-indexed day into the horizon
    date: str
    paths: dict  # {variable: (n_paths,) values for this specific day}
    percentiles: dict  # {variable: {percentile_str: value}} for this day
    done: bool


def nearest_point_idx(lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> int:
    """Nearest grid point by simple planar distance -- adequate for
    picking among points on a 0.5-degree CONUS grid, not meant for precise
    geodesy."""
    d2 = (lats - lat) ** 2 + (lons - lon) ** 2
    return int(np.argmin(d2))


def min_observations(model: OnlineModel, point_idx: int, variables: list[str]) -> int:
    idxs = [model.var_index(v) for v in variables]
    return int(model.n_updates[point_idx, idxs].min()) if idxs else 0


def run_monte_carlo(
    model: OnlineModel,
    point_idx: int,
    start_date: _dt.date,
    variables: list[str],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_paths: int = DEFAULT_N_PATHS,
    seed: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator[SimFrame]:
    horizon_days = max(1, min(horizon_days, MAX_HORIZON_DAYS))
    n_paths = max(1, min(n_paths, MAX_N_PATHS))
    rng = np.random.default_rng(seed)

    var_idx = [model.var_index(v) for v in variables]
    mean, std = model.mean_std_for(point_idx)  # (n_vars_all,), physical-unit de-normalization
    resid_std = model.std_for(point_idx)  # (n_vars_all,), normalized-space noise scale

    # Per-path lag history, seeded from the model's current lag buffer at
    # this point so day 1 of the forecast continues smoothly from the real
    # data the model last trained on.
    history = np.repeat(model.history[point_idx][None, :, :], n_paths, axis=0)  # (n_paths, n_vars_all, N_LAGS)
    weights = model.weights[point_idx]  # (n_vars_all, N_FEATURES)
    n_vars_all = len(model.variables)

    for day in range(horizon_days):
        if should_stop is not None and should_stop():
            return
        date = start_date + _dt.timedelta(days=day + 1)
        s1, c1, s2, c2 = doy_fourier(date.timetuple().tm_yday)

        X = np.empty((n_paths, n_vars_all, N_FEATURES), dtype=np.float64)
        X[..., 0] = 1.0
        X[..., 1] = s1
        X[..., 2] = c1
        X[..., 3] = s2
        X[..., 4] = c2
        X[..., 5] = history[..., -1]
        X[..., 6] = history[..., 0]

        pred_norm = np.einsum("nvf,vf->nv", X, weights)  # (n_paths, n_vars_all)
        noise = rng.normal(0.0, 1.0, size=pred_norm.shape) * resid_std[None, :]
        # Same defensive clip as the training step (model.py) -- keeps one
        # noisy day from producing an unbounded lag feature that a later
        # simulated day would otherwise compound.
        sample_norm = np.clip(pred_norm + noise, -15.0, 15.0)

        history = np.roll(history, -1, axis=-1)
        history[..., -1] = sample_norm

        sample_raw = sample_norm * std[None, :] + mean[None, :]  # (n_paths, n_vars_all)

        paths_frame = {}
        pct_frame = {}
        for vi, v in zip(var_idx, variables):
            values = sample_raw[:, vi]
            paths_frame[v] = values
            pct_frame[v] = {str(p): float(np.percentile(values, p)) for p in PERCENTILES}

        yield SimFrame(
            day_offset=day + 1,
            date=date.isoformat(),
            paths=paths_frame,
            percentiles=pct_frame,
            done=(day == horizon_days - 1),
        )
