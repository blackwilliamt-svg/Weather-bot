"""Contiguous U.S. (CONUS) land grid point generation.

Builds a lat/lon grid over a bounding box at a configurable spacing, then
filters to land points using global_land_mask (no shapefile download needed).
Points are sorted deterministically so point indices are stable across runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from global_land_mask import globe

from .config import Config


def generate_full_grid(config: Config) -> pd.DataFrame:
    """Return a DataFrame of land grid points with columns [point_id, lat, lon].

    point_id is a stable sequential index (0..N-1) assigned after sorting by
    (lat, lon), so the same config always yields the same ordering.
    """
    lat_min, lat_max, lon_min, lon_max = config.bbox
    lats = np.arange(lat_min, lat_max + config.spacing_deg / 2, config.spacing_deg)
    lons = np.arange(lon_min, lon_max + config.spacing_deg / 2, config.spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    is_land = globe.is_land(lat_flat, lon_flat)
    lat_land = lat_flat[is_land]
    lon_land = lon_flat[is_land]

    df = pd.DataFrame({"lat": lat_land, "lon": lon_land})
    df = df.sort_values(["lat", "lon"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "point_id", np.arange(len(df)))
    return df


def generate_test_grid(config: Config, n_points: int) -> pd.DataFrame:
    """Deterministic, geographically-spread subset of the full grid for test mode."""
    full = generate_full_grid(config)
    if n_points >= len(full):
        return full
    idx = np.linspace(0, len(full) - 1, num=n_points, dtype=int)
    idx = np.unique(idx)
    sample = full.iloc[idx].reset_index(drop=True)
    sample["point_id"] = np.arange(len(sample))
    return sample
