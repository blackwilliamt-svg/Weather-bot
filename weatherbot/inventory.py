"""Read-only inspection of a WeatherBot zarr store.

Reports point count/coverage, date range, variables, and on-disk size without
loading any actual weather data into memory. This is the intended entry point
for future phases (training/simulation/dashboard) to open the store lazily.
"""
from __future__ import annotations

import os

import xarray as xr


def open_store(store_path: str) -> xr.Dataset:
    """Lazily open the store. Data variables stay dask-backed — nothing is
    loaded into memory until the caller explicitly computes/indexes them.
    """
    return xr.open_zarr(store_path, consolidated=True)


def _dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


def summarize(store_path: str, list_points: bool = False) -> str:
    ds = open_store(store_path)
    n_points = ds.sizes["point"]
    n_time = ds.sizes["time"]
    time_min = ds["time"].min().values
    time_max = ds["time"].max().values
    variables = list(ds.data_vars)
    size_bytes = _dir_size_bytes(store_path)
    size_gb = size_bytes / (1024 ** 3)

    lines = [
        f"Store: {store_path}",
        f"Grid points: {n_points}",
        f"Date range: {str(time_min)[:10]} .. {str(time_max)[:10]} ({n_time} days)",
        f"Variables ({len(variables)}): {', '.join(variables)}",
        f"Size on disk: {size_gb:.3f} GB ({size_bytes:,} bytes)",
    ]
    if list_points:
        lats = ds["lat"].values
        lons = ds["lon"].values
        lines.append("Points:")
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            lines.append(f"  {i:>6}: ({lat:.4f}, {lon:.4f})")
    ds.close()
    return "\n".join(lines)
