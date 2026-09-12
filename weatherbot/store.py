"""Zarr archive template creation and region writes.

Uses the standard xarray "preallocate template, then region write" pattern:
init_store() declares the full (point, time) shape and dtype/compression via
a lazy dask-backed all-NaN template (metadata only, no chunk files touch disk
until actually written); write_region() fills in real data batch by batch;
finalize_store() consolidates metadata once at the end.
"""
from __future__ import annotations

import datetime as _dt
import os

import dask.array as da
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .config import Config


def build_time_index(config: Config, end_date: _dt.date | None = None) -> pd.DatetimeIndex:
    end = end_date or config.end_date()
    return pd.date_range(config.archive_start_date, end, freq="D")


def date_to_offset(date: _dt.date, config: Config) -> int:
    return (date - config.archive_start_date).days


def init_store(store_path: str, points: pd.DataFrame, time_index: pd.DatetimeIndex, config: Config) -> None:
    """Create (or overwrite) the zarr store shell: full shape, dtype, chunking,
    and compression declared, but no real data written yet.
    """
    n_points = len(points)
    n_time = len(time_index)
    chunks = (min(config.point_chunk, n_points), min(config.time_chunk_days, n_time))

    data_vars = {}
    for var in config.variables:
        arr = da.full((n_points, n_time), np.nan, dtype=np.float32, chunks=chunks)
        data_vars[var] = xr.DataArray(arr, dims=("point", "time"))

    ds = xr.Dataset(
        data_vars,
        coords={
            "point": np.arange(n_points, dtype=np.int64),
            "time": time_index,
            "lat": ("point", points["lat"].to_numpy(dtype=np.float64)),
            "lon": ("point", points["lon"].to_numpy(dtype=np.float64)),
        },
    )
    ds.attrs["archive_start_date"] = config.archive_start_date.isoformat()
    ds.attrs["spacing_deg"] = config.spacing_deg
    ds.attrs["created_utc"] = _dt.datetime.utcnow().isoformat()

    encoding = {
        var: {
            "compressor": numcodecs.Zstd(level=config.zstd_level),
            "dtype": "float32",
            "chunks": chunks,
        }
        for var in config.variables
    }

    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    ds.to_zarr(store_path, mode="w", compute=False, encoding=encoding, consolidated=True)


def write_region(
    store_path: str,
    data: dict[str, np.ndarray],
    point_slice: slice,
    time_slice: slice,
) -> None:
    """Write one real batch of float32 data into the store.

    `data` maps variable name -> 2D float32 array shaped
    (point_slice length, time_slice length).
    """
    n_points = point_slice.stop - point_slice.start
    n_time = time_slice.stop - time_slice.start
    data_vars = {
        var: (("point", "time"), arr.astype(np.float32))
        for var, arr in data.items()
    }
    ds = xr.Dataset(data_vars)
    if any(arr.shape != (n_points, n_time) for arr in data.values()):
        raise ValueError("write_region: data shape does not match point/time slice sizes")
    ds.to_zarr(
        store_path,
        region={"point": point_slice, "time": time_slice},
        consolidated=False,
    )


def finalize_store(store_path: str) -> None:
    zarr.consolidate_metadata(store_path)
