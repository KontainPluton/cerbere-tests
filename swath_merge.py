"""
swath_merge.py -- Utilities for merging AVHRR L2P satellite swath tiles.

Provides two main functions:
  - concatenate_tiles_along_track: stitch sequential segments from one orbit
  - merge_tiles_to_regular_grid:  regrid onto a regular lat/lon grid with
    quality-aware mosaicking, multiple resampling methods, and antimeridian
    crossing support.

Dependencies: xarray, numpy, pyresample, tqdm (optional)
"""

import time
import numpy as np
import xarray as xr

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class _ProgressTracker:
    """
    Lightweight progress tracker.
    Uses tqdm when available, falls back to print-based progress.
    """

    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

        if HAS_TQDM:
            self.pbar = tqdm(total=total, desc=desc, unit="file")
        else:
            self.pbar = None
            print(f"{desc}: 0/{total} files")

    def update(self, filename=""):
        self.current += 1
        elapsed = time.time() - self.start_time

        if self.pbar is not None:
            self.pbar.set_postfix_str(filename, refresh=False)
            self.pbar.update(1)
        else:
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)
            pct = 100 * self.current / self.total
            print(
                f"  [{pct:5.1f}%] {self.current}/{self.total}"
                f" -- {filename}"
                f" -- elapsed {_fmt_time(elapsed)},"
                f" remaining ~{_fmt_time(remaining)}"
            )

    def close(self):
        elapsed = time.time() - self.start_time
        if self.pbar is not None:
            self.pbar.close()
        print(f"{self.desc}: done in {_fmt_time(elapsed)}")


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _detect_antimeridian_crossing(file_list):
    for f in file_list:
        with xr.open_dataset(f) as ds:
            lons = ds.lon.values
            if np.any(lons < -90) and np.any(lons > 90):
                return True
    return False


def _lon_to_360(lon):
    return np.where(lon < 0, lon + 360.0, lon)


def _lon_to_180(lon):
    return np.where(lon > 180, lon - 360.0, lon)

# ---------------------------------------------------------------------------
# 1. Along-track concatenation (preserves native curvilinear grid)
# ---------------------------------------------------------------------------

def concatenate_tiles_along_track(file_list, output_path=None):
    """
    Concatenate multiple NetCDF tiles along the track dimension (nj).

    Use this for sequential swath segments from the same orbit.
    The curvilinear grid structure is preserved.

    Parameters
    ----------
    file_list : list of str
        List of NetCDF file paths to concatenate.
    output_path : str, optional
        Path to save the merged NetCDF file.

    Returns
    -------
    xarray.Dataset
        Merged dataset with tiles concatenated along 'nj' dimension.
    """
    datasets = []
    for f in sorted(file_list):
        # mask_and_scale=False keeps raw integer values with encoding metadata
        # intact, avoiding a numpy 2.0 incompatibility where in-place *= on
        # int arrays with float64 scale_factor raises UFuncTypeError during
        # lazy xr.concat alignment.
        ds = xr.open_dataset(f, mask_and_scale=False)
        datasets.append(ds)

    merged = xr.concat(
        datasets,
        dim="nj",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

    for ds in datasets:
        ds.close()

    if output_path:
        for var in merged.data_vars:
            merged[var].encoding.clear()
        for coord in merged.coords:
            merged[coord].encoding.clear()
        merged.to_netcdf(output_path)
        print(f"Saved concatenated file to: {output_path}")

    return merged


# ---------------------------------------------------------------------------
# 2. Regridding onto a regular lat/lon grid
# ---------------------------------------------------------------------------

def merge_tiles_to_regular_grid(
        file_list,
        output_path=None,
        resolution=0.01,
        bbox=None,
        variables=None,
        radius_of_influence=10000,
        resample_method="nearest",
        sigma=5000,
        min_quality_level=0,
        output_lon_convention="[-180, 180]",
):
    """
    Merge multiple L2P NetCDF swath tiles onto a regular lat/lon grid.

    Uses pyresample to interpolate curvilinear swath data onto a regular grid.
    Handles swaths that cross the antimeridian automatically.

    Parameters
    ----------
    file_list : list of str
        List of NetCDF file paths to merge.
    output_path : str, optional
        Path to save the merged NetCDF file.
    resolution : float
        Target grid resolution in degrees (default: 0.01).
    bbox : tuple, optional
        Bounding box (min_lon, min_lat, max_lon, max_lat).
        If None, computed from input data.
        For antimeridian crossings you can pass e.g. (170, 50, 190, 65)
        or (170, 50, -170, 65) -- both are understood.
    variables : list of str, optional
        List of variables to include. If None, uses all numeric variables.
    radius_of_influence : float
        Maximum distance in meters for source pixel lookup (default: 10km).
    resample_method : str
        'nearest', 'gauss', or 'custom' (median).
    sigma : float
        Gaussian std dev in metres (only for resample_method='gauss').
    min_quality_level : int
        Minimum quality_level (0-5). 0 disables filtering.
    output_lon_convention : str
        '[-180, 180]' (default) or '[0, 360]'.

    Returns
    -------
    xarray.Dataset
        Merged dataset on a regular lat/lon grid.
    """
    from pyresample import geometry, kd_tree

    # -- Detect antimeridian crossing ------------------------------------
    crosses_antimeridian = _detect_antimeridian_crossing(file_list)
    if crosses_antimeridian:
        print("Antimeridian crossing detected -- working in [0, 360] longitude space.")

    # -- Determine bounding box ------------------------------------------
    if bbox is None:
        min_lon, min_lat = 360, 90
        max_lon, max_lat = 0, -90

        for f in file_list:
            with xr.open_dataset(f) as ds:
                lons = ds.lon.values
                if crosses_antimeridian:
                    lons = _lon_to_360(lons)
                min_lon = min(min_lon, float(np.nanmin(lons)))
                max_lon = max(max_lon, float(np.nanmax(lons)))
                min_lat = min(min_lat, float(ds.lat.min()))
                max_lat = max(max_lat, float(ds.lat.max()))

        bbox = (min_lon, min_lat, max_lon, max_lat)
    else:
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon > max_lon:
            crosses_antimeridian = True
            max_lon = max_lon + 360.0
            bbox = (min_lon, min_lat, max_lon, max_lat)
            print("Antimeridian crossing inferred from bbox -- working in [0, 360].")

    min_lon, min_lat, max_lon, max_lat = bbox

    # -- Build target regular grid ---------------------------------------
    target_lons = np.arange(min_lon, max_lon, resolution)
    target_lats = np.arange(min_lat, max_lat, resolution)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
    target_def = geometry.GridDefinition(lons=target_lon_grid, lats=target_lat_grid)

    ny, nx = len(target_lats), len(target_lons)
    print(f"Target grid size: {ny} x {nx} ({ny * nx:,} pixels)")

    # -- Discover variables ----------------------------------------------
    with xr.open_dataset(file_list[0]) as sample_ds:
        if variables is None:
            variables = [
                v for v in sample_ds.data_vars
                if sample_ds[v].dtype in [np.float32, np.float64, np.int16, np.int32]
                   and "nj" in sample_ds[v].dims
            ]
        var_attrs = {}
        for v in variables:
            attrs = dict(sample_ds[v].attrs)
            for key in ["scale_factor", "add_offset", "_FillValue", "valid_min", "valid_max"]:
                attrs.pop(key, None)
            var_attrs[v] = attrs

    print(f"Variables to regrid: {variables}")
    print(f"Resampling method: {resample_method}")

    # -- Initialise output arrays ----------------------------------------
    merged_data = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in variables}
    best_quality = np.full((ny, nx), -1.0, dtype=np.float32)

    # -- Process each file -----------------------------------------------
    progress = _ProgressTracker(len(file_list), desc="Regridding tiles")

    for f in file_list:
        fname = f.split("/")[-1]
        with xr.open_dataset(f) as ds:
            src_lons = ds.lon.values.copy()
            if crosses_antimeridian:
                src_lons = _lon_to_360(src_lons)

            swath_def = geometry.SwathDefinition(lons=src_lons, lats=ds.lat.values)

            # Quality level
            has_quality = "quality_level" in ds.data_vars
            if has_quality:
                ql = ds["quality_level"].values
                if ql.ndim == 3:
                    ql = ql[0]
                ql = ql.astype(np.float32)
                if min_quality_level > 0:
                    ql[ql < min_quality_level] = np.nan

                ql_resampled = kd_tree.resample_nearest(
                    swath_def, ql, target_def,
                    radius_of_influence=radius_of_influence,
                    fill_value=np.nan,
                )
            else:
                ql_resampled = np.ones((ny, nx), dtype=np.float32)

            # Resample each variable
            for var in variables:
                data = ds[var].values
                if data.ndim == 3:
                    data = data[0]
                data = data.astype(np.float32)

                if has_quality and min_quality_level > 0:
                    ql_src = ds["quality_level"].values
                    if ql_src.ndim == 3:
                        ql_src = ql_src[0]
                    data[ql_src < min_quality_level] = np.nan

                if resample_method == "gauss":
                    resampled = kd_tree.resample_gauss(
                        swath_def, data, target_def,
                        radius_of_influence=radius_of_influence,
                        sigmas=sigma,
                        fill_value=np.nan,
                    )
                elif resample_method == "custom":
                    resampled = kd_tree.resample_custom(
                        swath_def, data, target_def,
                        radius_of_influence=radius_of_influence,
                        fill_value=np.nan,
                        weight_funcs=None,
                        reduce_func=np.nanmedian,
                    )
                else:
                    resampled = kd_tree.resample_nearest(
                        swath_def, data, target_def,
                        radius_of_influence=radius_of_influence,
                        fill_value=np.nan,
                    )

                valid_new = ~np.isnan(resampled)
                empty_target = np.isnan(merged_data[var])
                better_quality = ql_resampled > best_quality
                update_mask = valid_new & (empty_target | better_quality)
                merged_data[var][update_mask] = resampled[update_mask]

            valid_ql = ~np.isnan(ql_resampled)
            improve = valid_ql & (ql_resampled > best_quality)
            best_quality[improve] = ql_resampled[improve]

        progress.update(fname)

    progress.close()

    # -- Convert longitudes back if needed -------------------------------
    output_lons = target_lons.copy()
    if crosses_antimeridian and output_lon_convention == "[-180, 180]":
        output_lons = _lon_to_180(output_lons)
        sort_idx = np.argsort(output_lons)
        output_lons = output_lons[sort_idx]
        for var in variables:
            merged_data[var] = merged_data[var][:, sort_idx]
        best_quality = best_quality[:, sort_idx]

    # -- Build output dataset --------------------------------------------
    data_vars = {
        var: (["lat", "lon"], merged_data[var], var_attrs.get(var, {}))
        for var in variables
    }
    data_vars["mosaic_quality_level"] = (
        ["lat", "lon"],
        np.where(best_quality < 0, np.nan, best_quality),
        {
            "long_name": "Best quality_level used during mosaicking",
            "comment": f"Minimum accepted quality_level was {min_quality_level}",
        },
    )

    merged_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": (["lat"], target_lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": (["lon"], output_lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "Merged satellite data (quality-aware regridding)",
            "source_files": ", ".join([f.split("/")[-1] for f in file_list]),
            "resolution": f"{resolution} degrees",
            "resample_method": resample_method,
            "min_quality_level": str(min_quality_level),
            "antimeridian_crossing": str(crosses_antimeridian),
        },
    )

    if output_path:
        print(f"Writing output to: {output_path}")
        merged_ds.to_netcdf(output_path)
        print(f"Saved merged file to: {output_path}")

    return merged_ds