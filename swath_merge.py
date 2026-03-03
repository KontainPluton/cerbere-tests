"""
swath_merge.py — Utilities for merging AVHRR L2P satellite swath tiles.

Provides two main functions:
  - concatenate_tiles_along_track: stitch sequential segments from one orbit
  - merge_tiles_to_regular_grid:  regrille onto a regular lat/lon grid with
    quality-aware mosaicking and proper fill-value handling.

Dependencies: xarray, numpy, pyresample
"""

import numpy as np
import xarray as xr


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

def _mask_variable(data, ds, var_name, min_quality_level):
    """
    Mask invalid pixels in *data* (2-D float32 array) **in-place**.

    Three masking steps are applied in order:

    1. **Fill-value masking** — Any pixel whose raw value equals the NetCDF
       ``_FillValue`` (before or after scale/offset) is set to NaN.
       Also masks common sentinel values (-999, -32767, -32768) that some
       L2P products use when ``_FillValue`` is absent.

    2. **sea_surface_temperature range check** — If *var_name* looks like an
       SST field, values outside [–5 °C, +45 °C] (≈ 268–318 K) are masked.

    3. **Quality-level filtering** — If the dataset contains a
       ``quality_level`` variable **and** *min_quality_level* > 0,
       every pixel whose quality is below the threshold is masked.

    Parameters
    ----------
    data : np.ndarray (float32, 2-D)
        The array to mask.  Modified **in-place**.
    ds : xarray.Dataset
        The opened source dataset (used to read attributes and quality_level).
    var_name : str
        Name of the variable being processed.
    min_quality_level : int
        Minimum acceptable quality level (0–5).  0 disables the filter.

    Returns
    -------
    np.ndarray
        The same *data* array (modified in-place), for convenience.
    """

    # --- 1. Fill-value masking -------------------------------------------
    fill_value = ds[var_name].attrs.get("_FillValue", None)
    if fill_value is not None:
        data[data == np.float32(fill_value)] = np.nan

    # Catch common sentinel values that slip through
    for sentinel in (-999.0, -32767.0, -32768.0):
        data[data == np.float32(sentinel)] = np.nan

    # --- 2. Physical range check for SST --------------------------------
    sst_keywords = ("sea_surface_temperature", "sst", "analysed_sst")
    if any(kw in var_name.lower() for kw in sst_keywords):
        data[(data < -5.0) | (data > 45.0)] = np.nan   # Celsius
        data[(data < 268.0) & (data > 45.0)] = np.nan   # Kelvin (safety)
        data[data > 318.0] = np.nan

    # --- 3. Quality-level filtering -------------------------------------
    if min_quality_level > 0 and "quality_level" in ds.data_vars:
        ql = ds["quality_level"].values
        if ql.ndim == 3:
            ql = ql[0]
        data[ql < min_quality_level] = np.nan

    return data


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
):
    """
    Merge multiple L2P NetCDF swath tiles onto a regular lat/lon grid.

    Uses pyresample to interpolate curvilinear swath data onto a regular grid.
    Good for overlapping tiles or when a uniform grid output is needed.

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
    variables : list of str, optional
        List of variables to include. If None, uses all numeric variables.
    radius_of_influence : float
        Maximum distance in meters to look for source pixels (default: 10km).
    resample_method : str
        Resampling method: 'nearest', 'gauss', or 'custom'.
        - 'nearest': nearest neighbour (fast, default)
        - 'gauss': gaussian-weighted average (smoother, good for swath edges)
        - 'custom': custom reduce function using median (robust to outliers)
    sigma : float
        Standard deviation in metres for the Gaussian weight function
        (only used when resample_method='gauss'). Default 5000.
    min_quality_level : int
        Minimum quality_level a pixel must have to be used (0-5).
        Set to 0 to disable quality filtering (default).

    Returns
    -------
    xarray.Dataset
        Merged dataset on a regular lat/lon grid.
    """
    from pyresample import geometry, kd_tree

    # First pass: determine bounding box and collect metadata
    if bbox is None:
        min_lon, min_lat = 180, 90
        max_lon, max_lat = -180, -90

        for f in file_list:
            with xr.open_dataset(f) as ds:
                min_lon = min(min_lon, float(ds.lon.min()))
                max_lon = max(max_lon, float(ds.lon.max()))
                min_lat = min(min_lat, float(ds.lat.min()))
                max_lat = max(max_lat, float(ds.lat.max()))

        bbox = (min_lon, min_lat, max_lon, max_lat)

    min_lon, min_lat, max_lon, max_lat = bbox

    # Create target regular grid
    target_lons = np.arange(min_lon, max_lon, resolution)
    target_lats = np.arange(min_lat, max_lat, resolution)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

    target_def = geometry.GridDefinition(lons=target_lon_grid, lats=target_lat_grid)

    # Initialize output arrays with NaN
    ny, nx = len(target_lats), len(target_lons)

    # Determine variables to process
    with xr.open_dataset(file_list[0]) as sample_ds:
        if variables is None:
            variables = [
                v for v in sample_ds.data_vars
                if sample_ds[v].dtype in [np.float32, np.float64, np.int16, np.int32]
                   and "nj" in sample_ds[v].dims
            ]

        # Get variable attributes for output (filter out problematic attributes)
        var_attrs = {}
        for v in variables:
            attrs = dict(sample_ds[v].attrs)
            for key in ["scale_factor", "add_offset", "_FillValue", "valid_min", "valid_max"]:
                attrs.pop(key, None)
            var_attrs[v] = attrs

    # Initialize merged arrays and quality tracker
    merged_data = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in variables}
    best_quality = np.full((ny, nx), -1.0, dtype=np.float32)

    # Process each file
    for f in file_list:
        print(f"Processing: {f.split('/')[-1]}")
        with xr.open_dataset(f) as ds:
            # Define source swath geometry
            swath_def = geometry.SwathDefinition(lons=ds.lon.values, lats=ds.lat.values)

            # --- Regrid quality_level for mosaicking logic ---------------
            has_quality = "quality_level" in ds.data_vars
            if has_quality:
                ql = ds["quality_level"].values
                if ql.ndim == 3:
                    ql = ql[0]
                ql = ql.astype(np.float32)
                # Mask pixels below minimum quality before resampling
                if min_quality_level > 0:
                    ql[ql < min_quality_level] = np.nan

                ql_resampled = kd_tree.resample_nearest(
                    swath_def, ql, target_def,
                    radius_of_influence=radius_of_influence,
                    fill_value=np.nan,
                )
            else:
                ql_resampled = np.ones((ny, nx), dtype=np.float32)

            # --- Resample each science variable -------------------------
            for var in variables:
                data = ds[var].values
                if data.ndim == 3:
                    data = data[0]

                data = data.astype(np.float32)

                # Apply quality mask to source data before resampling
                if has_quality and min_quality_level > 0:
                    ql_src = ds["quality_level"].values
                    if ql_src.ndim == 3:
                        ql_src = ql_src[0]
                    data[ql_src < min_quality_level] = np.nan

                # Resample to target grid
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

                # Quality-aware mosaicking:
                # - If no data yet at this pixel -> fill unconditionally
                # - If data exists but new quality is strictly higher -> overwrite
                valid_new = ~np.isnan(resampled)
                empty_target = np.isnan(merged_data[var])
                better_quality = ql_resampled > best_quality

                update_mask = valid_new & (empty_target | better_quality)
                merged_data[var][update_mask] = resampled[update_mask]

            # Update quality tracker once per tile
            valid_ql = ~np.isnan(ql_resampled)
            improve = valid_ql & (ql_resampled > best_quality)
            best_quality[improve] = ql_resampled[improve]

    # Build output dataset
    data_vars = {
        var: (["lat", "lon"], merged_data[var], var_attrs.get(var, {}))
        for var in variables
    }

    # Include quality map for traceability
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
            "lon": (["lon"], target_lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "Merged satellite data (quality-aware regridding)",
            "source_files": ", ".join([f.split("/")[-1] for f in file_list]),
            "resolution": f"{resolution} degrees",
            "resample_method": resample_method,
            "min_quality_level": str(min_quality_level),
        },
    )

    if output_path:
        merged_ds.to_netcdf(output_path)
        print(f"Saved merged file to: {output_path}")

    return merged_ds


def merge_tiles_to_regular_grid_OLD(file_list, output_path=None,
                                resolution=0.01,
                                bbox=None,
                                variables=None,
                                radius_of_influence=10000):
    """
    Merge multiple NetCDF tiles onto a common regular lat/lon grid.

    Uses pyresample to interpolate curvilinear swath data onto a regular grid.
    Good for overlapping tiles or when a uniform grid output is needed.

    Parameters:
    -----------
    file_list : list of str
        List of NetCDF file paths to merge
    output_path : str, optional
        Path to save the merged NetCDF file
    resolution : float
        Target grid resolution in degrees (default: 0.01)
    bbox : tuple, optional
        Bounding box (min_lon, min_lat, max_lon, max_lat).
        If None, computed from input data.
    variables : list of str, optional
        List of variables to include. If None, uses all numeric variables.
    radius_of_influence : float
        Maximum distance in meters to look for source pixels (default: 10km)

    Returns:
    --------
    xarray.Dataset
        Merged dataset on a regular lat/lon grid
    """
    from pyresample import geometry, kd_tree

    # First pass: determine bounding box and collect metadata
    if bbox is None:
        min_lon, min_lat = 180, 90
        max_lon, max_lat = -180, -90

        for f in file_list:
            with xr.open_dataset(f) as ds:
                min_lon = min(min_lon, float(ds.lon.min()))
                max_lon = max(max_lon, float(ds.lon.max()))
                min_lat = min(min_lat, float(ds.lat.min()))
                max_lat = max(max_lat, float(ds.lat.max()))

        bbox = (min_lon, min_lat, max_lon, max_lat)

    min_lon, min_lat, max_lon, max_lat = bbox

    # Create target regular grid
    target_lons = np.arange(min_lon, max_lon, resolution)
    target_lats = np.arange(min_lat, max_lat, resolution)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

    target_def = geometry.GridDefinition(lons=target_lon_grid, lats=target_lat_grid)

    # Initialize output arrays with NaN
    ny, nx = len(target_lats), len(target_lons)

    # Determine variables to process
    with xr.open_dataset(file_list[0]) as sample_ds:
        if variables is None:
            # Get all 2D/3D numeric variables
            variables = [v for v in sample_ds.data_vars
                         if sample_ds[v].dtype in [np.float32, np.float64, np.int16, np.int32]
                         and 'nj' in sample_ds[v].dims]

        # Get variable attributes for output (filter out problematic attributes)
        var_attrs = {}
        for v in variables:
            attrs = dict(sample_ds[v].attrs)
            # Remove encoding-related attributes that can cause issues
            for key in ['scale_factor', 'add_offset', '_FillValue', 'valid_min', 'valid_max']:
                attrs.pop(key, None)
            var_attrs[v] = attrs

    # Initialize merged arrays
    merged_data = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in variables}

    # Process each file
    for f in file_list:
        print(f"Processing: {f.split('/')[-1]}")
        with xr.open_dataset(f) as ds:
            # Define source swath geometry
            swath_def = geometry.SwathDefinition(lons=ds.lon.values, lats=ds.lat.values)

            for var in variables:
                # Get data (squeeze time dimension if present)
                data = ds[var].values
                if data.ndim == 3:
                    data = data[0]  # Take first time slice

                # Resample to target grid
                resampled = kd_tree.resample_nearest(
                    swath_def, data.astype(np.float32), target_def,
                    radius_of_influence=radius_of_influence,
                    fill_value=np.nan
                )

                # Merge: use new data where we don't have data yet
                valid_mask = ~np.isnan(resampled) & np.isnan(merged_data[var])
                merged_data[var][valid_mask] = resampled[valid_mask]

    # Create output dataset
    merged_ds = xr.Dataset(
        data_vars={
            var: (['lat', 'lon'], merged_data[var], var_attrs.get(var, {}))
            for var in variables
        },
        coords={
            'lat': (['lat'], target_lats, {'units': 'degrees_north', 'standard_name': 'latitude'}),
            'lon': (['lon'], target_lons, {'units': 'degrees_east', 'standard_name': 'longitude'})
        },
        attrs={
            'title': 'Merged satellite data',
            'source_files': ', '.join([f.split('/')[-1] for f in file_list]),  # String, not list
            'resolution': f'{resolution} degrees'
        }
    )

    if output_path:
        merged_ds.to_netcdf(output_path)
        print(f"Saved merged file to: {output_path}")

    return merged_ds