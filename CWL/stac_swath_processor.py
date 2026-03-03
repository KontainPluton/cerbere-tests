"""
stac_swath_processor.py — Download STAC L2P swath tiles and merge them
onto a regular lat/lon grid.

Designed to run inside a CWL container. Accepts CLI arguments for:
  --stac_url          : Root STAC API URL (e.g. https://stac.ifremer.fr)
  --collection        : Collection ID (e.g. AVHRR_SST_METOP_B_OSISAF_L2P_v1_0)
  --bbox              : Bounding box as "min_lon,min_lat,max_lon,max_lat"
  --temporal_extent   : Temporal range as "start_datetime/end_datetime"
  --properties        : Comma-separated list of variables to regrid
  --resolution        : Target grid resolution in degrees (default: 0.01)
  --resample_method   : nearest | gauss | custom (default: nearest)
  --min_quality_level : Minimum quality level 0-5 (default: 0)
  --output_dir        : Output directory (default: current directory)

Dependencies: requests, numpy, xarray, pyresample, netCDF4
"""

import sys
import os
import json
import argparse
import time
import shutil
from urllib.parse import urlparse, urljoin

import requests
import numpy as np
import xarray as xr


# =========================================================================
# Progress tracker
# =========================================================================

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class _ProgressTracker:
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
            avg = elapsed / self.current
            remaining = avg * (self.total - self.current)
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


# =========================================================================
# STAC search and download
# =========================================================================

def search_stac_items(stac_url, collection, bbox, temporal_extent):
    """
    Search a STAC API for items matching bbox and temporal extent.

    Handles pagination automatically via the 'next' link.

    Parameters
    ----------
    stac_url : str
        Root URL of the STAC API (e.g. https://stac.ifremer.fr).
    collection : str
        Collection ID.
    bbox : list of float
        [min_lon, min_lat, max_lon, max_lat].
    temporal_extent : str
        ISO 8601 interval, e.g. "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z".

    Returns
    -------
    list of dict
        List of STAC Item dicts (GeoJSON Features).
    """
    # Try the /search endpoint first (STAC API - Item Search)
    search_url = stac_url.rstrip("/") + "/search"

    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": temporal_extent,
        "limit": 100,
    }

    all_items = []
    page = 1

    print(f"Searching STAC API: {search_url}")
    print(f"  Collection : {collection}")
    print(f"  BBox       : {bbox}")
    print(f"  Datetime   : {temporal_extent}")

    while True:
        print(f"  Fetching page {page}...")

        try:
            resp = requests.post(search_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError:
            # Fallback: some STAC servers only support GET on /collections/{id}/items
            if page == 1:
                print("  POST /search failed, falling back to GET /collections/.../items")
                return _search_stac_items_via_ogcapi(
                    stac_url, collection, bbox, temporal_extent
                )
            else:
                raise
        except Exception as e:
            print(f"ERROR: STAC search failed: {e}", file=sys.stderr)
            sys.exit(1)

        features = data.get("features", [])
        all_items.extend(features)
        print(f"  Got {len(features)} items (total: {len(all_items)})")

        # Pagination: look for a 'next' link
        next_link = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_link = link
                break

        if next_link is None:
            break

        # Follow next link
        next_href = next_link.get("href")
        next_method = next_link.get("method", "GET").upper()
        next_body = next_link.get("body")

        if next_method == "POST" and next_body:
            payload = next_body
        elif next_method == "POST":
            # Some APIs use a token-based approach
            next_token = next_link.get("body", {}).get("token") or next_link.get("token")
            if next_token:
                payload["token"] = next_token
            else:
                break
        else:
            # GET-based pagination — switch to GET
            try:
                resp = requests.get(next_href, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                all_items.extend(features)
                print(f"  Got {len(features)} items (total: {len(all_items)})")
                # Continue pagination
                next_link = None
                for link in data.get("links", []):
                    if link.get("rel") == "next":
                        next_link = link
                        break
                if next_link is None:
                    break
                continue
            except Exception:
                break

        page += 1

    print(f"Total items found: {len(all_items)}")
    return all_items


def _search_stac_items_via_ogcapi(stac_url, collection, bbox, temporal_extent):
    """
    Fallback: search via OGC API Features
    GET /collections/{collection}/items?bbox=...&datetime=...
    """
    items_url = (
        stac_url.rstrip("/")
        + f"/collections/{collection}/items"
    )

    params = {
        "bbox": ",".join(str(b) for b in bbox),
        "datetime": temporal_extent,
        "limit": 100,
    }

    all_items = []
    page = 1

    while items_url:
        print(f"  Fetching page {page} (OGC API)...")
        try:
            resp = requests.get(items_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"ERROR: Failed to fetch items: {e}", file=sys.stderr)
            sys.exit(1)

        features = data.get("features", [])
        all_items.extend(features)
        print(f"  Got {len(features)} items (total: {len(all_items)})")

        # Pagination
        items_url = None
        params = {}  # Reset params for next pages (URL already contains them)
        for link in data.get("links", []):
            if link.get("rel") == "next":
                items_url = link.get("href")
                break

        page += 1

    print(f"Total items found: {len(all_items)}")
    return all_items


def get_download_url(item):
    """
    Extract the download URL from a STAC item.
    Prioritises IFREMER alternate links, then falls back to primary href.
    """
    assets = item.get("assets", {})

    for key, asset_metadata in assets.items():
        roles = asset_metadata.get("roles", [])
        if "data" in roles:
            # Check for IFREMER alternate link
            alternates = asset_metadata.get("alternate", {})
            if "HTTPS_IFREMER" in alternates:
                ifremer_href = alternates["HTTPS_IFREMER"].get("href")
                if ifremer_href:
                    return ifremer_href

            # Fallback to primary href
            href = asset_metadata.get("href")
            if href:
                return href

    # Last resort: try any asset with an href
    for key, asset_metadata in assets.items():
        href = asset_metadata.get("href")
        if href:
            return href

    return None


def download_file(url, output_dir):
    """Download a file to output_dir, using /tmp as buffer."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path.strip("/"))
    if not filename or "." not in filename:
        filename = "downloaded_asset.nc"

    temp_path = os.path.join("/tmp", filename)
    dest_path = os.path.join(output_dir, filename)

    if os.path.exists(dest_path):
        return dest_path

    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        shutil.move(temp_path, dest_path)
        return dest_path
    except Exception as e:
        print(f"ERROR: Failed to download {url}: {e}", file=sys.stderr)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


# =========================================================================
# Antimeridian helpers
# =========================================================================

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


# =========================================================================
# Regridding
# =========================================================================

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
    See swath_merge.py for full docstring.
    """
    from pyresample import geometry, kd_tree

    # -- Detect antimeridian crossing
    crosses_antimeridian = _detect_antimeridian_crossing(file_list)
    if crosses_antimeridian:
        print("Antimeridian crossing detected -- working in [0, 360] longitude space.")

    # -- Determine bounding box
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

    # -- Build target regular grid
    target_lons = np.arange(min_lon, max_lon, resolution)
    target_lats = np.arange(min_lat, max_lat, resolution)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
    target_def = geometry.GridDefinition(lons=target_lon_grid, lats=target_lat_grid)
    ny, nx = len(target_lats), len(target_lons)
    print(f"Target grid size: {ny} x {nx} ({ny * nx:,} pixels)")

    # -- Discover variables
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

    # -- Initialise output
    merged_data = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in variables}
    best_quality = np.full((ny, nx), -1.0, dtype=np.float32)

    # -- Process each file
    progress = _ProgressTracker(len(file_list), desc="Regridding tiles")

    for f in file_list:
        fname = os.path.basename(f)
        with xr.open_dataset(f) as ds:
            src_lons = ds.lon.values.copy()
            if crosses_antimeridian:
                src_lons = _lon_to_360(src_lons)

            swath_def = geometry.SwathDefinition(lons=src_lons, lats=ds.lat.values)

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
                        sigmas=sigma, fill_value=np.nan,
                    )
                elif resample_method == "custom":
                    resampled = kd_tree.resample_custom(
                        swath_def, data, target_def,
                        radius_of_influence=radius_of_influence,
                        fill_value=np.nan, weight_funcs=None,
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

    # -- Convert longitudes back if needed
    output_lons = target_lons.copy()
    if crosses_antimeridian and output_lon_convention == "[-180, 180]":
        output_lons = _lon_to_180(output_lons)
        sort_idx = np.argsort(output_lons)
        output_lons = output_lons[sort_idx]
        for var in variables:
            merged_data[var] = merged_data[var][:, sort_idx]
        best_quality = best_quality[:, sort_idx]

    # -- Build output dataset
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
            "source_files": ", ".join(os.path.basename(f) for f in file_list),
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


# =========================================================================
# Main CLI
# =========================================================================

def parse_bbox(bbox_str):
    """Parse bbox string 'min_lon,min_lat,max_lon,max_lat' into list of floats."""
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}: {bbox_str}")
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Download STAC L2P swath tiles and merge onto a regular grid."
    )
    parser.add_argument("--stac_url", required=True,
                        help="Root STAC API URL")
    parser.add_argument("--collection", required=True,
                        help="STAC Collection ID")
    parser.add_argument("--bbox", required=True,
                        help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--temporal_extent", required=True,
                        help="Temporal range: start/end in ISO 8601")
    parser.add_argument("--properties", required=False, default=None,
                        help="Comma-separated variable names to regrid (default: all)")
    parser.add_argument("--resolution", type=float, default=0.01,
                        help="Target grid resolution in degrees (default: 0.01)")
    parser.add_argument("--resample_method", default="nearest",
                        choices=["nearest", "gauss", "custom"],
                        help="Resampling method (default: nearest)")
    parser.add_argument("--min_quality_level", type=int, default=0,
                        help="Minimum quality_level 0-5 (default: 0)")
    parser.add_argument("--output_dir", default=".",
                        help="Output directory (default: current directory)")

    args = parser.parse_args()

    # Parse inputs
    bbox = parse_bbox(args.bbox)
    variables = None
    if args.properties:
        variables = [v.strip() for v in args.properties.split(",")]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    download_dir = os.path.join("/tmp", "stac_tiles")
    os.makedirs(download_dir, exist_ok=True)

    # ---- Step 1: Search STAC for items ----
    print("=" * 60)
    print("STEP 1: Searching STAC catalog")
    print("=" * 60)

    items = search_stac_items(
        stac_url=args.stac_url,
        collection=args.collection,
        bbox=bbox,
        temporal_extent=args.temporal_extent,
    )

    if not items:
        print("No items found matching the search criteria.", file=sys.stderr)
        sys.exit(1)

    # ---- Step 2: Download tiles ----
    print()
    print("=" * 60)
    print("STEP 2: Downloading tiles")
    print("=" * 60)

    downloaded_files = []
    progress = _ProgressTracker(len(items), desc="Downloading tiles")

    for item in items:
        url = get_download_url(item)
        if url is None:
            print(f"  WARNING: No download URL for item {item.get('id', '?')}")
            progress.update("SKIPPED")
            continue

        filepath = download_file(url, download_dir)
        if filepath:
            downloaded_files.append(filepath)
            progress.update(os.path.basename(filepath))
        else:
            progress.update("FAILED")

    progress.close()

    if not downloaded_files:
        print("ERROR: No files were downloaded successfully.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully downloaded {len(downloaded_files)} files.")

    # ---- Step 3: Merge onto regular grid ----
    print()
    print("=" * 60)
    print("STEP 3: Regridding onto regular lat/lon grid")
    print("=" * 60)

    output_filename = "merged_sst.nc"
    output_path = os.path.join(output_dir, output_filename)

    # Use the user-provided bbox as the target grid extent
    grid_bbox = tuple(bbox)

    merge_tiles_to_regular_grid(
        file_list=downloaded_files,
        output_path=output_path,
        resolution=args.resolution,
        bbox=grid_bbox,
        variables=variables,
        resample_method=args.resample_method,
        min_quality_level=args.min_quality_level,
    )

    # ---- Cleanup ----
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
