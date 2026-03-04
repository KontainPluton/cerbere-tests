"""
stac_swath_processor.py - Download STAC L2P swath tiles and merge them
onto a regular lat/lon grid.

Supports --keep_time to preserve the time axis: each unique timestamp
from the source tiles becomes a time step in the output NetCDF.
Tiles sharing the same timestamp are fused via quality-aware mosaicking.

Dependencies: requests, numpy, xarray, pyresample, netCDF4
"""

import sys
import os
import json
import argparse
import time
import shutil
from collections import OrderedDict
from urllib.parse import urlparse

import requests
import numpy as np
import xarray as xr
import pandas as pd

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

_PROGRESS_FILE = None


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _write_progress_file(data):
    if _PROGRESS_FILE is None:
        return
    try:
        tmp = _PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _PROGRESS_FILE)
    except Exception:
        pass


class _ProgressTracker:
    def __init__(self, total, desc="Processing", step_name="processing"):
        self.total = total
        self.desc = desc
        self.step_name = step_name
        self.current = 0
        self.start_time = time.time()
        if HAS_TQDM:
            self.pbar = tqdm(total=total, desc=desc, unit="file")
        else:
            self.pbar = None
            print(f"{desc}: 0/{total} files")
        self._write_state("running", "")

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
        self._write_state("running", filename)

    def close(self):
        elapsed = time.time() - self.start_time
        if self.pbar is not None:
            self.pbar.close()
        print(f"{self.desc}: done in {_fmt_time(elapsed)}")
        self._write_state("completed", "")

    def _write_state(self, status, current_file):
        elapsed = time.time() - self.start_time
        if 0 < self.current < self.total:
            remaining_s = (elapsed / self.current) * (self.total - self.current)
        else:
            remaining_s = 0.0
        pct = (100.0 * self.current / self.total) if self.total > 0 else 0.0
        _write_progress_file({
            "step": self.step_name, "status": status,
            "progress_percent": round(pct, 1),
            "current": self.current, "total": self.total,
            "current_file": current_file,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_human": _fmt_time(elapsed),
            "remaining_seconds": round(remaining_s, 1),
            "remaining_human": _fmt_time(remaining_s),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


# =========================================================================
# STAC search and download
# =========================================================================

def search_stac_items(stac_url, collection, bbox, temporal_extent):
    search_url = stac_url.rstrip("/") + "/search"
    payload = {"collections": [collection], "bbox": bbox,
               "datetime": temporal_extent, "limit": 100}
    all_items = []
    page = 1

    print(f"Searching STAC API: {search_url}")
    print(f"  Collection : {collection}")
    print(f"  BBox       : {bbox}")
    print(f"  Datetime   : {temporal_extent}")

    _write_progress_file({"step": "search", "status": "running",
                          "progress_percent": 0, "message": "Searching STAC catalog...",
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    while True:
        print(f"  Fetching page {page}...")
        try:
            resp = requests.post(search_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError:
            if page == 1:
                print("  POST /search failed, falling back to OGC API")
                return _search_stac_items_via_ogcapi(stac_url, collection, bbox, temporal_extent)
            raise
        except Exception as e:
            print(f"ERROR: STAC search failed: {e}", file=sys.stderr)
            sys.exit(1)

        features = data.get("features", [])
        all_items.extend(features)
        print(f"  Got {len(features)} items (total: {len(all_items)})")

        next_link = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_link = link
                break
        if next_link is None:
            break

        next_href = next_link.get("href")
        next_method = next_link.get("method", "GET").upper()
        next_body = next_link.get("body")

        if next_method == "POST" and next_body:
            payload = next_body
        elif next_method == "POST":
            token = (next_link.get("body") or {}).get("token") or next_link.get("token")
            if token:
                payload["token"] = token
            else:
                break
        else:
            try:
                resp = requests.get(next_href, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                all_items.extend(data.get("features", []))
                if not any(l.get("rel") == "next" for l in data.get("links", [])):
                    break
                continue
            except Exception:
                break
        page += 1

    print(f"Total items found: {len(all_items)}")
    _write_progress_file({"step": "search", "status": "completed",
                          "progress_percent": 100, "message": f"Found {len(all_items)} items",
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return all_items


def _search_stac_items_via_ogcapi(stac_url, collection, bbox, temporal_extent):
    items_url = stac_url.rstrip("/") + f"/collections/{collection}/items"
    params = {"bbox": ",".join(str(b) for b in bbox),
              "datetime": temporal_extent, "limit": 100}
    all_items = []
    page = 1
    while items_url:
        print(f"  Fetching page {page} (OGC API)...")
        try:
            resp = requests.get(items_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        all_items.extend(data.get("features", []))
        items_url = None
        params = {}
        for link in data.get("links", []):
            if link.get("rel") == "next":
                items_url = link.get("href")
                break
        page += 1
    print(f"Total items found: {len(all_items)}")
    return all_items


def get_download_url(item):
    assets = item.get("assets", {})
    for key, meta in assets.items():
        if "data" in meta.get("roles", []):
            alt = meta.get("alternate", {})
            if "HTTPS_IFREMER" in alt:
                href = alt["HTTPS_IFREMER"].get("href")
                if href:
                    return href
            if meta.get("href"):
                return meta["href"]
    for key, meta in assets.items():
        if meta.get("href"):
            return meta["href"]
    return None


def download_file(url, output_dir):
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
        print(f"ERROR: download failed {url}: {e}", file=sys.stderr)
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
# Time extraction from NetCDF
# =========================================================================

def _read_tile_timestamp(filepath):
    """
    Read the timestamp from a NetCDF tile.
    Tries the 'time' variable first, then falls back to global attributes
    like 'start_time' or 'time_coverage_start'.
    Returns a numpy datetime64 or None.
    """
    with xr.open_dataset(filepath) as ds:
        # Try 'time' variable
        if "time" in ds.coords or "time" in ds.data_vars:
            t = ds["time"].values
            if t.ndim == 0:
                return pd.Timestamp(t.item()).to_datetime64()
            elif len(t) > 0:
                return pd.Timestamp(t[0].item() if hasattr(t[0], 'item') else t[0]).to_datetime64()

        # Fallback to global attributes
        for attr_name in ["start_time", "time_coverage_start", "start_date"]:
            val = ds.attrs.get(attr_name)
            if val:
                try:
                    return pd.Timestamp(val).to_datetime64()
                except Exception:
                    pass

    return None


def _group_files_by_timestamp(file_list):
    """
    Read each file's timestamp and group files sharing the same timestamp.
    Returns an OrderedDict: {np.datetime64 -> [file_path, ...]} sorted by time.
    """
    ts_map = {}
    for f in file_list:
        ts = _read_tile_timestamp(f)
        if ts is None:
            print(f"  WARNING: no timestamp found in {os.path.basename(f)}, using epoch")
            ts = np.datetime64("1970-01-01T00:00:00", "ns")
        ts_key = str(ts)
        if ts_key not in ts_map:
            ts_map[ts_key] = {"time": ts, "files": []}
        ts_map[ts_key]["files"].append(f)

    # Sort by time
    sorted_groups = sorted(ts_map.values(), key=lambda g: g["time"])
    result = OrderedDict()
    for g in sorted_groups:
        result[g["time"]] = g["files"]

    return result


# =========================================================================
# Core regridding of a single time step (one or more tiles)
# =========================================================================

def _regrid_tile_group(
        file_list, target_def, ny, nx, variables, var_attrs,
        crosses_antimeridian, radius_of_influence, resample_method,
        sigma, min_quality_level,
):
    """
    Regrid one group of tiles (same timestamp) onto the target grid.
    If multiple tiles, applies quality-aware mosaicking.

    Returns
    -------
    dict: {var_name -> np.ndarray(ny, nx)}, plus 'mosaic_quality_level'
    """
    from pyresample import geometry, kd_tree

    merged_data = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in variables}
    best_quality = np.full((ny, nx), -1.0, dtype=np.float32)

    for f in file_list:
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

    merged_data["mosaic_quality_level"] = np.where(best_quality < 0, np.nan, best_quality)
    return merged_data


# =========================================================================
# Main merge function
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
        keep_time=False,
):
    """
    Merge multiple L2P NetCDF swath tiles onto a regular lat/lon grid.

    If keep_time=True, the output has a 'time' dimension. Tiles are grouped
    by their NetCDF timestamp: same timestamp -> fused, different -> separate
    time steps.

    If keep_time=False (default), all tiles are fused into a single 2D grid.
    """
    from pyresample import geometry

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
    print(f"Keep time axis: {keep_time}")

    # -- Convert longitudes for output
    output_lons = target_lons.copy()
    sort_idx = None
    if crosses_antimeridian and output_lon_convention == "[-180, 180]":
        output_lons = _lon_to_180(output_lons)
        sort_idx = np.argsort(output_lons)
        output_lons = output_lons[sort_idx]

    # ==================================================================
    # CASE 1: keep_time=False (original behaviour, all fused into 2D)
    # ==================================================================
    if not keep_time:
        progress = _ProgressTracker(len(file_list), "Regridding tiles", "regridding")

        result = _regrid_tile_group(
            file_list, target_def, ny, nx, variables, var_attrs,
            crosses_antimeridian, radius_of_influence, resample_method,
            sigma, min_quality_level,
        )

        # Fake progress for all files
        for f in file_list:
            progress.update(os.path.basename(f))
        progress.close()

        # Apply lon reorder
        if sort_idx is not None:
            for key in result:
                result[key] = result[key][:, sort_idx]

        # Build dataset
        data_vars = {
            var: (["lat", "lon"], result[var], var_attrs.get(var, {}))
            for var in variables
        }
        data_vars["mosaic_quality_level"] = (
            ["lat", "lon"], result["mosaic_quality_level"],
            {"long_name": "Best quality_level used during mosaicking",
             "comment": f"Minimum accepted quality_level was {min_quality_level}"},
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

    # ==================================================================
    # CASE 2: keep_time=True (group by timestamp, one time step each)
    # ==================================================================
    else:
        print("Grouping tiles by timestamp...")
        groups = _group_files_by_timestamp(file_list)
        timestamps = list(groups.keys())
        n_times = len(timestamps)

        print(f"Found {n_times} distinct timestamp(s):")
        for ts, files in groups.items():
            n = len(files)
            label = "tile" if n == 1 else f"tiles -> will fuse"
            print(f"  {ts} : {n} {label}")

        # Allocate 3D arrays: (time, lat, lon)
        all_data = {v: np.full((n_times, ny, nx), np.nan, dtype=np.float32) for v in variables}
        all_quality = np.full((n_times, ny, nx), np.nan, dtype=np.float32)

        progress = _ProgressTracker(len(file_list), "Regridding tiles", "regridding")
        file_counter = 0

        for t_idx, (ts, tile_files) in enumerate(groups.items()):
            n_in_group = len(tile_files)
            if n_in_group > 1:
                print(f"  Timestamp {ts}: fusing {n_in_group} tiles")
            else:
                print(f"  Timestamp {ts}: regridding 1 tile")

            result = _regrid_tile_group(
                tile_files, target_def, ny, nx, variables, var_attrs,
                crosses_antimeridian, radius_of_influence, resample_method,
                sigma, min_quality_level,
            )

            # Apply lon reorder
            if sort_idx is not None:
                for key in result:
                    result[key] = result[key][:, sort_idx]

            for var in variables:
                all_data[var][t_idx, :, :] = result[var]
            all_quality[t_idx, :, :] = result["mosaic_quality_level"]

            for f in tile_files:
                progress.update(os.path.basename(f))

        progress.close()

        # Build 3D dataset with time dimension
        time_coords = np.array(timestamps, dtype="datetime64[ns]")

        data_vars = {
            var: (["time", "lat", "lon"], all_data[var], var_attrs.get(var, {}))
            for var in variables
        }
        data_vars["mosaic_quality_level"] = (
            ["time", "lat", "lon"], all_quality,
            {"long_name": "Best quality_level used during mosaicking",
             "comment": f"Minimum accepted quality_level was {min_quality_level}"},
        )

        merged_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": (["time"], time_coords, {"standard_name": "time"}),
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
                "time_steps": str(n_times),
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
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}: {bbox_str}")
    return parts


def main():
    global _PROGRESS_FILE

    parser = argparse.ArgumentParser(
        description="Download STAC L2P swath tiles and merge onto a regular grid."
    )
    parser.add_argument("--stac_url", required=True, help="Root STAC API URL")
    parser.add_argument("--collection", required=True, help="STAC Collection ID")
    parser.add_argument("--bbox", required=True, help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--temporal_extent", required=True, help="Temporal range: start/end ISO 8601")
    parser.add_argument("--properties", default=None, help="Comma-separated variable names (default: all)")
    parser.add_argument("--resolution", type=float, default=0.01, help="Grid resolution in degrees")
    parser.add_argument("--resample_method", default="nearest", choices=["nearest", "gauss", "custom"])
    parser.add_argument("--min_quality_level", type=int, default=0, help="Min quality_level 0-5")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--progress_file", default=None, help="JSON progress file path")
    parser.add_argument("--keep_time", action="store_true", default=False,
                        help="Preserve time axis in output NetCDF. "
                             "Each unique tile timestamp becomes a time step. "
                             "Tiles with identical timestamps are fused.")

    args = parser.parse_args()

    _PROGRESS_FILE = args.progress_file
    if _PROGRESS_FILE:
        print(f"Progress file: {_PROGRESS_FILE}")

    bbox = parse_bbox(args.bbox)
    variables = [v.strip() for v in args.properties.split(",")] if args.properties else None
    os.makedirs(args.output_dir, exist_ok=True)
    dl_dir = os.path.join("/tmp", "stac_tiles")
    os.makedirs(dl_dir, exist_ok=True)

    # Step 1: Search
    print("=" * 60 + "\nSTEP 1: Searching STAC catalog\n" + "=" * 60)
    items = search_stac_items(args.stac_url, args.collection, bbox, args.temporal_extent)
    if not items:
        print("No items found.", file=sys.stderr)
        _write_progress_file({"step": "search", "status": "failed", "message": "No items"})
        sys.exit(1)

    # Step 2: Download
    print("\n" + "=" * 60 + "\nSTEP 2: Downloading tiles\n" + "=" * 60)
    files = []
    prog = _ProgressTracker(len(items), "Downloading", "downloading")
    for item in items:
        url = get_download_url(item)
        if not url:
            prog.update("SKIPPED")
            continue
        fp = download_file(url, dl_dir)
        if fp:
            files.append(fp)
            prog.update(os.path.basename(fp))
        else:
            prog.update("FAILED")
    prog.close()

    if not files:
        print("ERROR: No files downloaded.", file=sys.stderr)
        sys.exit(1)
    print(f"\nDownloaded {len(files)} files.")

    # Step 3: Regrid
    print("\n" + "=" * 60 + "\nSTEP 3: Regridding\n" + "=" * 60)
    out = os.path.join(args.output_dir, "merged_sst.nc")
    merge_tiles_to_regular_grid(
        file_list=files, output_path=out, resolution=args.resolution,
        bbox=tuple(bbox), variables=variables,
        resample_method=args.resample_method,
        min_quality_level=args.min_quality_level,
        keep_time=args.keep_time,
    )

    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)
    print(f"Output: {out}")
    _write_progress_file({"step": "done", "status": "completed", "progress_percent": 100,
                          "output_file": out, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()