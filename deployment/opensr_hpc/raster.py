from __future__ import annotations

import os
import re
from math import floor
from pathlib import Path

import numpy as np


def ensure_proj_env() -> None:
    if os.environ.get("PROJ_LIB"):
        return

    from pyproj import datadir

    proj_dir = datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir


def guess_utm_epsg(lat: float, lon: float) -> int:
    lat = max(min(lat, 84.0), -80.0)
    zone = int(floor((lon + 180.0) / 6.0)) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def parse_epsg(text: str, lat: float, lon: float) -> int:
    match = re.search(r"(\d{4,5})", text)
    if match:
        return int(match.group(1))
    return guess_utm_epsg(lat, lon)


def scale_to_uint16(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == "f" and np.nanmax(data) <= 1.1:
        return np.clip(data * 10000.0, 0, 10000).astype("uint16")
    return data.astype("uint16")


def compute_centroid_lat_lon(tif_path: Path) -> tuple[float, float]:
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Dataset at {tif_path} lacks a CRS")
        center_row = src.height / 2.0
        center_col = src.width / 2.0
        x, y = src.transform * (center_col, center_row)
        transformed = warp_transform(src.crs, "EPSG:4326", [x], [y])
        lon = transformed[0]
        lat = transformed[1]
        return float(lat[0]), float(lon[0])


def compress_geotiff(src_path: Path, dest_path: Path) -> Path:
    from rasterio.shutil import copy as rio_copy

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    rio_copy(
        src_path,
        dest_path,
        driver="GTiff",
        COMPRESS="ZSTD",
        ZSTD_LEVEL=22,
        PREDICTOR=2,
        TILED=True,
        BLOCKXSIZE=512,
        BLOCKYSIZE=512,
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
    )
    return dest_path
