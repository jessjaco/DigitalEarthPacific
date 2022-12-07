import json
import io
import os
from pathlib import Path
import re
from time import time
from typing import Callable

import azure.storage.blob
from dask_gateway import GatewayCluster
from dask.distributed import Client, Lock
import geopandas as gpd
from osgeo import gdal
import rasterio
import rioxarray as rx
from stackstac import stack
import typer
from xarray import DataArray
import xrspatial.multispectral

from constants import STORAGE_AOI_PREFIX
from utils import (
    create_tiles,
    get_bbox,
    mosaic_scenes,
    raster_bounds,
    scale_to_int16,
    write_to_blob_storage,
)
from landsat_utils import (
    fix_bad_epsgs,
    item_collection_for_pathrow,
    mask_clouds,
)

CHUNKSIZE = 4096
OUTPUT_VALUE_MULTIPLIER = 1000
OUTPUT_SCALE_FACTOR = [1.0 / OUTPUT_VALUE_MULTIPLIER]
OUTPUT_NODATA = -32767


def evi(xr: DataArray) -> DataArray:
    """Receives a DataArray with bands "nir08", "red" & "blue" (i.e. landsat data
    as loaded by stackstac.stack and returns a DataArray with a single band
    representing EVI."""
    nir = xr.sel(band="nir08")
    red = xr.sel(band="red")
    blue = xr.sel(band="blue")
    return xrspatial.multispectral.evi(nir, red, blue)


def process_by_scene(function: Callable, year: int, output_prefix: str) -> None:

    pathrows = gpd.read_file(STORAGE_AOI_PREFIX / "pathrows_in_aoi.gpkg")
    aoi_by_pathrow = gpd.read_file(
        STORAGE_AOI_PREFIX / "aoi_split_by_landsat_pathrow.gpkg"
    )

    for i, row in pathrows.iterrows():
        last_time = time()
        path = row["PATH"]
        row = row["ROW"]
        these_areas = aoi_by_pathrow[
            (aoi_by_pathrow["PATH"] == path) & (aoi_by_pathrow["ROW"] == row)
        ]

        item_collection = item_collection_for_pathrow(
            path,
            row,
            dict(
                collections=["landsat-c2-l2"],
                datetime=str(year),
                bbox=get_bbox(these_areas),
            ),
        )

        if len(item_collection) == 0:
            print(f"{path:03d}-{row:03d} | ** NO ITEMS **")
            continue

        fix_bad_epsgs(item_collection)
        item_xr = (
            stack(
                item_collection,
                epsg=8859,
                chunksize=CHUNKSIZE,
                resolution=30,
            )
            .rio.write_crs("EPSG:8859")
            .rio.clip(these_areas.to_crs("EPSG:8859").geometry, all_touched=True)
        )

        item_xr = mask_clouds(item_xr)

        # This needs to be done manually (rescale arg of stack doesn't
        # work because these COGs don't have scale and offset set correctly.
        # AND after the cloud mask / applied only to value layers / etc.
        scale = 0.0000275
        offset = -0.2
        item_xr = item_xr * scale + offset
        annual_medians = (
            item_xr.resample(time="Y").median("time").squeeze("time", drop=True)
        )

        results = (
            function(annual_medians)
            .reset_coords(drop=True)
            .rio.write_crs(item_xr.rio.crs)
        )

        results = scale_to_int16(results, OUTPUT_VALUE_MULTIPLIER, OUTPUT_NODATA)

        try:
            write_to_blob_storage(
                results,
                f"{output_prefix}_{path}_{row}.tif",
                dict(driver="COG", compress="LZW", predictor=2),
            )
        except Exception as e:
            print(e)
        print(
            f"{path:03d}-{row:03d} | {(i+1):03d}/{len(pathrows.index)} | {round(time() - last_time)}s"
        )


def _get_prefix(year: int) -> str:
    return f"evi/{year}/evi_{year}"


def process_scenes(year: int) -> None:
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(400)
    with cluster.get_client() as client:
        process_by_scene(evi, year, _get_prefix(year))


def mosaic_scenes(year: int) -> None:
    bounds = raster_bounds(STORAGE_AOI_PREFIX / "aoi.tif")
    with Client() as local_client:
        mosaic_files(
            prefix=_get_prefix(year),
            bounds=bounds,
            client=local_client,
            scale_factor=1.0 / OUTPUT_VALUE_MULTIPLIER,
        )


def make_tiles(year: int, remake_mosaic: bool = True) -> None:
    bounds = raster_bounds(STORAGE_AOI_PREFIX / "aoi.tif")
    create_tiles(
        Path(__file__).with_name("evi_color_ramp.txt").as_posix(),
        _get_prefix(year),
        bounds,
        remake_mosaic=remake_mosaic,
    )


def main(
    year: int,
    run_scenes: bool = False,
    mosaic: bool = False,
    tile: bool = False,
    remake_mosaic_for_tiles: bool = True,
):
    if run_scenes:
        process_scenes(year)

    if mosaic:
        mosaic_scenes(year)

    if tile:
        make_tiles(year, remake_mosaic_for_tiles)


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    typer.run(main)
