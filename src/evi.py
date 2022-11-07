import json
import io
import os
from pathlib import Path
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
from xarray import DataArray
import xrspatial.multispectral

from constants import STORAGE_AOI_PREFIX
from utils import scale_to_int16, write_to_blob_storage
from landsat_utils import (
    fix_bad_epsgs,
    get_bbox,
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
            .rio.write_crs(item_xr.rio.crs)
            .reset_coords(drop=True)
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


def mosaic_tiles(
    prefix: str,
    client: Client,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
) -> None:
    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )
    blobs = [
        f"/vsiaz/{container_name}/{blob.name}"
        for blob in container_client.list_blobs()
        if blob.name.startswith(prefix)
    ]

    with rasterio.open(STORAGE_AOI_PREFIX / "aoi.tif") as t:
        bounds = list(t.bounds)

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    mosaic_file = f"data/{local_prefix}.tif"

    rx.open_rasterio(vrt_file, chunks=True).rio.to_raster(
        mosaic_file, compress="LZW", predictor=2, lock=Lock("rio", client=client)
    )

    with rasterio.open(mosaic_file, "r+") as dst:
        dst.scales = OUTPUT_SCALE_FACTOR


if __name__ == "__main__":
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(200)
    function_name = "evi"
    function = globals()[function_name]
    year = 2021
    prefix = f"{function_name}/{year}/{function_name}_{year}"
    with cluster.get_client() as client:
        print(client.dashboard_link)
        process_by_scene(function, year, prefix)

    with Client() as local_client:
        print(local_client.dashboard_link)
        mosaic_tiles(prefix=prefix, client=client)
