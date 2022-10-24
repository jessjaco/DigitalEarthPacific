import json
import io
import os
from pathlib import Path
from time import time
from typing import Callable, Dict, List

import azure.storage.blob
from distributed.utils import ensure_memoryview
from dask_gateway import Gateway, GatewayCluster
from dask.distributed import Client, Lock
import geopandas as gpd
import numpy as np
from osgeo import gdal
import pystac_client
from pystac import ItemCollection
import planetary_computer
import rasterio
import rioxarray as rx
from stackstac import stack
from xarray import DataArray
import xrspatial.multispectral

from utils import write_to_blob_storage

CHUNKSIZE = 4096
OUTPUT_VALUE_MULTIPLIER = 1000
OUTPUT_SCALE_FACTOR = [1. / OUTPUT_VALUE_MULTIPLIER]
OUTPUT_NODATA = -32767


def mask_landsat_clouds(xr: DataArray) -> DataArray:
    # dilated cloud, cirrus, cloud, cloud shadow
    mask_bitfields = [1, 2, 3, 4]
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    qa = xr.sel(band="qa_pixel").astype("uint16")
    bad = qa & bitmask  # just look at those 4 bits
    return xr.where(bad == 0)


def item_collection_for_pathrow(
    path: int, row: int, search_args: Dict
) -> ItemCollection:

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog.search(
        **search_args,
        query=[
            f"landsat:wrs_path={path:03d}",
            f"landsat:wrs_row={row:03d}",
        ],
    ).item_collection()


def fix_bad_epsgs(item_collection: ItemCollection) -> None:
    """Repairs some band epsg codes in stac items loaded from the Planetary
    Computer stac catalog"""
    # ** modifies in place **
    # See https://github.com/microsoft/PlanetaryComputer/discussions/113
    # Will get fixed at some point and we can remove this
    for item in item_collection:
        epsg = str(item.properties["proj:epsg"])
        item.properties["proj:epsg"] = int(f"{epsg[0:3]}{int(epsg[3:]):02d}")


def get_bbox(gpdf: gpd.GeoDataFrame) -> List[float]:
    bbox = gpdf.to_crs("EPSG:4326").bounds.values[0]
    # Or the opposite!
    bbox_crosses_antimeridian = bbox[0] < 0 and bbox[2] > 0
    if bbox_crosses_antimeridian:
        bbox[2] = bbox[0]
        bbox[0] = -179.999999999999
    return bbox


def ndvi(xr: DataArray) -> DataArray:
    """Receives a DataArray with bands "nir08" and "red" (i.e. landsat data
    as loaded by stackstac.stack and returns a DataArray with a single band
    representing NDVI."""
    nir = xr.sel(band="nir08")
    red = xr.sel(band="red")
    return np.divide(np.subtract(nir, red), np.add(nir, red))


def evi(xr: DataArray) -> DataArray:
    nir = xr.sel(band="nir08")
    red = xr.sel(band="red")
    blue = xr.sel(band="blue")
    return xrspatial.multispectral.evi(nir, red, blue)


def process_by_scene(function: Callable, year: int, output_prefix: str) -> None:

    pathrows = gpd.read_file("data/pathrows_in_aoi.gpkg")
    aoi_by_pathrow = gpd.read_file("data/aoi_split_by_landsat_pathrow.gpkg")

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
            print(f"{path:03d}-{row:03d} ** NO ITEMS ** ")
            continue

        fix_bad_epsgs(item_collection)
        item_xr = (
            stack(
                item_collection,
                epsg=8859,
                chunksize=CHUNKSIZE,
                resolution=30,
                rescale=True,
            )
            .rio.write_crs("EPSG:8859")
            .rio.clip(these_areas.to_crs("EPSG:8859").geometry, all_touched=True)
        )

        item_xr = mask_landsat_clouds(item_xr)

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

        scaled_results = (
            np.multiply(results, OUTPUT_VALUE_MULTIPLIER)
            .where(results.notnull(), OUTPUT_NODATA)
            .astype("int16")
            .rio.write_nodata(OUTPUT_NODATA)
            .rio.write_crs(item_xr.rio.crs)
        )

        try:
            write_to_blob_storage(
                scaled_results,
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
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
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

    with rasterio.open("data/aoi.tif") as t:
        bounds = list(t.bounds)

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    mosaic_file = f"data/{local_prefix}.tif"

    rx.open_rasterio(vrt_file, chunks=True).rio.to_raster(
        mosaic_file, compress="LZW", predictor=2, lock=Lock("rio", client=client)
    )

    with rasterio.open(mosaic_file, "r+") as dst:
        dst.scales = [OUTPUT_SCALE_FACTOR]


if __name__ == "__main__":
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(200)
    function_name = 'evi'
    function = globals()[function_name]
    year = 2021
    prefix = f"{function_name}/{year}/{function_name}_{year}"
#    with cluster.get_client() as client:
#        print(client.dashboard_link)
#        process_by_scene(function, year, prefix)

    with Client() as local_client:
        print(local_client.dashboard_link)
        mosaic_tiles(prefix=prefix, client=local_client)
