from functools import partial
from pathlib import Path
from typing import Callable, Optional, List

import xarray as xr
import geopandas as gpd
import geocube
from geocube.api.core import make_geocube
import rasterio


def make_geocube_dask(
    df: gpd.GeoDataFrame, measurements: List[str], like: xr.DataArray, **kwargs
):
    def rasterize_block(block):
        return (
            make_geocube(df, measurements=measurements, like=block, **kwargs)
            .to_array(measurements[0])
            .assign_coords(block.coords)
        )

    like = like.rename(dict(zip(["band"], measurements)))
    return like.map_blocks(rasterize_block, template=like)


import io
import os
from pathlib import Path
from typing import Dict, Union

import azure.storage.blob
from xarray import DataArray

def write_to_blob_storage(
    xr: DataArray,
    path: Union[str, Path],
    write_args: Dict,
    #    output_scale: List = [1.0],
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
) -> None:
    container_client = azure.storage.blob.ContainerClient(
            f"https://{storage_account}.blob.core.windows.net", 
            container_name=container_name, 
            credential=credential,
    )

    with io.BytesIO() as buffer:
        xr.rio.to_raster(buffer, **write_args)
        buffer.seek(0)
        blob_client = container_client.get_blob_client(path)
        blob_client.upload_blob(buffer, overwrite=True)


import numpy as np
import rioxarray
def scale_to_int16(da: DataArray, output_multiplier: int, output_nodata: int) -> DataArray:
    return (
        np.multiply(da, output_multiplier)
        .where(da.notnull(), output_nodata)
        .astype("int16")
        .rio.write_nodata(output_nodata)
        .rio.write_crs(da.rio.crs)
    )


def bounds(raster_path: Path) -> List:
    with rasterio.open(raster_path) as t:
        return list(t.bounds)

from dask.distributed import Client, Lock
from osgeo import gdal

def mosaic_tiles(
    prefix: str,
    bounds: List,
    client: Client,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
    scale_factor: float = None,
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

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    mosaic_file = f"data/{local_prefix}.tif"

    rioxarray.open_rasterio(vrt_file, chunks=True).rio.to_raster(
        mosaic_file, compress="LZW", predictor=2, lock=Lock("rio", client=client)
    )

    if scale_factor is not None:
        with rasterio.open(mosaic_file, "r+") as dst:
            dst.scales = scale_factor
