from functools import partial
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
    container_client = azure.storage.blob.ContainerClient( f"https://{storage_account}.blob.core.windows.net", container_name=container_name, credential=credential,)

    with io.BytesIO() as buffer:
        xr.rio.to_raster(buffer, **write_args)
        buffer.seek(0)
#        with rasterio.open(buffer, "r+") as dst:
#            dst.scales = output_scale
        blob_client = container_client.get_blob_client(path)
        blob_client.upload_blob(buffer, overwrite=True)

