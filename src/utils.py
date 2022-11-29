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


def scale_to_int16(
    da: DataArray, output_multiplier: int, output_nodata: int
) -> DataArray:
    return (
        np.multiply(da, output_multiplier)
        .where(da.notnull(), output_nodata)
        .astype("int16")
        .rio.write_nodata(output_nodata)
        .rio.write_crs(da.rio.crs)
    )


def raster_bounds(raster_path: Path) -> List:
    with rasterio.open(raster_path) as t:
        return list(t.bounds)


from dask.distributed import Client, Lock
from osgeo import gdal


def build_vrt(
    prefix: str,
    bounds: List,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
) -> Path:
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
    return Path(vrt_file)


import os
import osgeo_utils.gdal2tiles
from tqdm import tqdm


def create_tiles(
    color_file: str,
    prefix: str,
    bounds: List,
    remake_mosaic: bool = True,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
):
    if remake_mosaic:
        with Client() as local_client:
            mosaic_files(
                prefix=prefix,
                bounds=bounds,
                client=local_client,
                storage_account=storage_account,
                credential=credential,
                container_name=container_name,
                scale_factor=1.0 / 1000,
                overwrite=remake_mosaic,
            )
    dst_vrt_file = f"data/{Path(prefix).stem}_rgb.vrt"
    gdal.DEMProcessing(
        dst_vrt_file,
        str(_mosaic_file(prefix)),
        "color-relief",
        colorFilename=color_file,
        addAlpha=True,
    )
    dst_name = f"data/tiles/{prefix}"
    os.makedirs(dst_name, exist_ok=True)
    max_zoom = 11
    # First arg is just a dummy so the second arg is not removed (see gdal2tiles code)
    # I'm using 512 x 512 tiles so there's fewer files to copy over
    osgeo_utils.gdal2tiles.main(
        [
            "gdal2tiles.py",
            "--tilesize=512",
            "--processes=4",
            f"--zoom=0-{max_zoom}",
            "-x",
            dst_vrt_file,
            dst_name,
        ]
    )

    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    for local_path in tqdm(Path(dst_name).rglob("*")):
        if local_path.is_file():
            with open(local_path, "rb") as src:
                remote_path = Path("tiles") / "/".join(local_path.parts[4:])
                blob_client = container_client.get_blob_client(str(remote_path))
                blob_client.upload_blob(src, overwrite=True)
                local_path.unlink()


def _local_prefix(prefix: str) -> str:
    return Path(prefix).stem


def _mosaic_file(prefix: str) -> str:
    return f"data/{_local_prefix(prefix)}.tif"


def mosaic_files(
    prefix: str,
    bounds: List,
    client: Client,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
    scale_factor: float = None,
    overwrite: bool = True,
) -> None:

    mosaic_file = _mosaic_file(prefix)
    if not Path(mosaic_file).is_file() or overwrite:
        vrt_file = build_vrt(
            prefix, bounds, storage_account, credential, container_name
        )
        vrt_file = f"data/{_local_prefix(prefix)}.vrt"
        rioxarray.open_rasterio(vrt_file, chunks=True).rio.to_raster(
            _mosaic_file(prefix),
            compress="LZW",
            predictor=2,
            lock=Lock("rio", client=client),
        )

        if scale_factor is not None:
            with rasterio.open(mosaic_file, "r+") as dst:
                dst.scales = (scale_factor,)
