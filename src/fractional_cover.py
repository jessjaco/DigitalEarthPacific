from pathlib import Path
from time import time

from dask_gateway import GatewayCluster
from dask.distributed import Client
import geopandas as gpd
import rioxarray
from stackstac import stack
from xarray import DataArray
import xarray as xr

from constants import STORAGE_AOI_PREFIX
from fc.fc.fractional_cover import fractional_cover
from landsat_utils import (
    get_bbox,
    fix_bad_epsgs,
    mask_clouds,
    item_collection_for_pathrow,
)
from utils import scale_to_int16, write_to_blob_storage, raster_bounds, mosaic_files

CHUNKSIZE = 2048
OUTPUT_VALUE_MULTIPLIER = 10000
OUTPUT_NODATA = -32767

# Delete to use copies in blob storage; these are stored locally on the PC
# so they run a little faster
STORAGE_AOI_PREFIX = Path("data")


def fc_by_scene(year: int, output_prefix: str) -> None:
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
        # Now scale to what wofs tree expects
        l1_scale = 0.0001
        l1_rescale = 1.0 / l1_scale
        item_xr *= l1_rescale

        input_ds = (
            item_xr.to_dataset("band")[['red', 'blue', 'green', 'nir08', 'swir16', 'swir22']]
            .rename(dict(swir16='swir1', swir22='swir2', nir08='nir'))
            .isel(time=0)
            .where(lambda x: x > 0)
        )

        for data_var in input_ds.data_vars.keys():
            input_ds[data_var].attrs["nodata"] = float("nan")
        # results = input_ds.map_blocks(fractional_cover, template=input_ds)
        results = fractional_cover(input_ds).rio.write_crs('EPSG:8859')
        
        # This is for gdal
        for data_var in results.data_vars.keys():
            results[data_var] = results[data_var].astype('uint8').rio.write_nodata(255)
        del results.attrs['grid_mapping']
        #.astype('uint8').rio.write_nodata(255).rio.write_crs('EPSG:8859')
        

        #        results = (
        #            fc.fractional_cover(inpu
        #            .mean("time")
        #            .reset_coords(drop=True)
        #            .rio.write_crs(item_xr.rio.crs)
        #        )

#        results = scale_to_int16(results, OUTPUT_VALUE_MULTIPLIER, OUTPUT_NODATA)

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


if __name__ == "__main__":
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(200)
    year = 2021
    prefix = f"fc/{year}/fc_{year}"
# Code works with local client, but not on gateway
#    with cluster.get_client() as client:
    with Client() as client:
        print(client.dashboard_link)
        fc_by_scene(year, prefix)

    bounds = raster_bounds(STORAGE_AOI_PREFIX / "aoi.tif")
    with Client() as local_client:
        print(local_client.dashboard_link)
        mosaic_files(
            prefix=prefix,
            bounds=bounds,
            client=local_client,
            scale_factor=1.0 / OUTPUT_VALUE_MULTIPLIER,
        )
