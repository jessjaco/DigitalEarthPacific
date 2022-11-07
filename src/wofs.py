from pathlib import Path
from time import time

from dask_gateway import GatewayCluster
import geopandas as gpd
from stackstac import stack
from xarray import DataArray
import xarray as xr

from constants import STORAGE_AOI_PREFIX
from landsat_utils import (
    get_bbox,
    fix_bad_epsgs,
    mask_clouds,
    item_collection_for_pathrow,
)
from utils import scale_to_int16, write_to_blob_storage

CHUNKSIZE = 4096
OUTPUT_VALUE_MULTIPLIER = 10000
OUTPUT_NODATA = -32767

# Delete to use copies in blob storage; these are stored locally on the PC
# so they run a little faster
STORAGE_AOI_PREFIX = Path('data')


def normalized_ratio(band1: DataArray, band2: DataArray) -> DataArray:
    return (band1 - band2) / (band1 + band2)


def wofs(tm: DataArray) -> DataArray:
    tm = tm.to_dataset('band')
    tm['ndi52'] = normalized_ratio(tm.swir16, tm.green)
    tm['ndi43'] = normalized_ratio(tm.nir08, tm.red)
    tm['ndi72'] = normalized_ratio(tm.swir22, tm.green)
    l0 = tm.ndi52 <= -0.01
    l3 = l0 & (tm.blue <= 2083.5) & (tm.swir22 <= 323.5)
    w1 = l0 & l3 & (tm.ndi43 <= 0.61)
    l9 = ~l3 & (tm.blue < 1400.5)
    w7 = ~l9 & (tm.ndi43 <= -0.01)
    l12 = l9 & (tm.ndi72 <= -0.23)
    w2 = l12 & (tm.ndi43 <= 0.22)
    w3 = ~w2 & (tm.blue <= 473.0)
    w4 = ~l12 & (tm.blue <= 379.0)
    l2 = ~l0 & (tm.ndi52 <= 0.23)
    w5 = l2 & (tm.blue <= 334.5) & (tm.ndi43 <= 0.54) & (tm.ndi52 <= -0.12)
    l21 = ~w5 & (tm.red <= 364.5)
    w6 = l21 & (tm.blue <= 129.5)
    w8 = ~l21 & (tm.blue <= 300.5)
    w9 = (
        ~l2 & (tm.ndi52 <= 0.32) & 
        (tm.blue <= 249.5) & (tm.ndi43 <= 0.45) & (tm.red <= 364.5) & (tm.blue <= 129.5)
    )

    water = w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9
    return water.where(tm.red.notnull(), float("nan"))


def process_wofs(year: int, output_prefix: str) -> None:
    pathrows = gpd.read_file(STORAGE_AOI_PREFIX / "pathrows_in_aoi.gpkg")
    aoi_by_pathrow = gpd.read_file(
        STORAGE_AOI_PREFIX / "aoi_split_by_landsat_pathrow.gpkg"
    )
    for i, row in pathrows.iterrows():
        last_time = time()
        path = row["PATH"]
        row = row["ROW"]
        path = 93
        row = 62
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

        results = wofs(item_xr).mean("time").reset_coords(drop=True).rio.write_crs(item_xr.rio.crs)

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


if __name__ == "__main__":
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(200)
    year = 2021
    prefix = f"wofs/{year}/wofs_{year}"
    with cluster.get_client() as client:
        print(client.dashboard_link)
        process_wofs(year, prefix)
