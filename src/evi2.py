from pathlib import Path
from typing import Callable

from xarray import DataArray
import xrspatial.multispectral

from Processor import run_processor
from utils import scale_and_offset

OUTPUT_VALUE_MULTIPLIER = 1000
OUTPUT_NODATA = -32767
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


def evi_for_year(item_xr: DataArray) -> None:
    item_xr = scale_and_offset(item_xr, scale = 0.0000275, offset = -0.2)
    annual_medians = (
        item_xr.resample(time="Y").median("time").squeeze("time", drop=True)
    )

    return evi(annual_medians).reset_coords(drop=True).rio.write_crs(item_xr.rio.crs)



if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("year")
    parser.add_argument("-r", "--run_scenes", action="store_true")
    parser.add_argument("-m", "--mosaic", action="store_true")
    parser.add_argument("-t", "--tile", action="store_true")
    parser.add_argument("--remake_mosaic_for_tiles", action="store_true")

    args = parser.parse_args()

    color_ramp_file = Path(__file__).with_name("evi_color_ramp.txt").as_posix()
    os.makedirs("data", exist_ok=True)
    run_processor(
        year=args.year,
        scene_processor=evi_for_year,
        color_ramp_file=color_ramp_file,
        run_scenes=args.run_scenes,
        mosaic=args.mosaic,
        tile=args.tile,
        remake_mosaic_for_tiles=args.remake_mosaic_for_tiles,
    )
