from pathlib import Path

from xarray import DataArray
import xrspatial.multispectral

from Processor import run_processor
from utils import scale_and_offset


def evi(xr: DataArray) -> DataArray:
    """Receives a DataArray with bands "nir08", "red" & "blue" (i.e. landsat data
    as loaded by stackstac.stack and returns a DataArray with a single band
    representing EVI."""
    nir = xr.sel(band="nir08")
    red = xr.sel(band="red")
    blue = xr.sel(band="blue")
    return xrspatial.multispectral.evi(nir, red, blue)


def evi_for_year(landsat_xr: DataArray) -> None:
    annual_medians = landsat_xr.median("time").squeeze("time", drop=True)

    return evi(annual_medians).reset_coords(drop=True).rio.write_crs(landsat_xr.rio.crs)


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
        dataset_id="evi",
        color_ramp_file=color_ramp_file,
        run_scenes=args.run_scenes,
        mosaic=args.mosaic,
        tile=args.tile,
        remake_mosaic_for_tiles=args.remake_mosaic_for_tiles,
    )
