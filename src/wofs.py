from pathlib import Path

from xarray import DataArray

from Processor import run_processor
from utils import scale_and_offset


def normalized_ratio(band1: DataArray, band2: DataArray) -> DataArray:
    return (band1 - band2) / (band1 + band2)


def wofs(tm_da: DataArray) -> DataArray:
    # lX indicates a left path from node X
    # rX indicates a right
    # dX is just the logic for _that_ node
    tm = tm_da.to_dataset("band")
    tm["ndi52"] = normalized_ratio(tm.swir16, tm.green)
    tm["ndi43"] = normalized_ratio(tm.nir08, tm.red)
    tm["ndi72"] = normalized_ratio(tm.swir22, tm.green)

    d1 = tm.ndi52 <= -0.01
    l2 = d1 & (tm.blue <= 2083.5)
    d3 = tm.swir22 <= 323.5

    l3 = l2 & d3
    w1 = l3 & (tm.ndi43 <= 0.61)

    r3 = l2 & ~d3
    d5 = tm.blue <= 1400.5
    d6 = tm.ndi72 <= -0.23
    d7 = tm.ndi43 <= 0.22
    w2 = r3 & d5 & d6 & d7

    w3 = r3 & d5 & d6 & ~d7 & (tm.blue <= 473.0)

    w4 = r3 & d5 & ~d6 & (tm.blue <= 379.0)
    w7 = r3 & ~d5 & (tm.ndi43 <= -0.01)

    d11 = tm.ndi52 <= 0.23
    l13 = ~d1 & d11 & (tm.blue <= 334.5) & (tm.ndi43 <= 0.54)
    d14 = tm.ndi52 <= -0.12

    w5 = l13 & d14
    r14 = l13 & ~d14
    d15 = tm.red <= 364.5

    w6 = r14 & d15 & (tm.blue <= 129.5)
    w8 = r14 & ~d15 & (tm.blue <= 300.5)

    w10 = (
        ~d1
        & ~d11
        & (tm.ndi52 <= 0.32)
        & (tm.blue <= 249.5)
        & (tm.ndi43 <= 0.45)
        & (tm.red <= 364.5)
        & (tm.blue <= 129.5)
    )

    water = w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w10
    return water.where(tm.red.notnull(), float("nan"))


def wofs_for_year(landsat_xr: DataArray) -> None:
    """landsat_xr is a DataArray with all landsat bands and likely multiple
    readings in time, with correct scale and offset applied"""
    # This needs to be done manually (rescale arg of stack doesn't
    # work because these COGs don't have scale and offset set correctly.
    # AND after the cloud mask / applied only to value layers / etc.
    # Now scale to what wofs tree expects
    l1_scale = 0.0001
    l1_rescale = 1.0 / l1_scale
    landsat_xr = scale_and_offset(landsat_xr, scale=[l1_rescale])

    return (
        wofs(landsat_xr)
        .mean("time")
        .reset_coords(drop=True)
        .rio.write_crs(landsat_xr.rio.crs)
    )


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

    color_ramp_file = Path(__file__).with_name("wofs_color_ramp.txt").as_posix()
    os.makedirs("data", exist_ok=True)
    run_processor(
        year=args.year,
        scene_processor=wofs_for_year,
        dataset_id="wofs",
        color_ramp_file=color_ramp_file,
        run_scenes=args.run_scenes,
        mosaic=args.mosaic,
        tile=args.tile,
        remake_mosaic_for_tiles=args.remake_mosaic_for_tiles,
        output_value_multiplier=10000,
    )
