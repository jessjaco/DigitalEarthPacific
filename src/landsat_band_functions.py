import numpy as np
import xarray as xr
from xarray import DataArray


def ndvi(xr: DataArray) -> DataArray:
    """Receives a DataArray with bands "nir08" & "red" (i.e. landsat data
    as loaded by stackstac.stack and returns a DataArray with a single band
    representing NDVI."""
    nir = xr.sel(band="nir08")
    red = xr.sel(band="red")
    return np.divide(np.subtract(nir, red), np.add(nir, red))
