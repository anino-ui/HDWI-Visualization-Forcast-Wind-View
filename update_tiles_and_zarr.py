import os
import sys
import subprocess
import shutil
import numpy as np
import xarray as xr
import rioxarray 
import pandas as pd
import glob
import time
import json

# Load Config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

NC_FOLDER = CONFIG["paths"]["nc_source_folder"]
TILE_OUTPUT_FOLDER = CONFIG["paths"]["tile_output_folder"]
ZARR_OUTPUT_FOLDER = CONFIG["paths"]["zarr_output_folder"]
ZOOM_LEVELS = CONFIG["data_processing"]["zoom_levels"]

# Create Mappings from Config
API_PREFIX_TO_NC_VAR = {k: v["nc_var"] for k, v in CONFIG["variables"].items()}
API_PREFIX_TO_ZARR_VAR = {k: v["zarr_var"] for k, v in CONFIG["variables"].items()}
COLOR_FILES = {k: v["color_table"] for k, v in CONFIG["variables"].items()}

print("--- Initializing: Clearing output directories ---")
for folder in [TILE_OUTPUT_FOLDER, ZARR_OUTPUT_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def _generate_tile(da: xr.DataArray, prefix: str, timestamp_str: str, min_max_vals: dict) -> None:
    tile_timestamp_folder = f"{prefix}.{timestamp_str}"
    tile_dir = os.path.join(TILE_OUTPUT_FOLDER, tile_timestamp_folder)
    print(f"⚙️ Generating tiles for {tile_timestamp_folder}...")

    da = da.rio.set_spatial_dims("longitude", "latitude").rio.write_crs("EPSG:4326")
    tmp_tif = os.path.join(TILE_OUTPUT_FOLDER, f"{tile_timestamp_folder}.tif")
    vrt_path = os.path.join(TILE_OUTPUT_FOLDER, f"{tile_timestamp_folder}.vrt")
    color_tif = os.path.join(TILE_OUTPUT_FOLDER, f"{tile_timestamp_folder}_color.tif")

    da.rio.to_raster(tmp_tif)
    
    translate_cmd = [
        "gdal_translate", "-of", "VRT", "-ot", "Byte",
        "-scale", str(int(min_max_vals["min"])), str(int(min_max_vals["max"])), "0", "255",
        tmp_tif, vrt_path,
    ]
    subprocess.run(translate_cmd, check=True)

    color_cmd = ["gdaldem", "color-relief", vrt_path, COLOR_FILES[prefix], color_tif, "-alpha"]
    subprocess.run(color_cmd, check=True)

    tile_cmd = [
        sys.executable, "/home/anino/data-sbarc/miniconda3/envs/leafmap/bin/gdal2tiles.py",
        "-z", ZOOM_LEVELS, "-w", "none", color_tif, tile_dir,
    ]
    subprocess.run(tile_cmd, check=True)

    for tmp in (tmp_tif, vrt_path, color_tif):
        if os.path.exists(tmp): os.remove(tmp)

def _update_zarr(da: xr.DataArray, prefix: str, timestamp_str: str) -> None:
    var_name = API_PREFIX_TO_ZARR_VAR[prefix]
    time_val = pd.to_datetime(timestamp_str, format="%Y-%m-%dT%H_%M_%S")
    zarr_path = os.path.join(ZARR_OUTPUT_FOLDER, f"{var_name}.zarr")
    
    da = da.rio.set_spatial_dims("longitude", "latitude").rio.write_crs("EPSG:4326")
    da = da.expand_dims(time=[time_val])
    ds_to_save = da.to_dataset(name=var_name)

    if not os.path.exists(zarr_path):
        ds_to_save.to_zarr(zarr_path, mode="w", consolidated=True, encoding={"time": {"units": "nanoseconds since 1970-01-01"}})
    else:
        ds_to_save.to_zarr(zarr_path, mode="a", append_dim="time", consolidated=True)

def main():
    print(f"--- Loading data from: {NC_FOLDER} ---")
    file_list = sorted(glob.glob(os.path.join(NC_FOLDER, "*.nc")))
    if not file_list: return

    datasets = [xr.open_dataset(f) for f in file_list]
    ds_merged = xr.merge(datasets)
    ds_raw = ds_merged.isel(members=-1)

    # --- DERIVED VARIABLES LOGIC ---
    # NOTE: This remains specific to the current problem (Wind Speed calculation).
    # If you change problems, check if 'u10' and 'v10' exist before doing this.
    ds = ds_raw.copy()
    if 'u10' in ds and 'v10' in ds:
        print("Calculating derived wind speed/direction...")
        ds['ws'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        direction_rad = np.arctan2(ds['u10'], ds['v10'])
        ds['dir'] = (np.rad2deg(direction_rad) + 180) % 360
    
    DYNAMIC_MIN_MAX = {}
    for prefix, conf in CONFIG["variables"].items():
        try:
            DYNAMIC_MIN_MAX[prefix] = {"min": conf["min_val"], "max": conf["max_val"]}
        except:
            DYNAMIC_MIN_MAX[prefix] = {"min": min(ds[prefix]), "max": max(ds[prefix])}

    for time_step in ds.time:
        timestamp_str = pd.to_datetime(time_step.values).strftime("%Y-%m-%dT%H_%M_%S")
        ds_at_time = ds.sel(time=time_step)

        for prefix, nc_var in API_PREFIX_TO_NC_VAR.items():
            if nc_var in ds_at_time:
                try:
                    _generate_tile(ds_at_time[nc_var], prefix, timestamp_str, DYNAMIC_MIN_MAX[prefix])
                    _update_zarr(ds_at_time[nc_var], prefix, timestamp_str)
                except Exception as e:
                    print(f"Error processing {prefix}: {e}")

    ds.close()

if __name__ == "__main__":
    main()