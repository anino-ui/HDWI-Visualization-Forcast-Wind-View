import os
import base64
from io import BytesIO
from functools import lru_cache
from typing import List
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime as dt
from datetime import datetime, timedelta
import json

# =============================================================================
# Load Configuration
# =============================================================================
CONFIG_FILE = "config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found.")
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

CONFIG = load_config()

# =============================================================================
# Apply Configuration
# =============================================================================

DATA_FOLDER = CONFIG["paths"]["nc_source_folder"]
ZARR_OUTPUT_FOLDER = CONFIG["paths"]["zarr_output_folder"]
VECTOR_COARSEN_FACTOR = CONFIG["data_processing"]["vector_coarsen_factor"]

# Flatten the config structure for easier lookups in the API
VARIABLE_MAP = {k: v["zarr_var"] for k, v in CONFIG["variables"].items()}
VARIABLE_UNITS = {k: v["unit"] for k, v in CONFIG["variables"].items()}

app = FastAPI(title=CONFIG["app_settings"]["title"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_modified_map = {}

# =============================================================================
# Pydantic Models (Unchanged)
# =============================================================================

class PointRequest(BaseModel):
    lat: float
    lon: float
    variable: str

class PointRequestMultiple(BaseModel):
    lat: float
    lon: float
    variable: List[str]

class ProbeRequest(BaseModel):
    lat: float
    lon: float
    timestamp: str
    variable: str

class AreaRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    variables: List[str]
    timestamp: str
    begin_time: str = None  # Optional: Start time for time range selection
    end_time: str = None    # Optional: End time for time range selection
    selected_statistics: List[str] = ["mean", "min", "max"]  # Statistics to plot

# =============================================================================
# Logic
# =============================================================================

@lru_cache(maxsize=16)
def load_dataset(variable_name: str):
    zarr_path = os.path.join(ZARR_OUTPUT_FOLDER, f"{variable_name}.zarr")
    print(f"📦 Loading Zarr for '{variable_name}' from disk...")
    try:
        ds = xr.open_zarr(zarr_path, decode_times=False)
    except FileNotFoundError:
        print(f"❌ ERROR: Zarr file not found: {zarr_path}")
        raise
    except Exception as e:
        print(f"❌ ERROR: Could not open Zarr file {zarr_path}: {e}")
        raise

    if not pd.api.types.is_datetime64_any_dtype(ds.time):
        try:
            decoded_times = pd.to_datetime(ds.time.values, unit='ns')
            ds = ds.assign_coords(time=decoded_times)
        except Exception as e:
            print(f"❌ ERROR: Failed to decode time for {variable_name}: {e}")
            raise
    return ds

def check_reload(variable_name: str):
    global _last_modified_map
    zarr_path = os.path.join(ZARR_OUTPUT_FOLDER, f"{variable_name}.zarr")
    try:
        current_mtime = os.path.getmtime(zarr_path)
        if _last_modified_map.get(variable_name) != current_mtime:
            print(f"🔁 Zarr file for '{variable_name}' has changed. Clearing cache.")
            load_dataset.cache_clear()
            _last_modified_map[variable_name] = current_mtime
    except:
        pass

def _check_nc_file_changes(data_folder: str) -> bool:
    global _last_modified_map
    nc_files = glob.glob(os.path.join(data_folder, "*.nc"))
    if not nc_files:
        return False
    latest_file = max(nc_files, key=os.path.getmtime)
    current_mtime = os.path.getmtime(latest_file)
    cache_key = "__nc_files_timestamp__"
    if _last_modified_map.get(cache_key) != current_mtime:
        print("🔁 Source NetCDF files have changed. Clearing timestamp cache.")
        _last_modified_map[cache_key] = current_mtime
        return True
    return False

@lru_cache(maxsize=1)
def get_all_available_timestamps(data_folder: str) -> List[str]:
    print("Scanning source NetCDF files for all available timestamps...")
    existing_timestamps = set()
    nc_files = glob.glob(os.path.join(data_folder, "*.nc"))
    
    for nc_path in nc_files:
        try:
            with xr.open_dataset(nc_path, decode_times=True, chunks={}) as ds:
                if "time" in ds.coords:
                    for t in ds.time.values:
                        ts_str = pd.to_datetime(t).strftime("%Y-%m-%dT%H_%M_%S")
                        existing_timestamps.add(ts_str)
        except Exception as e:
            print(f"⚠️ Could not read timestamps from {nc_path}: {e}")
            
    return sorted(list(existing_timestamps))

# =============================================================================
# NEW ENDPOINT: Serve Config to Frontend
# =============================================================================
@app.get("/config")
def get_config():
    """Returns the configuration for the frontend to consume."""
    return CONFIG

# =============================================================================
# Existing Endpoints (Modified slightly to use global vars)
# =============================================================================

@app.get("/tile_frames/{prefix}")
def get_tiles_frames(prefix: str):
    if _check_nc_file_changes(DATA_FOLDER):
        get_all_available_timestamps.cache_clear()
    
    try:
        all_timestamps = get_all_available_timestamps(DATA_FOLDER)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to scan source data files.")
    
    frames = []
    for ts_str in all_timestamps:
        try:
            dt_obj = datetime.strptime(ts_str, "%Y-%m-%dT%H_%M_%S")
            folder_name = f"{prefix}.{ts_str}"
            frames.append({
                "time": dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "folder": folder_name
            })
        except:
            continue
    return {"tileFrames": frames}

@app.post("/probe/")
def probe_data(probe: ProbeRequest):
    var_key = probe.variable
    if var_key not in VARIABLE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown variable: {var_key}")

    var_name = VARIABLE_MAP[var_key]
    try:
        check_reload(var_name)
        ds = load_dataset(var_name)
        da = ds[var_name]
        value = da.sel(latitude=probe.lat, longitude=probe.lon, method="nearest").sel(time=probe.timestamp, method="nearest")
        
        val_out = None if np.isnan(value) else float(value)
        return {"value": val_out, "units": VARIABLE_UNITS.get(var_key)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/popup_timeserie/")
def get_popup_timeseries_single(point: PointRequest):
    var_key = point.variable
    if var_key not in VARIABLE_MAP:
        return {"error": f"Unknown variable: {var_key}"}

    var_name = VARIABLE_MAP[var_key]
    try:
        check_reload(var_name)
        ds = load_dataset(var_name)
        da = ds[var_name]

        ts_da = da.sel(latitude=point.lat, longitude=point.lon, method="nearest")
        df = ts_da.to_dataframe().reset_index().dropna().sort_values("time")
        time_filter_start = datetime.now() - dt.timedelta(days=3.5)
        df = df[df["time"] >= time_filter_start]

        if df.empty:
            return {"error": "No data found."}

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["time"], df[var_name], marker="o", linestyle="-", color="steelblue")
        ax.set_title(f"{CONFIG['variables'][var_key]['name']}")
        ax.set_ylabel(f"Value ({VARIABLE_UNITS.get(var_key)})")
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "series": [{"datetime": t.isoformat(), "value": float(v)} for t, v in zip(df["time"], df[var_name])],
            "image_base64": img_base64,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/popup_timeserie_multiple/")
def get_popup_timeseries_multiple(point: PointRequestMultiple):
    plot_data = []
    results_data = []
    time_filter_start = datetime.now() - dt.timedelta(days=3.5)

    for var_key in point.variable:
        if var_key not in VARIABLE_MAP: continue
        try:
            var_name = VARIABLE_MAP[var_key]
            check_reload(var_name)
            ds = load_dataset(var_name)
            da = ds[var_name]
            ts_da = da.sel(latitude=point.lat, longitude=point.lon, method="nearest")
            df = ts_da.to_dataframe().reset_index().dropna().sort_values("time")
            df = df[df["time"] >= time_filter_start]
            
            if df.empty: continue

            results_data.append({
                "variable": var_key,
                "series": [{"datetime": t.isoformat(), "value": float(v)} for t, v in zip(df["time"], df[var_name])]
            })
            plot_data.append((df["time"], df[var_name], CONFIG['variables'][var_key]['name'], VARIABLE_UNITS.get(var_key)))
        except: continue

    if not plot_data:
        return {"error": "No data found."}

    num_subplots = len(plot_data)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 3 * num_subplots), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, (times, values, v_name, units) in enumerate(plot_data):
        axes[i].plot(times, values, marker="o", linestyle="-", color="steelblue", markersize=4)
        axes[i].set_title(v_name)
        axes[i].set_ylabel(f"({units})")
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"series": results_data, "image_base64": img_base64}

@app.post("/area_analysis/")
def area_analysis(req: AreaRequest):
    """Performs statistical analysis for multiple variables over a geographic area."""
    analysis_results = {}

    for var_key in req.variables:
        if var_key not in VARIABLE_MAP:
            continue

        var_name = VARIABLE_MAP[var_key]
        var_unit = VARIABLE_UNITS.get(var_key, "")

        try:
            check_reload(var_name)
            ds = load_dataset(var_name)

            # 1. Drop Duplicates (from previous fix)
            if "time" in ds.coords:
                ds = ds.drop_duplicates(dim="time")

            # 2. FIX: Sort Coordinates
            # Sort time Ascending (Oldest -> Newest)
            if "time" in ds.dims:
                ds = ds.sortby("time")

            # Sort Latitude Descending (90 -> -90) to match your slice(max, min)
            if "latitude" in ds.dims:
                ds = ds.sortby("latitude", ascending=False)

            # Sort Longitude Ascending (-180 -> 180)
            if "longitude" in ds.dims:
                ds = ds.sortby("longitude")

            da = ds[var_name]

            # Select spatial area and time slice
            area_da = da.sel(
                latitude=slice(req.lat_max, req.lat_min), # Works because we forced lat to be Descending
                longitude=slice(req.lon_min, req.lon_max),
            )

            # This line specifically fails if 'time' is not unique
            current_frame_da = area_da.sel(time=req.timestamp, method="nearest")

            # --- Calculate Stats for the current timestamp ---
            stats_current = {}
            if current_frame_da.size > 0:
                stats_current = {
                    "mean": float(current_frame_da.mean()),
                    "min": float(current_frame_da.min()),
                    "max": float(current_frame_da.max()),
                    "std_dev": float(current_frame_da.std()),
                    "median": float(np.nanmedian(current_frame_da.values)),
                }

            # --- Time Range Selection ---
            # Use custom time range if provided, otherwise use default 3.5 days
            if req.begin_time and req.end_time:
                time_filter_start = pd.to_datetime(req.begin_time)
                time_filter_end = pd.to_datetime(req.end_time)
                ts_da = area_da.sel(time=slice(time_filter_start, time_filter_end))
            elif req.begin_time:
                time_filter_start = pd.to_datetime(req.begin_time)
                ts_da = area_da.sel(time=slice(time_filter_start, None))
            elif req.end_time:
                time_filter_end = pd.to_datetime(req.end_time)
                ts_da = area_da.sel(time=slice(None, time_filter_end))
            else:
                time_filter_start = datetime.now() - timedelta(days=3.5)
                ts_da = area_da.sel(time=slice(time_filter_start, None))

            # --- Calculate statistics over time ---
            stat_functions = {
                "mean": lambda x: x.mean(dim=["latitude", "longitude"]),
                "min": lambda x: x.min(dim=["latitude", "longitude"]),
                "max": lambda x: x.max(dim=["latitude", "longitude"]),
                "std": lambda x: x.std(dim=["latitude", "longitude"]),
                "median": lambda x: x.median(dim=["latitude", "longitude"])
            }

            # Calculate all requested statistics
            stat_series = {}
            for stat_name in req.selected_statistics:
                if stat_name in stat_functions:
                    stat_series[stat_name] = stat_functions[stat_name](ts_da)

            # --- Generate Combined Time Series Plot (Original) ---
            fig_ts, ax_ts = plt.subplots(figsize=(10, 5))

            colors = {"mean": "blue", "min": "green", "max": "red", "std": "orange", "median": "purple"}
            linestyles = {"mean": "-", "min": "--", "max": "--", "std": "-.", "median": ":"}

            for stat_name, stat_data in stat_series.items():
                ax_ts.plot(
                    stat_data.time,
                    stat_data,
                    label=stat_name.capitalize(),
                    color=colors.get(stat_name, "black"),
                    linestyle=linestyles.get(stat_name, "-")
                )

            # Add fill between min-max if both are selected
            if "min" in stat_series and "max" in stat_series:
                ax_ts.fill_between(
                    stat_series["min"].time,
                    stat_series["min"],
                    stat_series["max"],
                    color="gray",
                    alpha=0.2,
                    label="Min-Max Range"
                )

            ax_ts.set(
                title=f"Time Series for {var_name.replace('_', ' ').title()} in Selected Area",
                xlabel="Time",
                ylabel=f"Value ({var_unit})",
            )
            ax_ts.legend()
            ax_ts.grid(True, linestyle='--', alpha=0.6)
            fig_ts.autofmt_xdate()
            fig_ts.tight_layout()

            buf_ts = BytesIO()
            fig_ts.savefig(buf_ts, format="png", dpi=120)
            plt.close(fig_ts)
            ts_img_base64 = base64.b64encode(buf_ts.getvalue()).decode("utf-8")

            # --- Generate Individual Statistical Plots ---
            individual_stat_plots = {}
            for stat_name, stat_data in stat_series.items():
                fig_stat, ax_stat = plt.subplots(figsize=(10, 5))
                ax_stat.plot(
                    stat_data.time,
                    stat_data,
                    marker="o",
                    markersize=3,
                    linestyle="-",
                    color=colors.get(stat_name, "steelblue"),
                    linewidth=2
                )
                ax_stat.set(
                    title=f"{stat_name.capitalize()} - {var_name.replace('_', ' ').title()}",
                    xlabel="Time",
                    ylabel=f"{stat_name.capitalize()} Value ({var_unit})",
                )
                ax_stat.grid(True, linestyle='--', alpha=0.6)
                fig_stat.autofmt_xdate()
                fig_stat.tight_layout()

                buf_stat = BytesIO()
                fig_stat.savefig(buf_stat, format="png", dpi=120)
                plt.close(fig_stat)
                individual_stat_plots[stat_name] = base64.b64encode(buf_stat.getvalue()).decode("utf-8")

            # --- Generate Box Plot of Overall Distribution ---
            all_values = ts_da.values.flatten()
            all_values = all_values[~np.isnan(all_values)]

            fig_bp, ax_bp = plt.subplots(figsize=(6, 5))
            if len(all_values) > 0:
                ax_bp.boxplot(all_values, vert=True, patch_artist=True)
            ax_bp.set(
                title=f"Data Distribution for {var_name.replace('_', ' ').title()}",
                ylabel=f"Value ({var_unit})",
            )
            ax_bp.set_xticks([]) # Hide x-axis ticks
            ax_bp.grid(True, linestyle='--', alpha=0.6)
            fig_bp.tight_layout()

            buf_bp = BytesIO()
            fig_bp.savefig(buf_bp, format="png", dpi=120)
            plt.close(fig_bp)
            bp_img_base64 = base64.b64encode(buf_bp.getvalue()).decode("utf-8")

            # --- Generate Static Plot of Current Selected Square ---
            fig_static, ax_static = plt.subplots(figsize=(8, 6))

            # Plot the current frame data
            im = ax_static.imshow(
                current_frame_da.values,
                extent=[req.lon_min, req.lon_max, req.lat_min, req.lat_max],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )

            ax_static.set(
                title=f"Current Frame - {var_name.replace('_', ' ').title()}\n{req.timestamp}",
                xlabel="Longitude",
                ylabel="Latitude",
            )

            # Add colorbar
            cbar = fig_static.colorbar(im, ax=ax_static)
            cbar.set_label(f"Value ({var_unit})")

            ax_static.grid(True, linestyle='--', alpha=0.3, color='white')
            fig_static.tight_layout()

            buf_static = BytesIO()
            fig_static.savefig(buf_static, format="png", dpi=120)
            plt.close(fig_static)
            static_img_base64 = base64.b64encode(buf_static.getvalue()).decode("utf-8")

            analysis_results[var_key] = {
                "variable_name": var_name.replace("_", " ").title(),
                "units": var_unit,
                "stats_current_frame": stats_current,
                "timeseries_plot": ts_img_base64,
                "boxplot": bp_img_base64,
                "static_plot": static_img_base64,
                "individual_stat_plots": individual_stat_plots,
            }

        except Exception as e:
            analysis_results[var_key] = {
                "error": f"Failed to analyze {var_name}: {str(e)}"
            }

    if not analysis_results:
        raise HTTPException(status_code=404, detail="No valid data found for any selected variable.")

    return analysis_results

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return 0.0 if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@app.get("/wind_vector_data")
async def get_wind_data(timestamp: str = Query(...)):
    try:
        # NOTE: Using hardcoded u10/v10 keys from the config
        u_var = VARIABLE_MAP['u10']
        v_var = VARIABLE_MAP['v10']
        
        check_reload(u_var)
        check_reload(v_var)
        
        ds_u = load_dataset(u_var)[u_var].sel(time=timestamp, method='nearest')
        ds_v = load_dataset(v_var)[v_var].sel(time=timestamp, method='nearest')

        coarse_u = ds_u.coarsen(latitude=VECTOR_COARSEN_FACTOR, longitude=VECTOR_COARSEN_FACTOR, boundary="trim").mean()
        coarse_v = ds_v.coarsen(latitude=VECTOR_COARSEN_FACTOR, longitude=VECTOR_COARSEN_FACTOR, boundary="trim").mean()

        lats = coarse_u['latitude'].values
        lons = coarse_u['longitude'].values
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            coarse_u = np.flip(coarse_u, axis=0)
            coarse_v = np.flip(coarse_v, axis=0)

        lat_res = abs(round(lats[1] - lats[0], 6))
        lon_res = abs(round(lons[1] - lons[0], 6))

        u_header = {
            "parameterCategory": 2, "parameterNumber": 2, "lo1": lons[0], "la1": lats[0],
            "dx": lon_res, "dy": lat_res, "nx": len(lons), "ny": len(lats), "units": "m s-1"
        }
        v_header = u_header.copy()
        v_header["parameterNumber"] = 3

        velocity_data = [
            {"header": u_header, "data": coarse_u.values.flatten()},
            {"header": v_header, "data": coarse_v.values.flatten()}
        ]
        return json.loads(json.dumps(velocity_data, cls=NpEncoder))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG["app_settings"]["host"], port=CONFIG["app_settings"]["port"])