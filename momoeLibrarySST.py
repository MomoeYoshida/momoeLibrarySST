# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Get the packages and version nums that work from your jupyter environment.
# Do this and upload the file to HPC > pip3 install --user -r requirements.txt everytime you add/change below module importing
# #!pip freeze > requirements.txt

# %%
# Automatically export .py version of this library, version controlled by git connected to remote momoeLibrarySST.git.
{
  "jupytext": {
    "formats": "ipynb,momoeLibrarySST.py"
  }
}

# %%
import os
import datetime
from scipy.io import loadmat
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

# %%
# HPC
#home_dir = "/gpfs01/v2/Q9157/momoe/geo_polar_blended_sst/Linux_JCUHPC/blended_home"

# Local
home_dir = "/Users/momotalo/Library/CloudStorage/OneDrive-JamesCookUniversity/PhD_MomoeYoshida/Coding/blended_home"


# %%
# Usage: generate_julian_days("20250131", "20250205")

def generate_julian_days(start_date, end_date):
    # Convert string → datetime
    print(f'Generating Julian days from {start_date} to {end_date}...')
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end   = pd.to_datetime(end_date,   format="%Y%m%d")
    
    # Generate daily range
    dates = pd.date_range(start=start, end=end, freq="D")
    
    # Convert to Julian format for file access
    julian_days = [d.strftime("%Y_%j") for d in dates]
    
    return julian_days


# %%
# Usage: julian_to_datetime("2025_031")
def julian_to_datetime(julian_str):
    """Convert julian date to calendar date."""
    return datetime.datetime.strptime(julian_str, "%Y_%j")


# %%
# Usage: extract_var_from_l4_mat_file("sst_analysis", "2025_010", "/Analysis")

def extract_var_from_l4_mat_file(varname, date, file_dir="/Analysis"):
    """Read a .mat file and extract and display a variable."""
    print(f'Extracting a variable {varname} for date {date}...')
    file = home_dir+file_dir+"/"+varname+"_"+date+".mat"
    print(f'Reading a file {file}...')
    data = loadmat(file)
    print(data.keys())
    var = data[varname]

    # If min is -999 then replace them with NANs.
    if np.min(var)==-999.0:
        print("Replacing -999 with NANs...")
        var_clean = np.where(var == -999, np.nan, var.astype(float))
    else:
        var_clean = var
    
    print(f'Variable: {varname}')
    print(f'Shape: {var.shape}') # print(f'text {variable}')
    print(var)
    print(f'Min: {np.nanmin(var)}') # when data contains nans
    print(f'Max: {np.nanmax(var)}')

    return var_clean
    


# %%
# Usage: add_lats_lons(var)

def add_lats_lons(var, est_res=0.05):
    """Create xarray DataArray including lats and lons."""
    # Read the shape of var.
    nlat = var.shape[0]
    nlon = var.shape[1]
    lat_res = 180 / nlat
    lon_res = 360 / nlon
    print(f'Resolution based on nlat and nlon: {lat_res}°x{lon_res}°')
    if (lat_res != est_res) or (lon_res != est_res):
        print(f'***Warning: lat and/or lon_res is different to the estimated resolution {est_res}.')

    # --- Create lat/lon arrays
    lats = np.linspace(-90 + lat_res/2, 90 - lat_res/2, nlat)
    lons = np.linspace(-180 + lon_res/2, 180 - lon_res/2, nlon)
    
    # --- Create xarray DataArray
    da = xr.DataArray(
        var,
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
        name="var"
    )
    return da
    


# %%
# Usage: trim_da(da, extent=(140,155,-25,-10))

def trim_da(da, extent=(140,155,-25,-10)):
    """Trim an xarray DataArray by geographic extent."""
    print(f'Trimming xarray data array by geographic extent {extent}...')
    lon_min, lon_max, lat_min, lat_max = extent

    # Trim da.
    trim_da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    return trim_da


# %%
# Usage: save_ts_nc("20250131", "20250205")
# TNT: may need to add some codes to ensure coordinates match between sst and cm and files?
# HPC: Save this as .py > Upload on HPC Code/ > Run a function in the compute node
# Default extent: GBR

def save_ts_nc(start_date, end_date):
    """Save time series of variables in a netCDF file.""" 
    filename = home_dir+'/Output/'+"gbr_sst_cm_timeseries_"+start_date+"-"+end_date+".nc"
    # --- Delete existing file if it exists ---
    if os.path.exists(filename):
        print(f"Deleting existing file: {filename}")
        os.remove(filename)
    
    julian_days = generate_julian_days(start_date, end_date)
    
    ds_list = []

    for jd in julian_days:
        print(f'***Processing {jd}...')
        # --- SST ---
        sst = extract_var_from_l4_mat_file("sst_analysis", jd, "/Analysis")
        da_sst = trim_da(add_lats_lons(sst))
    
        # --- Correlation map ---
        cm = extract_var_from_l4_mat_file("correlation_map", jd, "/Analysis")
        da_cm = trim_da(add_lats_lons(cm))
    
        # --- Convert time ---
        time = julian_to_datetime(jd)
    
        # --- Convert to Dataset ---
        ds_day = xr.Dataset(
            {
                "sst": da_sst,
                "correlation_map": da_cm
            }
        )
    
        # --- Add time dimension ---
        ds_day = ds_day.expand_dims(time=[time])
        ds_list.append(ds_day)
    for i, ds in enumerate(ds_list):
        if i == 0:
            ref_lat = ds.lat
            ref_lon = ds.lon
        else:
            print(i, ds.lat.equals(ref_lat), ds.lon.equals(ref_lon))
    ds_all = xr.concat(ds_list, dim="time")
    print('Saving a time-series into a netCDF file .nc...')
    ds_all.to_netcdf(filename)


# %%
"""
night_plus_minus_hrs: local solar time ± night_plus_minus_hrs hours if you want to adjust the nighttime thresholds
"""

def extract_nighttime_AIMS_InWT_stdvals(aims_csv_filename, sun_csv_filename, 
                                        start_date=datetime.date(2025, 1, 1), end_date=datetime.date(2025, 4, 1), 
                                        night_plus_minus_hrs=0):
    """Extract AIMS in-water temperature data (logger) nighttime-only based on the local sunset and sunrise times."""
    
    # 1. Read watertemp data and sun_rise&set data.
    temp_df = pd.read_csv(aims_csv_filename)
    sun_df = pd.read_csv(sun_csv_filename)

    # Check UTC consistency.
    if not (sun_df["utc_offset"] == "UTC+0:00").all():
        raise ValueError(
        "ERROR: sun_csv_file contains non-UTC+0:00 values in 'utc_offset'. "
        "Please ensure all rows are 'UTC+0:00'.")
    if not temp_df["time"].astype(str).str.endswith("+00:00").all():
        raise ValueError(
        "ERROR: aims_csv_file contains non-UTC+00:00 timestamps in 'time'. "
        "Please ensure all rows end with '+00:00' (UTC).")

    
    sun_df['rise_date'] = pd.to_datetime(sun_df['rise_date'], format='ISO8601', utc=True)
    sun_df['set_date']  = pd.to_datetime(sun_df['set_date'],  format='ISO8601', utc=True)
    temp_df['time']     = pd.to_datetime(temp_df['time'],     format='ISO8601', utc=True)
    sun_df = sun_df.sort_values(['set_date', 'rise_date']).reset_index(drop=True)
    temp_df = temp_df.sort_values("time")

    # 2. Create a list of dates from start_date to end_date (inclusive).
    dates = []  # initialise
    date_now = start_date
    while date_now <= end_date:  # end_date (inclusive)
        dates.append(pd.Timestamp(date_now, tz="UTC"))
        date_now += datetime.timedelta(days=1)

    # 3. Create numpy arrays of watertemp values and times from "temp_df".
    watertemps = np.array(temp_df['qc_val'], dtype=float)
    times = np.array(temp_df['time'])

    # 4. Find nighttime watertemp values based on the "set_rise_df".
    nighttime_means  = [] # initialise
    nighttime_stdevs = []
    pointcounts = [] # number of nighttime points used for the averaging for each date
    
    # Compute a nightime-only watertemp value corresponding to each date.
    for day_start in dates:
        day_end = day_start + pd.Timedelta(days=1)

        # --- Sunset: last sunset BEFORE end of day ---
        sunset_candidates = sun_df[sun_df["set_date"] <= day_end]
        if sunset_candidates.empty:
            raise ValueError(f"No sunset found before {day_end}")
        sunset_time = sunset_candidates.iloc[-1]["set_date"]

        # --- Sunrise: first sunrise AFTER start of day ---
        sunrise_candidates = sun_df[sun_df["rise_date"] >= day_start]
        if sunrise_candidates.empty:
            raise ValueError(f"No sunrise found after {day_start}")
        sunrise_time = sunrise_candidates.iloc[0]["rise_date"]

        # 9. Night window (with optional buffer)
        nighttime_lowerbound = sunset_time + pd.Timedelta(hours=night_plus_minus_hrs)
        nighttime_upperbound = sunrise_time - pd.Timedelta(hours=night_plus_minus_hrs)
        
        # Find nighttime indices.
        print(f'Finding watertemp values where {nighttime_lowerbound} < "times" < {nighttime_upperbound}...')
        nighttime_indices = np.where((times > nighttime_lowerbound) & (times < nighttime_upperbound))[0]

        # Count number of nighttime points
        pointcount = len(nighttime_indices)
        print(f'Found {pointcount} nighttime data points')
        pointcounts.append(pointcount)
        
        # Compute mean & stdev of nighttime watertemp values using Welford.
        mean_iw, stdev_iw = welford_mean_stdev(watertemps[nighttime_indices])
        nighttime_means.append(mean_iw)
        nighttime_stdevs.append(stdev_iw)
        
    
    lat_vals = temp_df['lat'].unique()
    lon_vals = temp_df['lon'].unique()

    if len(lat_vals) != 1 or len(lon_vals) != 1:
        raise ValueError("Latitude or longitude has multiple values — expected a single site.")

    lat = lat_vals[0]
    lon = lon_vals[0]

    # 5. Save as NetCDF.
    nighttime_means = np.array(nighttime_means, dtype=float)
    nighttime_stdevs = np.array(nighttime_stdevs, dtype=float)
    pointcounts = np.array(pointcounts, dtype=int)

    time_out = np.array(dates, dtype="datetime64[ns]")

    ds = xr.Dataset(
        data_vars=dict(
            nighttime_mean=("time", nighttime_means),
            nighttime_stdev=("time", nighttime_stdevs),
            nighttime_npoints=("time", pointcounts),
        ),
        coords=dict(
            time=time_out,
            lat=lat,
            lon=lon
        ),
        attrs=dict(
            description="Nighttime-only in-water temperature statistics derived from AIMS logger",
            method="Sunset-to-sunrise windowing with UTC-based astronomical times",
            night_buffer_hours=night_plus_minus_hrs
        )
    )

    def fmt(x):
        return f"{x:.4f}"

    lat_str = fmt(lat).replace(".", "p").replace("-", "minus")
    lon_str = fmt(lon).replace(".", "p").replace("-", "minus")

    output_nc = home_dir+'/Output/'+f"nighttime_AIMS_InWT_lat{lat_str}_lon{lon_str}.nc"

    # 6. Save NetCDF
    ds.to_netcdf(output_nc)

    print(f"\nSaved NetCDF file: {output_nc}")


# %%
def welford_mean_stdev(values):
    """Compute mean and sample stdev using Welford’s algorithm."""
    n = 0
    mean = 0.0
    M2 = 0.0

    for v in values:
        if np.isnan(v):
            continue
        n += 1
        delta = v - mean
        mean += delta / n
        M2 += delta * (v - mean)

    if n == 0:
        return float('nan'), float('nan')
    if n == 1:
        return mean, float('nan')   # mean is valid, stdev not defined

    stdev = math.sqrt(M2 / (n - 1))  # sample stdev
    return mean, stdev


# %%
def extract_lat_lon(iwt_file):
    """Extract lat and lon values from input filename (InWT data file e.g., nighttime_AIMS_InWT_latminus18p5691_lon146p4823.nc)."""
    # Extract the lat/lon part using regex
    match = re.search(r'lat([a-z0-9p\-]+)_lon([a-z0-9p\-]+)', iwt_file)
    
    if not match:
        raise ValueError("Could not find lat/lon in filename")
    
    lat_str, lon_str = match.groups()
    
    # Convert encoding back to float
    def decode(coord):
        coord = coord.replace("minus", "-").replace("p", ".")
        return float(coord)
    
    lat_input = decode(lat_str)
    lon_input = decode(lon_str)
    
    return lat_input, lon_input


# %%
def extract_pixel_timeseries(sst_file, iwt_file):
    """Merge (matching dates) and extract time series of InWT, SST, SST–InWT and RMSE."""
    lat_input, lon_input = extract_lat_lon(iwt_file)
    
    sst_ds = xr.open_dataset(sst_file)
    iwt_ds = xr.open_dataset(iwt_file)
    
    # Extract both variables at nearest pixel.
    sst_pixel = sst_ds.sel(lat=lat_input, lon=lon_input, method="nearest")

    # Print actual selected location.
    pixel_lat = float(sst_pixel.lat.values)
    pixel_lon = float(sst_pixel.lon.values)
    
    print(f"Requested: lat={lat_input}, lon={lon_input}")
    print(f"Closest pixel: lat={pixel_lat:.3f}, lon={pixel_lon:.3f}") # display artifact

    # Align time (intersection only) and merge datasets.
    sst_pixel, iwt_ds = xr.align(sst_pixel, iwt_ds, join="inner")
    iwt_ds = iwt_ds.drop_vars(["lat", "lon"], errors="ignore")
    merged_ds = xr.merge([sst_pixel, iwt_ds])

    # Calculate SST – InWT.
    ts_sst = merged_ds["sst"]
    ts_iwt = merged_ds["nighttime_mean"]
    merged_ds["sst_minus_nighttime_mean"] = ts_sst - ts_iwt

    # Calculate RMSE between SST and InWT.
    rmse_val = np.sqrt(((ts_sst - ts_iwt) ** 2).mean().values) # single scalar
    merged_ds["rmse"] = xr.DataArray(rmse_val)

    # Save NetCDF.
    iwt_base = os.path.basename(iwt_file).replace(".nc", "") # remove ".nc"
    sst_base = os.path.basename(sst_file).replace("gbr_", "")
    time_vals = pd.to_datetime(merged_ds.time.values)
    start_date = time_vals.min().strftime("%Y%m%d")
    end_date   = time_vals.max().strftime("%Y%m%d")
    date_part = f"{start_date}-{end_date}"
    sst_base = sst_base.split("_timeseries_")[0] + f"_timeseries_{date_part}.nc"
    filename = f"{iwt_base}_{sst_base}"
    output_nc = home_dir+'/Output/'+filename
    merged_ds.to_netcdf(output_nc)
    print(f"\nSaved NetCDF file: {output_nc}")

    mark_pixel_point(pixel_lat, lat_input, pixel_lon, lon_input)
    plot_ts(merged_ds)
    plot_ts_diff(merged_ds)
    
    return merged_ds


# %%
def plot_ts(ds, transparent=False):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    if transparent:
        fig.patch.set_alpha(0)
        ax1.set_facecolor((0, 0, 0, 0))

    # --- Extract variables safely ---
    ts_sst = ds["sst"]
    ts_iwt = ds["nighttime_mean"]
    ts_cm  = ds["correlation_map"]

    # --- Left axis (Temperature) ---
    temps = []
    
    ax1.plot(ts_sst.time, ts_sst, color="lightblue", marker="o", label="SST")
    temps.append(ts_sst.values)

    ax1.plot(ts_iwt.time, ts_iwt, color="darkblue", marker="o", label="InWT")
    temps.append(ts_iwt.values)

    ax1.set_ylabel("Temperature (°C)", color="black")

    # Dynamic ylim
    all_temp = np.concatenate([t.flatten() for t in temps])
    ax1.set_ylim(all_temp.min() - 0.5, all_temp.max() + 0.5)

    # --- Right axis (other variables) ---
    ax2 = ax1.twinx()
    ax2.plot(ts_cm.time, ts_cm, color="tab:red", marker="s", label="Correlation Map")
    ax2.set_ylabel("Correlation Map", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # --- X axis ---
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.tick_params(axis='x', rotation=60)

    # --- Legend (combined) ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


# %%
def plot_ts_diff(ds, transparent=False):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    if transparent:
        fig.patch.set_alpha(0)
        ax1.set_facecolor((0, 0, 0, 0))

    # --- Extract variables safely ---
    ts_diff = ds["sst_minus_nighttime_mean"]
    ts_cm  = ds["correlation_map"]

    # --- Left axis (Temperature) ---
    
    ax1.plot(ts_diff.time, ts_diff, color="blue", marker="o", label="SST–InWT")
    ax1.set_ylabel("SST–InWT (°C)", color="black")
    ax1.axhline(y=0, color="black", linestyle="--")


    # --- Right axis (other variables) ---
    ax2 = ax1.twinx()
    ax2.plot(ts_cm.time, ts_cm, color="tab:red", marker="s", label="Correlation Map")
    ax2.set_ylabel("Correlation Map", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # --- X axis ---
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.tick_params(axis='x', rotation=60)

    # --- Legend (combined) ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


# %%
def plot_scatter(pixel_ts, x="correlation_map", y="sst_minus_nighttime_mean", xlabel="Correlation Map", ylabel="SST – InWT (°C)"):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # --- Extract variables safely ---
    dep_var = pixel_ts[y]
    indep_var  = pixel_ts[x]
    
    ax.scatter(indep_var, dep_var, color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linestyle="--") # horizontal
    ax.axvline(8, color="grey") # vertical
    ax.axvline(16, color="grey")
    ax.axvline(32, color="grey")
    
    plt.tight_layout()
    plt.show()


# %%
"""
This function can be used to check whether the temperature logger location (point) is within the CoralTemp 
0.05x0.05 pixel (pixel).
"""

def mark_pixel_point(pixel_lat, lat_input, pixel_lon, lon_input, extent=(140,155,-25,-10), projection=ccrs.PlateCarree(central_longitude=0), 
                     res=0.05, marker_type='.', pixel_color='cyan', point_color='blue', extent_var = 0.1, point_size=5):
    """Mark a pixel and point in terms of longitude and latitude."""
    
    data_crs = ccrs.PlateCarree(central_longitude=0)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,9),
                                   gridspec_kw={'width_ratios': [8, 4]},
                                   subplot_kw={'projection': projection})
    
    if extent is not None:
        ax1.set_extent(extent)
    
    else:
        # Set the extent to the limits of the projection.
        ax1.set_global()

    ax2.set_extent([lon_input-extent_var, lon_input+extent_var, lat_input-extent_var, lat_input+extent_var])
    ax2.annotate(u'\u25B2\nN', xy=(0.04, 0.85), xycoords='axes fraction', fontsize=14)
    
    ax1.coastlines(resolution='10m')
    ax2.coastlines(resolution='10m')
    
    gl1 = ax1.gridlines(draw_labels=True)
    gl2 = ax2.gridlines(draw_labels=True)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.left_labels = True
    gl1.bottom_labels = True
    gl2.top_labels = True
    gl2.right_labels = True
    gl2.left_labels = False
    gl2.bottom_labels = False
    
    ax1.scatter(lon_input, lat_input, color=point_color, transform=data_crs)
    
    # Mark a pixel.
    rect = patches.Rectangle(((pixel_lon-res/2), (pixel_lat-res/2)), res, res, transform=data_crs, linewidth=0, color=pixel_color)
    ax2.add_patch(rect)
    
    # Mark a point.
    ax2.scatter(lon_input, lat_input, color=point_color, transform=data_crs, s=point_size)
    
    
    plt.show()

# %%


def map_da(da, figname, vmin=None, vmax=None, proj=ccrs.PlateCarree(), data_crs=ccrs.PlateCarree(), coast_res='10m'):
    """Map an xarray DataArray."""
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

    print(f'Min: {np.nanmin(da.values)}')
    print(f'Max: {np.nanmax(da.values)}')
    
    # Plot data
    im = ax.pcolormesh(
        da.lon, da.lat, da.values,
        transform=data_crs,
        cmap="turbo", vmin=vmin, vmax=vmax
    )
    
    # Add coastlines and borders
    ax.coastlines(resolution=coast_res, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    
    # Set map extent to match da
    ax.set_extent([
        float(da.lon.min()), float(da.lon.max()),
        float(da.lat.min()), float(da.lat.max())
    ], crs=data_crs)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(da.name)

    plt.savefig(home_dir+'/Output/'+figname, dpi=1200, bbox_inches="tight")
    plt.show()



