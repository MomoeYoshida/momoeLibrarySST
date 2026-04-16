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
from datetime import datetime
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    return datetime.strptime(julian_str, "%Y_%j")


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
def extract_pixel_timeseries(ds, lat_input, lon_input):
    """Extract time series at closest pixel to given lat/lon."""
    
    # Extract both variables at nearest pixel.
    pixel_ds = ds.sel(lat=lat_input, lon=lon_input, method="nearest")

    # Extract variables.
    ts_sst = pixel_ds["sst"]
    ts_cm  = pixel_ds["correlation_map"]
    
    # Print actual selected location
    pixel_lat = float(pixel_ds.lat.values)
    pixel_lon = float(pixel_ds.lon.values)
    
    print(f"Requested: lat={lat_input}, lon={lon_input}")
    print(f"Closest pixel: lat={pixel_lat:.3f}, lon={pixel_lon:.3f}") # display artifact

    mark_pixel_point(pixel_lat, lat_input, pixel_lon, lon_input)
    plot_ts(ts_sst, ts_cm)
    
    return pixel_ds


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
def plot_ts(ts_sst, ts_cm, transparent=False):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    if transparent:
        # Transparent figure background
        fig.patch.set_alpha(0)
        ax1.set_facecolor((0, 0, 0, 0))
    
    # --- Left axis (SST) ---
    ax1.plot(ts_sst.time, ts_sst, color="tab:blue", marker="o", label="SST")
    ax1.set_ylabel("SST (°C)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    # --- Right axis (correlation_map) ---
    ax2 = ax1.twinx()
    cm_min, cm_max = 8, 32
    ax2.plot(ts_cm.time, ts_cm, color="tab:red", marker="s", label="Correlation Map")
    ax2.set_ylabel("Correlation Map", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.set_ylim(cm_min, cm_max)
    ax2.set_yticks(list(range(cm_min, cm_max+1, 4)))
    
    # --- X axis ---
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.tick_params(axis='x', rotation=60)
    
    # # --- Combine legends ---
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    plt.tight_layout()
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



