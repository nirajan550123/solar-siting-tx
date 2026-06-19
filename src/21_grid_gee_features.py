"""
22b_grid_gee_ghi.py
Extract GEE features (slope, elevation, land cover) for all grid cells in batches,
and GHI via a coarse API subsample + spatial interpolation (GHI is smooth, low-importance).
Output: data/interim/grid_features.gpkg  (grid + all 7 features)
"""
import ee, os, time, json, math
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import requests
from pathlib import Path
from scipy.interpolate import griddata
from dotenv import load_dotenv

ee.Initialize(project=os.getenv("GEE_PROJECT", "your-gee-project-id"))
load_dotenv(); KEY=os.getenv("NREL_API_KEY")

grid = gpd.read_file("data/interim/grid_distances.gpkg")  # EPSG:6580
g4326 = grid.to_crs(4326)
lons = g4326.geometry.x.values; lats = g4326.geometry.y.values
N = len(grid); print("Grid cells:", N)

# ---------- GEE: slope, elevation, land cover (batched point sampling) ----------
dem = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
slope = ee.Terrain.slope(dem).multiply(math.pi/180).tan().multiply(100).rename("slope_pct")
elev  = dem.rename("elevation")
nlcd  = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
           .filter(ee.Filter.eq("system:index","2021")).first().select("landcover").rename("lc"))
img = slope.addBands(elev).addBands(nlcd)

slope_v=np.full(N,np.nan); elev_v=np.full(N,np.nan); lc_v=np.full(N,np.nan)
BATCH=500
print("Sampling GEE features in batches of", BATCH)
for i in range(0, N, BATCH):
    j=min(i+BATCH,N)
    feats=[ee.Feature(ee.Geometry.Point([float(lons[k]),float(lats[k])]),{"k":int(k)}) for k in range(i,j)]
    fc=ee.FeatureCollection(feats)
    samp=img.reduceRegions(fc, ee.Reducer.first(), scale=30).getInfo()["features"]
    for f in samp:
        p=f["properties"]; k=p["k"]
        slope_v[k]=p.get("slope_pct"); elev_v[k]=p.get("elevation"); lc_v[k]=p.get("lc")
    if (i//BATCH)%5==0: print(f"  {j}/{N}")
    time.sleep(0.2)

grid["slope_pct"]=slope_v; grid["elevation"]=elev_v; grid["lc_majority"]=lc_v
print("GEE done. nulls:", grid[["slope_pct","elevation","lc_majority"]].isna().sum().to_dict())

# ---------- GHI: coarse API subsample + interpolate ----------
# sample every Mth cell to keep API calls ~ a few hundred
M = max(1, N // 400)
idx = np.arange(0, N, M)
print(f"\nGHI: pulling {len(idx)} coarse samples then interpolating to {N} cells")
BASE="https://developer.nrel.gov/api/solar/solar_resource/v1.json"
def ghi(lon,lat):
    for a in range(3):
        try:
            r=requests.get(BASE,params={"api_key":KEY,"lat":lat,"lon":lon},timeout=60)
            if r.status_code==200: return r.json()["outputs"]["avg_ghi"]["annual"]
            time.sleep(2)
        except Exception: time.sleep(2)
    return None
sx,sy,sv=[],[],[]
for c,k in enumerate(idx):
    v=ghi(lons[k],lats[k])
    if v is not None:
        sx.append(grid.geometry.x.iloc[k]); sy.append(grid.geometry.y.iloc[k]); sv.append(v)
    time.sleep(1.05)
    if c%50==0: print(f"  GHI {c}/{len(idx)}")

gx=grid.geometry.x.values; gy=grid.geometry.y.values
ghi_grid=griddata((sx,sy),sv,(gx,gy),method="linear")
# fill edge NaNs with nearest
nanmask=np.isnan(ghi_grid)
if nanmask.any():
    ghi_grid[nanmask]=griddata((sx,sy),sv,(gx[nanmask],gy[nanmask]),method="nearest")
grid["ghi_nsrdb"]=ghi_grid
print("GHI interpolated. range:", round(np.nanmin(ghi_grid),2), "-", round(np.nanmax(ghi_grid),2))

grid.to_file("data/interim/grid_features.gpkg", driver="GPKG")
print("\nSaved grid + 7 features. Final nulls:")
print(grid[["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]].isna().sum().to_dict())
