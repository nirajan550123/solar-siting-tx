"""
nc_05_features.py
"""
import ee, os, time, math
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import requests
from pathlib import Path
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from dotenv import load_dotenv

ee.Initialize(project=os.getenv("GEE_PROJECT", "your-gee-project-id"))
load_dotenv(); KEY=os.getenv("NREL_API_KEY")
CRS = 32119

ev = gpd.read_file("data/interim/nc_eval_distances.gpkg").to_crs(CRS).reset_index(drop=True)
# centroids for coordinate sampling (geometries are mixed polygon/point)
cent_m = ev.geometry.centroid
cent_4326 = cent_m.to_crs(4326)
lons = cent_4326.x.values; lats = cent_4326.y.values
N = len(ev); print("NC eval points:", N)

dem = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
slope = ee.Terrain.slope(dem).multiply(math.pi/180).tan().multiply(100).rename("slope_pct")
elev = dem.rename("elevation")
nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
          .filter(ee.Filter.eq("system:index","2021")).first().select("landcover").rename("lc"))
img = slope.addBands(elev).addBands(nlcd)

sl=np.full(N,np.nan); el=np.full(N,np.nan); lc=np.full(N,np.nan)
B=500
for i in range(0,N,B):
    j=min(i+B,N)
    fc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lons[k]),float(lats[k])]),{"k":int(k)}) for k in range(i,j)])
    for f in img.reduceRegions(fc, ee.Reducer.first(), scale=30).getInfo()["features"]:
        p=f["properties"]; k=p["k"]; sl[k]=p.get("slope_pct"); el[k]=p.get("elevation"); lc[k]=p.get("lc")
    print(f"  GEE {j}/{N}"); time.sleep(0.2)
ev["slope_pct"]=sl; ev["elevation"]=el; ev["lc_majority"]=lc

M=max(1,N//300); idx=np.arange(0,N,M)
print(f"GHI: {len(idx)} coarse pulls -> interpolate")
BASE="https://developer.nrel.gov/api/solar/solar_resource/v1.json"
def ghi(lo,la):
    for _ in range(3):
        try:
            r=requests.get(BASE,params={"api_key":KEY,"lat":la,"lon":lo},timeout=60)
            if r.status_code==200: return r.json()["outputs"]["avg_ghi"]["annual"]
            time.sleep(2)
        except Exception: time.sleep(2)
    return None
cx = cent_m.x.values; cy = cent_m.y.values
sx,sy,sv=[],[],[]
for c,k in enumerate(idx):
    v=ghi(lons[k],lats[k])
    if v is not None: sx.append(cx[k]); sy.append(cy[k]); sv.append(v)
    time.sleep(1.05)
    if c%50==0: print(f"  GHI {c}/{len(idx)}")
gh=griddata((sx,sy),sv,(cx,cy),method="linear")
nan=np.isnan(gh)
if nan.any(): gh[nan]=griddata((sx,sy),sv,(cx[nan],cy[nan]),method="nearest")
ev["ghi_nsrdb"]=gh

for col in ["slope_pct","elevation","lc_majority"]:
    miss=ev[col].isna()
    if miss.any():
        kn=ev[~miss]; knc=kn.geometry.centroid
        t=cKDTree(np.c_[knc.x,knc.y])
        mc=ev[miss].geometry.centroid
        _,ii=t.query(np.c_[mc.x,mc.y],k=1)
        ev.loc[miss,col]=kn.iloc[ii][col].values
        print(f"filled {int(miss.sum())} {col} nulls")
ev["lc_majority"]=ev["lc_majority"].round().astype(int)

ev.to_file("data/processed/nc_eval_table.gpkg", driver="GPKG")
FEATS=["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
print("\nNC eval table:", len(ev), "rows | label balance:", ev["label"].value_counts().to_dict())
print("Nulls:", ev[FEATS].isna().sum().to_dict())
print("GHI range:", round(ev["ghi_nsrdb"].min(),2),"-",round(ev["ghi_nsrdb"].max(),2))
