"""
nc_09_transfer_stability.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import warnings; warnings.filterwarnings("ignore")

ee.Initialize(project=os.getenv("GEE_PROJECT", "your-gee-project-id"))
load_dotenv(); KEY=os.getenv("NREL_API_KEY")
NC_CRS=32119; EXCLUDED=[11,12,23,24,41,42,43,90,95]; BUFFER=1000
FEATS=["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
N_REP=10; N_PA=753

pos = gpd.read_file("data/processed/positives_nc.gpkg").to_crs(NC_CRS)
nc_eval = gpd.read_file("data/processed/nc_eval_table.gpkg")
pres_feat = nc_eval[nc_eval.label==1].copy()

nc_ee = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq("STUSPS","NC"))
nc_geom = nc_ee.geometry()
dem = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
slope_pct = ee.Terrain.slope(dem).multiply(math.pi/180).tan().multiply(100)
nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq("system:index","2021")).first().select("landcover"))
lc_ok = nlcd.remap(ee.List(EXCLUDED), ee.List.repeat(0,len(EXCLUDED)),1)
pos4326=pos.to_crs(4326)
facfc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(g.centroid.x),float(g.centroid.y)])) for g in pos4326.geometry])
dist=facfc.distance(searchRadius=5000).unmask(9999)
allowed=slope_pct.lte(5).And(lc_ok).And(dist.gte(BUFFER)).selfMask().rename("a").clip(nc_geom)

# DRAW PER REPLICATE to stay under 5000-element cap
parts=[]
for rep in range(N_REP):
    samp=allowed.stratifiedSample(numPoints=int(N_PA*1.2),classBand="a",region=nc_geom,scale=300,seed=100+rep,geometries=True)
    info=samp.getInfo()["features"]
    rows=[{"lon":f["geometry"]["coordinates"][0],"lat":f["geometry"]["coordinates"][1],"rep":rep} for f in info][:N_PA]
    parts.append(pd.DataFrame(rows))
    print(f"  drew replicate {rep}: {len(rows)} PAs")
alldf=pd.concat(parts,ignore_index=True)
allpa=gpd.GeoDataFrame(alldf,geometry=gpd.points_from_xy(alldf.lon,alldf.lat),crs=4326).to_crs(NC_CRS).reset_index(drop=True)
print(f"Total PAs: {len(allpa)}")

trans=gpd.read_file("data/raw/hifld_transmission_nc.gpkg").to_crs(NC_CRS)
subs=gpd.read_file("data/raw/hifld_substations_nc.gpkg").to_crs(NC_CRS)
roads=gpd.read_file("data/raw/tl_2024_37_prisecroads/tl_2024_37_prisecroads.shp").to_crs(NC_CRS)
def nd(g,t):
    j=gpd.sjoin_nearest(g[["geometry"]],t[["geometry"]],how="left",distance_col="d")
    return j.groupby(j.index)["d"].min().reindex(range(len(g))).values
allpa["dist_transmission"]=nd(allpa,trans); allpa["dist_substation"]=nd(allpa,subs); allpa["dist_road"]=nd(allpa,roads)

a4326=allpa.to_crs(4326); lons=a4326.geometry.x.values; lats=a4326.geometry.y.values; Np=len(allpa)
img=slope_pct.rename("slope_pct").addBands(dem.rename("elevation")).addBands(nlcd.rename("lc"))
sl=np.full(Np,np.nan); el=np.full(Np,np.nan); lc=np.full(Np,np.nan)
for i in range(0,Np,500):
    j=min(i+500,Np)
    fc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lons[k]),float(lats[k])]),{"k":int(k)}) for k in range(i,j)])
    for f in img.reduceRegions(fc,ee.Reducer.first(),scale=30).getInfo()["features"]:
        p=f["properties"];k=p["k"];sl[k]=p.get("slope_pct");el[k]=p.get("elevation");lc[k]=p.get("lc")
    print(f"  GEE {j}/{Np}"); time.sleep(0.2)
allpa["slope_pct"]=sl; allpa["elevation"]=el; allpa["lc_majority"]=lc

M=max(1,Np//300); idx=np.arange(0,Np,M)
BASE="https://developer.nrel.gov/api/solar/solar_resource/v1.json"
def ghi(lo,la):
    for _ in range(3):
        try:
            r=requests.get(BASE,params={"api_key":KEY,"lat":la,"lon":lo},timeout=60)
            if r.status_code==200: return r.json()["outputs"]["avg_ghi"]["annual"]
            time.sleep(2)
        except: time.sleep(2)
    return None
sx,sy,sv=[],[],[]; cx=allpa.geometry.x.values; cy=allpa.geometry.y.values
for c,k in enumerate(idx):
    v=ghi(lons[k],lats[k])
    if v is not None: sx.append(cx[k]);sy.append(cy[k]);sv.append(v)
    time.sleep(1.05)
    if c%50==0: print(f"  GHI {c}/{len(idx)}")
gh=griddata((sx,sy),sv,(cx,cy),method="linear"); nan=np.isnan(gh)
if nan.any(): gh[nan]=griddata((sx,sy),sv,(cx[nan],cy[nan]),method="nearest")
allpa["ghi_nsrdb"]=gh
for col in ["slope_pct","elevation","lc_majority"]:
    m=allpa[col].isna()
    if m.any():
        kn=allpa[~m]; t=cKDTree(np.c_[kn.geometry.x,kn.geometry.y]); _,ii=t.query(np.c_[allpa[m].geometry.x,allpa[m].geometry.y],k=1)
        allpa.loc[m,col]=kn.iloc[ii][col].values
allpa["lc_majority"]=allpa["lc_majority"].round().astype(int)

tx=gpd.read_file("data/processed/model_table.gpkg")
rf=RandomForestClassifier(n_estimators=800,class_weight="balanced_subsample",random_state=0,n_jobs=-1).fit(tx[FEATS].values,tx["label"].values)
aucs=[]
for rep in range(N_REP):
    paset=allpa[allpa["rep"]==rep]
    if len(paset)<10: continue
    ev=pd.concat([pres_feat[FEATS].assign(label=1), paset[FEATS].assign(label=0)])
    p=rf.predict_proba(ev[FEATS].values)[:,1]
    aucs.append(roc_auc_score(ev["label"].values,p))
aucs=np.array(aucs)
print(f"\n=== NC TRANSFER STABILITY ({len(aucs)} draws) ===")
print(f"  transfer ROC-AUC: mean {aucs.mean():.3f}  SD {aucs.std():.3f}  min {aucs.min():.3f}  max {aucs.max():.3f}")
print(f"  original single-draw was 0.772")
