"""
nc_02_mask_pseudoabsences.py
NC developable mask + 753 pseudo-absences (1 km buffer), same rules as TX.
Single set (evaluation, not training). Saves NC boundary, mask-drawn PAs.
"""
import ee, math
import os
import geopandas as gpd
import pandas as pd
from pathlib import Path

ee.Initialize(project=os.getenv("GEE_PROJECT", "your-gee-project-id"))
NC_CRS = 32119
EXCLUDED = [11,12,23,24,41,42,43,90,95]
BUFFER_M = 1000
N_PA = 753

# NC boundary from national TIGER file
states = gpd.read_file("data/raw/tl_2024_us_state/tl_2024_us_state.shp")
nc_bnd = states[states["STUSPS"]=="NC"].to_crs(NC_CRS)
nc_bnd.to_file("data/processed/nc_boundary.gpkg", driver="GPKG")
print("NC area (sq km):", round(nc_bnd.geometry.area.iloc[0]/1e6,0))

# GEE NC geometry
nc_ee = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq("STUSPS","NC"))
nc_geom = nc_ee.geometry()

# developable mask
dem = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
slope_pct = ee.Terrain.slope(dem).multiply(math.pi/180).tan().multiply(100)
slope_ok = slope_pct.lte(5)
nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
          .filter(ee.Filter.eq("system:index","2021")).first().select("landcover"))
lc_ok = nlcd.remap(ee.List(EXCLUDED), ee.List.repeat(0,len(EXCLUDED)),1)
developable = slope_ok.And(lc_ok)

# 1 km buffer around NC facilities
pos = gpd.read_file("data/processed/positives_nc.gpkg").to_crs(4326)
fac_pts = [ee.Feature(ee.Geometry.Point([float(g.centroid.x),float(g.centroid.y)])) for g in pos.geometry]
fac_fc = ee.FeatureCollection(fac_pts)
dist = fac_fc.distance(searchRadius=5000).unmask(9999)
allowed = developable.And(dist.gte(BUFFER_M)).selfMask().rename("allowed").clip(nc_geom)

# draw PAs (oversample for footprint filtering later)
samp = allowed.stratifiedSample(numPoints=int(N_PA*1.3), classBand="allowed",
                                region=nc_geom, scale=300, seed=7, geometries=True)
info = samp.getInfo()["features"]
rows = [{"lon":f["geometry"]["coordinates"][0],"lat":f["geometry"]["coordinates"][1]} for f in info]
df = pd.DataFrame(rows)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,df.lat), crs=4326).to_crs(NC_CRS)
gdf = gdf.iloc[:N_PA].reset_index(drop=True)  # take what we need
gdf.to_file("data/processed/pseudo_absences_nc.gpkg", driver="GPKG")
print(f"Drew {len(gdf)} NC pseudo-absences (target {N_PA})")
