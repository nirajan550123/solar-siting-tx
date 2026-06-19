"""
nc_04_distances.py
Compute NC distance features (transmission, substation, road) for NC presences + pseudo-absences.
EPSG:32119 (NC metres). Builds the NC evaluation table's distance columns and compares
presence vs pseudo-absence (preview of transfer signal).
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

CRS = 32119
pos = gpd.read_file("data/processed/positives_nc.gpkg").to_crs(CRS)
pa  = gpd.read_file("data/processed/pseudo_absences_nc.gpkg").to_crs(CRS)
trans = gpd.read_file("data/raw/hifld_transmission_nc.gpkg").to_crs(CRS)
subs  = gpd.read_file("data/raw/hifld_substations_nc.gpkg").to_crs(CRS)
roads = gpd.read_file("data/raw/tl_2024_37_prisecroads/tl_2024_37_prisecroads.shp").to_crs(CRS)
print(f"infra: transmission {len(trans)} | substations {len(subs)} | roads {len(roads)}")

# build combined eval table: presences (label 1, footprint polygons) + PAs (label 0, points)
pos_e = pos[["geometry"]].copy(); pos_e["label"]=1
pa_e  = pa[["geometry"]].copy();  pa_e["label"]=0
# for presences measure from polygon; PAs from point
def nd(geoms, target):
    gg = gpd.GeoDataFrame(geometry=geoms, crs=CRS)
    j = gpd.sjoin_nearest(gg[["geometry"]], target[["geometry"]], how="left", distance_col="d")
    return j.groupby(j.index)["d"].min().reindex(range(len(gg))).values

eval_tbl = pd.concat([pos_e, pa_e], ignore_index=True)
eval_g = gpd.GeoDataFrame(eval_tbl, geometry="geometry", crs=CRS)
eval_g["dist_transmission"] = nd(eval_g.geometry, trans)
eval_g["dist_substation"]   = nd(eval_g.geometry, subs)
eval_g["dist_road"]         = nd(eval_g.geometry, roads)

eval_g.to_file("data/interim/nc_eval_distances.gpkg", driver="GPKG")
print("\nNulls:", eval_g[["dist_transmission","dist_substation","dist_road"]].isna().sum().to_dict())

print("\n=== NC presence vs pseudo-absence (median m) ===")
for c in ["dist_transmission","dist_substation","dist_road"]:
    p = eval_g[eval_g.label==1][c].median(); a = eval_g[eval_g.label==0][c].median()
    print(f"  {c:18s} presence {p:8.0f}   pseudo-abs {a:8.0f}")
print("\n(TX was: transmission 55 vs 6305 | substation 2456 vs 9272 | road 1740 vs 5121)")
