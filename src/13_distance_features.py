"""
15_distance_features.py
Compute distance (m) from each point to nearest transmission line, substation, and road.
All in EPSG:6580 (true metres). Presences measured from footprint polygon; PAs from point.
Adds dist_transmission, dist_substation, dist_road to model_table.
"""
import numpy as np
import geopandas as gpd
from pathlib import Path

MT  = Path("data/processed/model_table.gpkg")
POS = Path("data/processed/positives_tx.gpkg")
TRANS = Path("data/raw/hifld_transmission_tx.gpkg")
SUBS  = Path("data/raw/hifld_substations_tx.gpkg")
ROADS = Path("data/raw/tl_2024_48_prisecroads/tl_2024_48_prisecroads.shp")

CRS = 6580

m = gpd.read_file(MT).to_crs(CRS).reset_index(drop=True)
pos = gpd.read_file(POS).to_crs(CRS)

print("Loading infrastructure...")
trans = gpd.read_file(TRANS).to_crs(CRS)
subs  = gpd.read_file(SUBS).to_crs(CRS)
roads = gpd.read_file(ROADS).to_crs(CRS)
print(f"  transmission: {len(trans)} | substations: {len(subs)} | roads: {len(roads)}")

# build geometry to measure FROM:
#   presences (label 1) -> footprint polygon (match by order: first 176 rows are presences)
#   pseudo-absences      -> the point itself
# model_table stores PA points and presence centroids; for presences use the actual polygons.
geom_from = m.geometry.copy()
pres_mask = (m["label"] == 1).values
assert pres_mask.sum() == len(pos), "presence count mismatch"
geom_from.loc[pres_mask] = pos.geometry.values   # use polygons for presences
mfrom = gpd.GeoDataFrame(m.drop(columns="geometry"), geometry=geom_from, crs=CRS)

def nearest_dist(points_gdf, target_gdf, name):
    j = gpd.sjoin_nearest(points_gdf[["geometry"]], target_gdf[["geometry"]],
                          how="left", distance_col="d")
    # sjoin_nearest can duplicate rows on ties; keep min per original index
    d = j.groupby(j.index)["d"].min()
    print(f"  {name}: min {d.min():.0f}  median {d.median():.0f}  max {d.max():.0f} m")
    return d.reindex(points_gdf.index).values

print("\nComputing distances...")
m["dist_transmission"] = nearest_dist(mfrom, trans, "transmission")
m["dist_substation"]   = nearest_dist(mfrom, subs,  "substation")
m["dist_road"]         = nearest_dist(mfrom, roads, "road")

# nulls?
newcols = ["dist_transmission","dist_substation","dist_road"]
print("\nNull counts:", m[newcols].isna().sum().to_dict())

# save geometry as the model_table points (not the polygons) -> restore original geom
m = gpd.GeoDataFrame(m, geometry=m.geometry, crs=CRS)
m.to_file(MT, driver="GPKG")
print("\nSaved. Model table now has", len(m.columns)-1, "non-geometry columns:")
print(list(c for c in m.columns if c != "geometry"))
print("\nDistance feature summary (m):")
print(m[newcols].describe().round(0).to_string())
