"""
22a_grid_distances.py
Build a 5 km prediction grid over Texas and compute the three infrastructure-distance features
locally (fast). GEE features + GHI come in Part B.
Output: data/interim/grid_distances.gpkg
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

CRS = 6580
CELL = 5000  # 5 km
TX = gpd.read_file("data/processed/tx_boundary.gpkg").to_crs(CRS)
tx_geom = TX.geometry.iloc[0]
xmin, ymin, xmax, ymax = TX.total_bounds

# regular grid of cell centroids, keep those inside TX
xs = np.arange(xmin+CELL/2, xmax, CELL)
ys = np.arange(ymin+CELL/2, ymax, CELL)
pts = [Point(x,y) for x in xs for y in ys]
grid = gpd.GeoDataFrame(geometry=pts, crs=CRS)
grid = grid[grid.within(tx_geom)].reset_index(drop=True)
print(f"Grid cells inside Texas (5 km): {len(grid)}")

# distance features (same sources as training)
print("Loading infrastructure + computing distances...")
trans = gpd.read_file("data/raw/hifld_transmission_tx.gpkg").to_crs(CRS)
subs  = gpd.read_file("data/raw/hifld_substations_tx.gpkg").to_crs(CRS)
roads = gpd.read_file("data/raw/tl_2024_48_prisecroads/tl_2024_48_prisecroads.shp").to_crs(CRS)

def nd(points, target, name):
    j = gpd.sjoin_nearest(points[["geometry"]], target[["geometry"]], how="left", distance_col="d")
    d = j.groupby(j.index)["d"].min().reindex(points.index).values
    print(f"  {name}: median {np.median(d):.0f} m")
    return d

grid["dist_transmission"] = nd(grid, trans, "transmission")
grid["dist_substation"]   = nd(grid, subs,  "substation")
grid["dist_road"]         = nd(grid, roads, "road")

Path("data/interim").mkdir(parents=True, exist_ok=True)
grid.to_file("data/interim/grid_distances.gpkg", driver="GPKG")
print(f"\nSaved {len(grid)} grid cells with distance features to data/interim/grid_distances.gpkg")
print("Nulls:", grid[["dist_transmission","dist_substation","dist_road"]].isna().sum().to_dict())
