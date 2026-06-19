"""
22c_fill_grid_nulls.py
Nearest-neighbor spatial fill for grid nulls (slope 3176, lc 7).
Defensible: slope is smooth + lowest-importance feature; lc nulls are edge cells.
"""
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

grid = gpd.read_file("data/interim/grid_features.gpkg")
xy = np.c_[grid.geometry.x, grid.geometry.y]

for col in ["slope_pct","lc_majority"]:
    miss = grid[col].isna()
    n = int(miss.sum())
    if n:
        known = grid[~miss]
        tree = cKDTree(np.c_[known.geometry.x, known.geometry.y])
        _, idx = tree.query(xy[miss.values], k=1)
        grid.loc[miss, col] = known.iloc[idx][col].values
        print(f"Filled {n} nulls in {col}")

grid["lc_majority"] = grid["lc_majority"].round().astype(int)
grid.to_file("data/interim/grid_features.gpkg", driver="GPKG")
feats = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
print("Final nulls:", grid[feats].isna().sum().to_dict())
print("Rows:", len(grid))
