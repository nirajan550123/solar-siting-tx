"""
10_merge_features.py
The GHI csv was written one row per feature row in the SAME order, so we join positionally.
Output: data/processed/model_table.gpkg
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree

FEAT = Path("data/interim/features_gee.gpkg")
GHI  = Path("data/interim/nsrdb_ghi.csv")
OUT  = Path("data/processed/model_table.gpkg")

gdf = gpd.read_file(FEAT).reset_index(drop=True)
ghi = pd.read_csv(GHI).reset_index(drop=True)

assert len(gdf) == len(ghi), f"Length mismatch: {len(gdf)} vs {len(ghi)}"
# sanity: labels must line up if positional alignment is valid
assert (gdf["label"].values == ghi["label"].values).all(), "Label order mismatch - not positional!"

# drop proxy ghi, attach NSRDB ghi positionally
gdf = gdf.drop(columns=["ghi"])
gdf["ghi_nsrdb"] = ghi["ghi_nsrdb"].values

def nn_fill(frame, col):
    known = frame[frame[col].notna()]; miss = frame[frame[col].isna()]
    if len(miss) == 0: return frame, 0
    tree = cKDTree(np.c_[known.geometry.x, known.geometry.y])
    _, idx = tree.query(np.c_[miss.geometry.x, miss.geometry.y], k=1)
    frame.loc[miss.index, col] = known.iloc[idx][col].values
    return frame, len(miss)

for col in ["ghi_nsrdb", "slope_pct", "elevation", "lc_majority"]:
    gdf, n = nn_fill(gdf, col)
    print(f"Backfilled {n} nulls in {col}")

OUT.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUT, driver="GPKG")

print("\nFinal table:", len(gdf), "rows")
print("Null counts:")
print(gdf[["ghi_nsrdb","slope_pct","elevation","lc_majority"]].isna().sum().to_string())
print("\nClass balance (label):")
print(gdf["label"].value_counts().to_string())
print("\nGHI summary:")
print(gdf["ghi_nsrdb"].describe().round(3).to_string())
