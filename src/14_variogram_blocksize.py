"""
16_variogram_full.py
Re-estimate spatial autocorrelation range on the FULL continuous predictor set
(now including short-range infrastructure-distance features) to set a usable block size.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import skgstat as skg

m = gpd.read_file("data/processed/model_table.gpkg")
preds = ["ghi_nsrdb","slope_pct","elevation","dist_transmission","dist_substation","dist_road"]
coords = np.c_[m.geometry.x, m.geometry.y]

print("Spatial autocorrelation range per predictor (raw, spherical variogram):\n")
ranges = {}
for p in preds:
    vals = m[p].values.astype(float)
    V = skg.Variogram(coords, vals, n_lags=20, model="spherical", maxlag="median")
    rng = V.parameters[0]
    ranges[p] = rng
    print(f"  {p:18s} {rng/1000:8.1f} km")

vals = np.array(list(ranges.values()))
print(f"\nMedian range: {np.median(vals)/1000:.1f} km")
print(f"Mean range:   {np.mean(vals)/1000:.1f} km")
print(f"Min range:    {np.min(vals)/1000:.1f} km  (shortest-range predictor)")

med = np.median(vals)/1000
print(f"\n-> candidate block size: {med:.0f} km")
print(f"-> TX is ~1300 km across; {med:.0f} km blocks give ~{(1300/med)**2:.0f} blocks (need >=5 well-populated for 5 folds)")
pd.DataFrame({"predictor":list(ranges.keys()),
             "range_km":[r/1000 for r in ranges.values()]}).to_csv(
             "data/interim/variogram_ranges_full.csv", index=False)
print("Saved full ranges to data/interim/variogram_ranges_full.csv")
