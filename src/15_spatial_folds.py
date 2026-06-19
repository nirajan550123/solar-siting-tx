"""
17_spatial_folds.py
Build spatial block folds for the sensitivity ladder (50/130/200/300 km).
Grid-based square blocks; blocks assigned to 5 folds with presence-balancing.
Reports presences/absences per fold at each block size so we confirm the design is viable.
Saves fold assignments to data/processed/model_table.gpkg (new cols: fold_50, fold_130, ...).
"""
import numpy as np
import geopandas as gpd
from pathlib import Path

MT = Path("data/processed/model_table.gpkg")
m = gpd.read_file(MT).to_crs(6580).reset_index(drop=True)
K = 5
SIZES_KM = [50, 130, 200, 300]
rng = np.random.default_rng(42)

xmin, ymin, xmax, ymax = m.total_bounds
x = m.geometry.x.values; y = m.geometry.y.values

def assign_folds(size_m):
    # block id from grid cell
    bx = ((x - xmin) // size_m).astype(int)
    by = ((y - ymin) // size_m).astype(int)
    block = bx * 100000 + by
    ublocks = np.unique(block)
    # allocate blocks to K folds, balancing presence count per fold
    pres_per_block = {b: int(((m["label"].values==1) & (block==b)).sum()) for b in ublocks}
    # greedy: sort blocks by presence count desc, assign each to the fold with fewest presences
    order = sorted(ublocks, key=lambda b: -pres_per_block[b])
    fold_pres = {f:0 for f in range(K)}
    block_fold = {}
    for b in order:
        f = min(fold_pres, key=fold_pres.get)
        block_fold[b] = f
        fold_pres[f] += pres_per_block[b]
    fold = np.array([block_fold[b] for b in block])
    return fold, len(ublocks)

print(f"{'size_km':>8} {'n_blocks':>9} | presences per fold (k=5)")
for s in SIZES_KM:
    fold, nb = assign_folds(s*1000)
    m[f"fold_{s}"] = fold
    pres = [int(((m["label"].values==1)&(fold==f)).sum()) for f in range(K)]
    absc = [int(((m["label"].values==0)&(fold==f)).sum()) for f in range(K)]
    print(f"{s:>8} {nb:>9} | presences {pres}  (min {min(pres)})")
    if min(pres) < 15:
        print(f"           WARNING: a fold has <15 presences at {s} km - metrics may be unstable")

m.to_file(MT, driver="GPKG")
print("\nSaved fold columns:", [c for c in m.columns if c.startswith('fold_')])
print("Primary block size for headline = 130 km (fold_130).")
