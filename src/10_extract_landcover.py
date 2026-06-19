"""
11_fix_landcover.py
Re-extract NLCD land cover as a CLEAN INTEGER MODE (most common class) per geometry,
replacing the broken fractional lc_majority. Re-merges positionally into model_table.
"""
import ee, math
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

ee.Initialize(project=os.getenv("GEE_PROJECT", "your-gee-project-id"))

POS = Path("data/processed/positives_tx.gpkg")
PA  = Path("data/processed/pseudo_absences.gpkg")
MT  = Path("data/processed/model_table.gpkg")

# NLCD 2021, integer band, NO resampling (sample at native 30 m)
nlcd = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
          .filter(ee.Filter.eq("system:index","2021")).first()
          .select("landcover").rename("lc"))

# Rebuild the SAME geometries used in feature extraction:
# presences = footprint polygons; pseudo-absences = circles of eq-radius (median footprint)
pos = gpd.read_file(POS)
median_area = pos.geometry.area.median()
eq_radius = math.sqrt(median_area / math.pi)
pos_4326 = pos.to_crs(4326)

pa = gpd.read_file(PA)
pa_buf = pa.copy(); pa_buf["geometry"] = pa.geometry.buffer(eq_radius)
pa_4326 = pa_buf.to_crs(4326)

def mode_lc(gdf_4326, label):
    feats = []
    for idx, row in gdf_4326.iterrows():
        feats.append(ee.Feature(ee.Geometry(row.geometry.__geo_interface__),
                                {"rid": int(idx), "label": label}))
    fc = ee.FeatureCollection(feats)
    # mode reducer at native 30 m -> dominant integer class
    red = nlcd.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
    out = []
    for f in red.getInfo()["features"]:
        p = f["properties"]
        m = p.get("mode")
        out.append({"rid": p["rid"], "label": p["label"],
                    "lc_clean": int(round(m)) if m is not None else None})
    return pd.DataFrame(out)

print("Re-extracting land cover (mode, integer)...")
df_pos = mode_lc(pos_4326, 1); print("  presences:", len(df_pos))
df_pa  = mode_lc(pa_4326, 0);  print("  pseudo-absences:", len(df_pa))
lc = pd.concat([df_pos, df_pa], ignore_index=True)

# --- merge positionally into model_table (same order: presences then PAs) ---
m = gpd.read_file(MT).reset_index(drop=True)
assert len(m) == len(lc), f"length mismatch {len(m)} vs {len(lc)}"
assert (m["label"].values == lc["label"].values).all(), "label order mismatch"

m = m.drop(columns=["lc_majority"])
m["lc_majority"] = lc["lc_clean"].values

# nearest-neighbor fill if any null
from scipy.spatial import cKDTree
miss = m[m["lc_majority"].isna()]
if len(miss):
    known = m[m["lc_majority"].notna()]
    tree = cKDTree(np.c_[known.geometry.x, known.geometry.y])
    _, idx = tree.query(np.c_[miss.geometry.x, miss.geometry.y], k=1)
    m.loc[miss.index, "lc_majority"] = known.iloc[idx]["lc_majority"].values
    print("Filled", len(miss), "lc nulls by nearest neighbor")

m["lc_majority"] = m["lc_majority"].astype(int)
m.to_file(MT, driver="GPKG")

print("\nClean lc_majority classes:", sorted(m["lc_majority"].unique().tolist()))
print("Class distribution:")
print(m["lc_majority"].value_counts().sort_index().to_string())
print("\nNulls:", m["lc_majority"].isna().sum(), "| rows:", len(m))
