"""
99_audit.py
Full-project integrity check. Verifies every data artifact exists, has the right shape,
correct CRS, no nulls where there shouldn't be, and internal consistency across files.
Run anytime to confirm the pipeline is sound.
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd

def line(): print("-"*64)

print("="*64); print("PROJECT INTEGRITY AUDIT"); print("="*64)

# --- 1. Files exist ---
expected = {
    "raw USPVDB": "data/raw/uspvdbGeoJSON/uspvdb_v4_0_20260414.geojson",
    "TX states shp": "data/raw/tl_2024_us_state/tl_2024_us_state.shp",
    "positives": "data/processed/positives_tx.gpkg",
    "pseudo-absences": "data/processed/pseudo_absences.gpkg",
    "TX boundary": "data/processed/tx_boundary.gpkg",
    "GEE features": "data/interim/features_gee.gpkg",
    "NSRDB GHI csv": "data/interim/nsrdb_ghi.csv",
    "MODEL TABLE": "data/processed/model_table.gpkg",
}
line(); print("1. FILE EXISTENCE")
for name, p in expected.items():
    print(f"  [{'OK' if Path(p).exists() else 'MISSING'}] {name}: {p}")

# --- 2. Positives ---
line(); print("2. POSITIVES (positives_tx.gpkg)")
pos = gpd.read_file("data/processed/positives_tx.gpkg")
print(f"  rows: {len(pos)} (expect 176) | CRS: {pos.crs.to_epsg()} (expect 6580)")
print(f"  capacity AC min: {pos['p_cap_ac'].min()} (expect >=1)")
print(f"  all p_state==TX: {(pos['p_state']=='TX').all()} | all ground: {(pos['p_sys_type']=='ground').all()}")

# --- 3. Pseudo-absences ---
line(); print("3. PSEUDO-ABSENCES (pseudo_absences.gpkg)")
pa = gpd.read_file("data/processed/pseudo_absences.gpkg")
print(f"  rows: {len(pa)} (expect 1760) | CRS: {pa.crs.to_epsg()} (expect 6580)")
print(f"  replicates: {sorted(pa['replicate'].unique())} (expect 0..9)")
print(f"  per replicate: {pa['replicate'].value_counts().sort_index().tolist()} (expect ten 176s)")

# --- 4. Model table (the key one) ---
line(); print("4. MODEL TABLE (model_table.gpkg)")
m = gpd.read_file("data/processed/model_table.gpkg")
print(f"  rows: {len(m)} (expect 1936) | CRS: {m.crs.to_epsg()} (expect 6580)")
print(f"  columns: {list(m.columns)}")
print(f"  label balance: {m['label'].value_counts().sort_index().to_dict()} (expect 0:1760, 1:176)")
feats = ["ghi_nsrdb","slope_pct","elevation","lc_majority"]
print(f"  null counts: {m[feats].isna().sum().to_dict()} (expect all 0)")
print("  feature ranges:")
for f in feats:
    print(f"    {f}: {round(m[f].min(),2)} .. {round(m[f].max(),2)}")
print(f"  lc_majority classes present: {sorted(m['lc_majority'].dropna().unique().tolist())}")

# --- 5. Cross-file consistency ---
line(); print("5. CONSISTENCY CHECKS")
print(f"  positives + pseudo-absences = {len(pos)+len(pa)} (model table = {len(m)}) -> {'MATCH' if len(pos)+len(pa)==len(m) else 'MISMATCH'}")
print(f"  presences in model table = {(m['label']==1).sum()} (positives file = {len(pos)}) -> {'MATCH' if (m['label']==1).sum()==len(pos) else 'MISMATCH'}")

# --- 6. Slope mask sanity: any presence above 5%? (informational) ---
line(); print("6. INFORMATIONAL")
hi_slope = (m.loc[m['label']==1,'slope_pct']>5).sum()
print(f"  presences with slope>5% (built on steeper land than mask allows): {hi_slope}")
print(f"  (informational - real facilities CAN exceed the PA mask threshold; not an error)")

line(); print("AUDIT COMPLETE")
