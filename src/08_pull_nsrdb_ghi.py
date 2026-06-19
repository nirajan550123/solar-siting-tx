"""
09_pull_nsrdb_ghi.py
Long-term annual-average GHI per point. Deduped, resumable.
Output: data/interim/nsrdb_ghi.csv  (rid, label, ghi_nsrdb in kWh/m2/day)
"""
import os, time, json
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("NREL_API_KEY"); EMAIL = os.getenv("NREL_API_EMAIL")
FEAT = Path("data/interim/features_gee.gpkg")
CACHE = Path("data/interim/nsrdb_srapi_cache.json")
OUT = Path("data/interim/nsrdb_ghi.csv")
BASE = "https://developer.nrel.gov/api/solar/solar_resource/v1.json"

gdf = gpd.read_file(FEAT).to_crs(4326)
gdf["lon"] = gdf.geometry.x; gdf["lat"] = gdf.geometry.y
gdf["cell"] = (gdf["lon"].map(lambda v: round(v/0.04)*0.04).astype(str)
               + "_" + gdf["lat"].map(lambda v: round(v/0.04)*0.04).astype(str))
uniq = gdf.drop_duplicates("cell")[["cell","lon","lat"]].reset_index(drop=True)
print("Unique cells to query:", len(uniq))

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def pull(lon, lat):
    params = {"api_key": KEY, "lat": lat, "lon": lon}
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=60)
            if r.status_code == 200:
                j = r.json()
                return j["outputs"]["avg_ghi"]["annual"]
            elif r.status_code == 429:
                time.sleep(4*(attempt+1)); continue
            else:
                print("  HTTP", r.status_code, r.text[:120]); time.sleep(2)
        except Exception as e:
            print("  retry", attempt, e); time.sleep(2)
    return None

for i, row in uniq.iterrows():
    if row["cell"] in cache: continue
    cache[row["cell"]] = pull(row["lon"], row["lat"])
    CACHE.write_text(json.dumps(cache))
    time.sleep(1.05)
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(uniq)}")

gdf["ghi_nsrdb"] = gdf["cell"].map(cache)
out = gdf[["rid","label","ghi_nsrdb"]].copy()
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print("\nSaved", len(out), "rows to", OUT)
print("GHI nulls:", out["ghi_nsrdb"].isna().sum())
print("GHI range (kWh/m2/day):", round(out["ghi_nsrdb"].min(),2), "-", round(out["ghi_nsrdb"].max(),2))
