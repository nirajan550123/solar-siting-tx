"""
30a_ecoregion_join.py
"""
import os, zipfile, io, urllib.request, ssl
import geopandas as gpd
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/ecoregions_tx"); RAW.mkdir(parents=True, exist_ok=True)
TARGET_CRS = 6580
URL = "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/tx/tx_eco_l3.zip"

if not any(RAW.glob("*.shp")):
    print("Downloading TX L3 ecoregions from EPA S3...")
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    req = urllib.request.Request(URL, headers={"User-Agent":"Mozilla/5.0"})
    data = urllib.request.urlopen(req, context=ctx, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(RAW)
    print("  extracted:", [p.name for p in RAW.glob('*.shp')])

shp = list(RAW.glob("*.shp"))[0]
eco = gpd.read_file(shp).to_crs(TARGET_CRS)
print("\nEcoregion columns:", list(eco.columns))
namecol = next((c for c in ["US_L3NAME","L3_KEY","NA_L3NAME","US_L3CODE"] if c in eco.columns), eco.columns[0])
print(f"Using ecoregion name column: {namecol}")
eco = eco[[namecol,"geometry"]].rename(columns={namecol:"ecoregion"})
eco = eco.dissolve("ecoregion").reset_index()
print(f"Distinct L3 ecoregions in TX: {len(eco)}")

pos = gpd.read_file("data/processed/positives_tx.gpkg").to_crs(TARGET_CRS)
pos["geometry"] = pos.geometry.centroid
pj = gpd.sjoin(pos, eco, how="left", predicate="within").drop(columns="index_right")

print(f"\n=== PRESENCE COUNTS PER ECOREGION (n={len(pos)} total) ===")
counts = pj["ecoregion"].value_counts(dropna=False)
print(counts.to_string())
print(f"\nUnassigned (outside any polygon): {pj['ecoregion'].isna().sum()}")
print(f"Ecoregions with >=1 presence: {pj['ecoregion'].nunique()}")
print(f"Ecoregions with >=10 presences: {(counts>=10).sum()}")
print(f"Ecoregions with >=8 presences:  {(counts>=8).sum()}")

pj.to_file("data/processed/positives_tx_ecoregion.gpkg", driver="GPKG")
eco.to_file("data/processed/tx_ecoregions_l3.gpkg", driver="GPKG")
print("\nSaved positives_tx_ecoregion.gpkg + tx_ecoregions_l3.gpkg")
