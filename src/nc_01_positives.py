"""
nc_01_positives.py
Extract NC ground-mounted utility-scale solar facilities from USPVDB (same filter as TX).
NC-appropriate projected CRS: EPSG:32119 (NAD83 / North Carolina, metres), matching
the metres-based distance features the TX-trained model expects.
"""
from pathlib import Path
import geopandas as gpd

RAW = Path("data/raw/uspvdbGeoJSON/uspvdb_v4_0_20260414.geojson")
OUT = Path("data/processed/positives_nc.gpkg")
NC_CRS = 32119

gdf = gpd.read_file(RAW)
nc = gdf[(gdf["p_state"]=="NC") & (gdf["p_sys_type"]=="ground")].copy()
print("NC ground-mounted facilities:", len(nc))
print("Capacity AC (MW):", round(nc["p_cap_ac"].min(),1), "-", round(nc["p_cap_ac"].max(),1),
      "| median", round(nc["p_cap_ac"].median(),1))
print("Below 1 MW:", int((nc["p_cap_ac"]<1).sum()))
print("Install years:", int(nc["p_year"].min()), "-", int(nc["p_year"].max()))

nc = nc.to_crs(NC_CRS)
OUT.parent.mkdir(parents=True, exist_ok=True)
nc.to_file(OUT, driver="GPKG")
print(f"\nSaved {len(nc)} NC positives to {OUT} (EPSG:{NC_CRS})")
