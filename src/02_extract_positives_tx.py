'''
02_extract_positives_tx.py
Filter USPVDB v4.0 to Texas ground-mounted utility-scale solar facilities.
Decision (see decisions_log): p_state == 'TX' AND p_sys_type == 'ground' (pure ground only;
compound canopy/rooftop/floating types excluded because their siting logic differs).
Writes positives to data/processed/positives_tx.gpkg in an equal-area CRS.
'''
from pathlib import Path
import geopandas as gpd

RAW = Path('data/raw/uspvdbGeoJSON/uspvdb_v4_0_20260414.geojson')
OUT = Path('data/processed/positives_tx.gpkg')

# Texas-appropriate projected CRS: NAD83 / Texas Centric Albers Equal Area (EPSG:6580)
# Equal-area is correct for area-true distance/sampling work.
TX_CRS = 6580

gdf = gpd.read_file(RAW)
print('Total USPVDB facilities:', len(gdf))

# --- Filter ---
tx = gdf[(gdf['p_state'] == 'TX') & (gdf['p_sys_type'] == 'ground')].copy()
print('TX ground-mounted facilities:', len(tx))

# --- Verify capacity floor (USPVDB is nominally >= 1 MW; confirm) ---
print('\nCapacity AC (MW) summary:')
print(tx['p_cap_ac'].describe().to_string())
below_1mw = (tx['p_cap_ac'] < 1).sum()
print('Facilities below 1 MW AC:', below_1mw)

# --- Year range ---
print('\nInstall year range:', int(tx['p_year'].min()), 'to', int(tx['p_year'].max()))

# --- Reproject to equal-area and save ---
tx = tx.to_crs(TX_CRS)
OUT.parent.mkdir(parents=True, exist_ok=True)
tx.to_file(OUT, driver='GPKG')
print('\nSaved', len(tx), 'TX positives to', OUT)
print('Output CRS:', tx.crs)
