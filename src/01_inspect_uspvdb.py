'''
01_inspect_uspvdb.py
Inspect USPVDB v4.0 structure before any filtering.
Prints columns, dtypes, key-field value counts, and a sample row.
No filtering, no writing. Look first, filter second.
'''
from pathlib import Path
import geopandas as gpd
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

RAW = Path('data/raw/uspvdbGeoJSON/uspvdb_v4_0_20260414.geojson')

print('Loading', RAW)
gdf = gpd.read_file(RAW)

print('\n=== SHAPE ===')
print(gdf.shape, '(rows, cols)')

print('\n=== CRS ===')
print(gdf.crs)

print('\n=== GEOMETRY TYPES ===')
print(gdf.geometry.geom_type.value_counts())

print('\n=== COLUMNS & DTYPES ===')
print(gdf.dtypes)

print('\n=== ONE SAMPLE ROW (transposed) ===')
print(gdf.drop(columns='geometry').iloc[0].to_string())

# Look for the fields we will need: state, capacity, site/mount type, year
candidates = [c for c in gdf.columns if any(k in c.lower()
              for k in ['state','st_','capac','mw','type','mount','site','year','status'])]
print('\n=== CANDIDATE KEY FIELDS ===')
print(candidates)

for c in candidates:
    print(f'\n--- value counts: {c} ---')
    vc = gdf[c].value_counts(dropna=False)
    print(vc.head(20).to_string())
