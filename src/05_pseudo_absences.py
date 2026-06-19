'''
05_pseudo_absences.py
Generate 10 replicate pseudo-absence sets (n=176 each) for Texas, server-side in GEE.

Allowed sampling area = developable mask AND (>1 km from any USPVDB facility).
  - developable: slope <= 5% grade AND NLCD not in excluded classes
  - 1 km exclusion buffer around facilities (Barbet-Massin 2012; VanDerWal 2009)
10 balanced replicates of 176 points each, distinct random seeds, averaged later (not pooled).

DEM migrated to USGS/3DEP/10m_collection (mosaic) per deprecation notice.
Outputs: data/processed/pseudo_absences.gpkg  (all 10 replicates, column 'replicate' = 0..9)
'''
import ee
import os
import geopandas as gpd
import pandas as pd
from pathlib import Path

ee.Initialize(project=os.getenv('GEE_PROJECT', 'your-gee-project-id'))

POS = Path('data/processed/positives_tx.gpkg')
OUT = Path('data/processed/pseudo_absences.gpkg')
N_PER = 176          # 1:1 with presences
N_REP = 10           # replicates
BUFFER_M = 1000      # 1 km exclusion buffer

# --- Texas geometry ---
tx = ee.FeatureCollection('TIGER/2018/States').filter(ee.Filter.eq('STUSPS', 'TX'))
tx_geom = tx.geometry()

# --- Facilities -> EE FeatureCollection (read local, send coords up) ---
pos = gpd.read_file(POS).to_crs(4326)
pos_pts = pos.copy()
pos_pts['geometry'] = pos_pts.geometry.centroid
feats = [ee.Feature(ee.Geometry.Point([float(g.x), float(g.y)])) for g in pos_pts.geometry]
fac_fc = ee.FeatureCollection(feats)
print('Facilities sent to EE:', len(feats))

# --- Developable mask (migrated DEM) ---
dem = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation')
slope_deg = ee.Terrain.slope(dem)
slope_pct = slope_deg.multiply(3.14159265 / 180.0).tan().multiply(100)
slope_ok = slope_pct.lte(5)

nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD') \
         .filter(ee.Filter.eq('system:index', '2021')).first().select('landcover')
excluded_lc = ee.List([11, 12, 23, 24, 41, 42, 43, 90, 95])
lc_ok = nlcd.remap(excluded_lc, ee.List.repeat(0, excluded_lc.size()), 1)

developable = slope_ok.And(lc_ok)

# --- 1 km exclusion buffer around facilities, as a raster mask ---
# distance-to-facility >= 1000 m  ->  allowed
dist = fac_fc.distance(searchRadius=5000).unmask(9999)  # metres to nearest facility
outside_buffer = dist.gte(BUFFER_M)

allowed = developable.And(outside_buffer).selfMask().rename('allowed').clip(tx_geom)

# --- Draw replicates via stratifiedSample (class 1 = allowed) ---
all_rows = []
for rep in range(N_REP):
    samp = allowed.stratifiedSample(
        numPoints=N_PER, classBand='allowed', region=tx_geom,
        scale=300, seed=rep, geometries=True
    )
    info = samp.getInfo()
    n = len(info['features'])
    for f in info['features']:
        x, y = f['geometry']['coordinates']
        all_rows.append({'replicate': rep, 'lon': x, 'lat': y})
    print(f'  replicate {rep}: drew {n} points')

df = pd.DataFrame(all_rows)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(6580)
OUT.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUT, driver='GPKG')
print('\nTotal pseudo-absence points:', len(gdf), '(expected', N_PER * N_REP, ')')
print('Saved to', OUT)
