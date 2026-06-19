'''
07_extract_gee_features.py
Extract GEE raster features for presences (over footprint polygons) and pseudo-absences
(over circles whose radius = equivalent-circle radius of the MEDIAN facility footprint).

Raster features (server-side, GEE):
  - ghi        : annual global horizontal irradiance (NSRDB)  [solar resource]
  - slope_pct  : mean slope, percent grade (3DEP)             [terrain]
  - elevation  : mean elevation, m (3DEP)                     [terrain]
  - lc_majority: majority NLCD 2021 class                     [land cover]

Output: data/interim/features_gee.gpkg  (all 1936 points, label col: 1=presence, 0=absence)
'''
import ee, math
import os
import geopandas as gpd
import pandas as pd
from pathlib import Path

ee.Initialize(project=os.getenv('GEE_PROJECT', 'your-gee-project-id'))

POS = Path('data/processed/positives_tx.gpkg')
PA  = Path('data/processed/pseudo_absences.gpkg')
OUT = Path('data/interim/features_gee.gpkg')

# --- Median facility footprint -> equivalent-circle radius ---
pos = gpd.read_file(POS)  # EPSG:6580, polygons
pos['footprint_m2'] = pos.geometry.area
median_area = pos['footprint_m2'].median()
eq_radius = math.sqrt(median_area / math.pi)
print('Median facility footprint (m^2):', round(median_area, 1))
print('Equivalent-circle radius (m):   ', round(eq_radius, 1))

# --- Build geometry sets for sampling ---
# Presences: actual footprint polygons (reproject to 4326 for EE)
pos_4326 = pos.to_crs(4326)
# Pseudo-absences: buffer points by eq_radius (in projected CRS), then to 4326
pa = gpd.read_file(PA)  # EPSG:6580 points
pa_buf = pa.copy()
pa_buf['geometry'] = pa.geometry.buffer(eq_radius)
pa_4326 = pa_buf.to_crs(4326)

# --- GEE feature image ---
dem = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation')
slope_pct = ee.Terrain.slope(dem).multiply(math.pi/180).tan().multiply(100).rename('slope_pct')
elev = dem.rename('elevation')

# NOTE: GHI is NOT extracted here. The final pipeline pulls long-term annual-average
# GHI per point from the NREL Solar Resource API in the next script (08_pull_nsrdb_ghi.py)
# and merges it positionally. This script extracts only terrain and land cover.

feat_img = slope_pct.addBands(elev)

def sample_set(gdf_4326, label):
    feats = []
    for idx, row in gdf_4326.iterrows():
        gj = row.geometry.__geo_interface__
        feats.append(ee.Feature(ee.Geometry(gj), {'rid': int(idx), 'label': label}))
    fc = ee.FeatureCollection(feats)
    # continuous means
    means = feat_img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=100)
    # land-cover majority (mode)
    modes = nlcd.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
    mi = means.getInfo()['features']
    md = {f['properties']['rid']: f['properties'].get('mode') for f in modes.getInfo()['features']}
    rows = []
    for f in mi:
        p = f['properties']
        rows.append({
            'rid': p['rid'], 'label': p['label'],
            'slope_pct': p.get('slope_pct'),
            'elevation': p.get('elevation'), 'lc_majority': md.get(p['rid'])
        })
    return pd.DataFrame(rows)

print('\nSampling presences over footprints...')
df_pos = sample_set(pos_4326, 1)
print('  presences sampled:', len(df_pos))
print('Sampling pseudo-absences over', round(eq_radius,0), 'm circles...')
df_pa = sample_set(pa_4326, 0)
print('  pseudo-absences sampled:', len(df_pa))

df = pd.concat([df_pos, df_pa], ignore_index=True)

# attach geometry (centroids) + replicate id for PA
pos_c = pos.geometry.centroid
geom_pos = gpd.GeoDataFrame(df_pos.assign(replicate=-1), geometry=list(pos_c), crs=6580)
geom_pa  = gpd.GeoDataFrame(df_pa.assign(replicate=list(pa['replicate'])),
                            geometry=list(pa.geometry), crs=6580)
out = pd.concat([geom_pos, geom_pa], ignore_index=True)
out = gpd.GeoDataFrame(out, geometry='geometry', crs=6580)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_file(OUT, driver='GPKG')
print('\nSaved', len(out), 'rows to', OUT)
print('\nNull counts per feature:')
print(out[['ghi','slope_pct','elevation','lc_majority']].isna().sum().to_string())
