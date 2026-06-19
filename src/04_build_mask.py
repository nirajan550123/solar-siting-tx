'''
04_build_mask.py
Build the developable-land mask for Texas, entirely server-side in Google Earth Engine.

Exclusion rules (paper-backed, see decisions_log):
  - Slope > 5% grade excluded            (Lopez et al. 2012; Al Garni & Awasthi 2017)
  - NLCD developed High [24] + Med [23]  excluded; Open [21] + Low [22] retained (Hernandez et al. 2015)
  - Water [11], ice/snow [12], forest [41/42/43], woody wetland [90], herb wetland [95] excluded
  - Retained candidate covers: barren [31], shrub [52], grassland [71], pasture/hay [81], cropland [82]

DEM:  USGS/3DEP/10m   (authoritative US 10 m elevation; slope in PERCENT, not degrees)
LULC: USGS NLCD 2021  (NLCD/2021/LAND_COVER via the released collection)

Output: prints the candidate-area fraction of Texas. No export yet -- we check the number first.
'''
import ee
import os

ee.Initialize(project=os.getenv('GEE_PROJECT', 'your-gee-project-id'))

# --- Texas boundary from the TIGER states FeatureCollection (server-side) ---
states = ee.FeatureCollection('TIGER/2018/States')
tx = states.filter(ee.Filter.eq('STUSPS', 'TX'))
tx_geom = tx.geometry()

# --- Slope (PERCENT grade) from 3DEP 10 m ---
dem = ee.Image('USGS/3DEP/10m').select('elevation')
# ee.Terrain.slope returns DEGREES; convert to percent grade = tan(deg)*100
slope_deg = ee.Terrain.slope(dem)
slope_pct = slope_deg.multiply(3.14159265 / 180.0).tan().multiply(100).rename('slope_pct')
slope_ok = slope_pct.lte(5)  # keep <= 5% grade

# --- NLCD 2021 land cover ---
nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD') \
         .filter(ee.Filter.eq('system:index', '2021')) \
         .first().select('landcover')

# Excluded land-cover codes
excluded_lc = ee.List([11, 12, 23, 24, 41, 42, 43, 90, 95])
lc_ok = nlcd.remap(excluded_lc, ee.List.repeat(0, excluded_lc.size()), 1).rename('lc_ok')
# remap: excluded codes -> 0, everything else (default) -> 1

# --- Combined developable mask ---
developable = slope_ok.And(lc_ok).rename('developable').clip(tx_geom)

# --- Quantify candidate-area fraction of Texas ---
# Pixel-area weighted mean of the binary mask = fraction developable.
pixel_area = ee.Image.pixelArea()
stats = developable.multiply(pixel_area).reduceRegion(
    reducer=ee.Reducer.sum(), geometry=tx_geom, scale=300, maxPixels=1e13, bestEffort=True
)
total = pixel_area.clip(tx_geom).reduceRegion(
    reducer=ee.Reducer.sum(), geometry=tx_geom, scale=300, maxPixels=1e13, bestEffort=True
)

dev_area_m2 = ee.Number(stats.get('developable'))
tot_area_m2 = ee.Number(total.get('area'))
frac = dev_area_m2.divide(tot_area_m2)

print('Computing candidate-area fraction (this calls GEE servers, ~10-30s)...')
print('Texas total area (sq km):  ', round(tot_area_m2.getInfo() / 1e6, 1))
print('Developable area (sq km):  ', round(dev_area_m2.getInfo() / 1e6, 1))
print('Developable FRACTION of TX:', round(frac.getInfo(), 4))
