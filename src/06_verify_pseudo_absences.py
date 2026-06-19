'''
06_verify_pseudo_absences.py
Guard checks on the pseudo-absence sample before feature extraction:
  1. All points inside Texas
  2. No pseudo-absence within 1 km of any real facility (buffer enforced)
  3. Replicate counts correct
'''
from pathlib import Path
import geopandas as gpd

POS = Path('data/processed/positives_tx.gpkg')
PA  = Path('data/processed/pseudo_absences.gpkg')
BND = Path('data/processed/tx_boundary.gpkg')

pos = gpd.read_file(POS)          # EPSG:6580
pa  = gpd.read_file(PA)           # EPSG:6580
tx  = gpd.read_file(BND)          # EPSG:6580

# 1. inside Texas
pa_cent = pa.copy()
inside = pa_cent.within(tx.geometry.iloc[0])
print('Pseudo-absences inside TX:', int(inside.sum()), '/', len(pa))

# 2. buffer check: nearest facility distance for every PA point
fac_centroids = pos.copy()
fac_centroids['geometry'] = fac_centroids.geometry.centroid
fac_union = fac_centroids.geometry.union_all()
pa['dist_to_nearest_facility_m'] = pa.geometry.distance(fac_union)
min_dist = pa['dist_to_nearest_facility_m'].min()
violations = (pa['dist_to_nearest_facility_m'] < 1000).sum()
print('Minimum PA-to-facility distance (m):', round(min_dist, 1))
print('PA points within 1 km of a facility (should be 0):', int(violations))

# 3. replicate counts
print('\nPoints per replicate:')
print(pa['replicate'].value_counts().sort_index().to_string())
