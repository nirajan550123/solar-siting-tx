'''
03_boundary_and_map.py
Extract the Texas boundary from Census TIGER states, save it, sanity-check that all
176 positives fall inside Texas, and produce the first figure.
'''
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

STATES = Path('data/raw/tl_2024_us_state/tl_2024_us_state.shp')
POS    = Path('data/processed/positives_tx.gpkg')
BND_OUT = Path('data/processed/tx_boundary.gpkg')
FIG_OUT = Path('outputs/figures/tx_positives_map.png')

TX_CRS = 6580  # NAD83 Texas Centric Albers Equal Area

# --- Texas boundary ---
states = gpd.read_file(STATES)
# TIGER uses STUSPS for the 2-letter postal code
tx = states[states['STUSPS'] == 'TX'].copy().to_crs(TX_CRS)
print('Texas boundary rows:', len(tx))
print('Texas area (sq km):', round(tx.geometry.area.iloc[0] / 1e6, 1))

BND_OUT.parent.mkdir(parents=True, exist_ok=True)
tx.to_file(BND_OUT, driver='GPKG')
print('Saved boundary to', BND_OUT)

# --- Positives ---
pos = gpd.read_file(POS)  # already EPSG:6580
print('\nPositives loaded:', len(pos))

# --- Sanity check: are any facilities OUTSIDE Texas? ---
# Use centroid-in-polygon test (facilities are polygons; centroid is robust enough here)
tx_geom = tx.geometry.iloc[0]
pos_cent = pos.copy()
pos_cent['geometry'] = pos_cent.geometry.centroid
inside = pos_cent.within(tx_geom)
print('Facilities inside TX:', int(inside.sum()), '/ ', len(pos))
outside = pos_cent[~inside]
if len(outside) > 0:
    print('WARNING - facilities outside TX boundary:')
    print(outside[['p_name','p_county','p_cap_ac']].to_string())
else:
    print('All facilities fall inside Texas. Good.')

# --- Figure ---
fig, ax = plt.subplots(figsize=(10, 10))
tx.boundary.plot(ax=ax, color='#3b3024', linewidth=1.2)
tx.plot(ax=ax, color='#f0ead8', alpha=0.4)
# size markers by capacity so the big plants read visually
pos_cent.plot(ax=ax, markersize=pos_cent['p_cap_ac'] / 3, color='#c25b29',
              edgecolor='#3b3024', linewidth=0.3, alpha=0.8)
ax.set_title('Texas utility-scale ground-mounted solar facilities (USPVDB v4.0, n=176)\n'
             'marker size proportional to AC capacity', fontsize=12)
ax.set_axis_off()
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
print('\nSaved figure to', FIG_OUT)
