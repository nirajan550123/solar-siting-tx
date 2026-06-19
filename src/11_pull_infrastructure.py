"""
14_pull_infrastructure.py
Pull Texas transmission lines + substations from live HIFLD ArcGIS REST services,
paginating past the 2000-record cap, clipped to the TX bounding box.
Saves to data/raw/. Roads handled separately via TIGER.
"""
import requests, json, time
import geopandas as gpd
from pathlib import Path

OUTDIR = Path("data/raw"); OUTDIR.mkdir(parents=True, exist_ok=True)
TX = gpd.read_file("data/processed/tx_boundary.gpkg").to_crs(4326)
xmin, ymin, xmax, ymax = TX.total_bounds
print(f"TX bbox: {xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}")

# Candidate HIFLD REST endpoints (try in order; first that works wins)
SERVICES = {
    "transmission": [
        "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0",
        "https://services5.arcgis.com/caWDr9qv9f34KIAZ/arcgis/rest/services/Transmission_Lines/FeatureServer/0",
    ],
    "substations": [
        "https://services5.arcgis.com/caWDr9qv9f34KIAZ/arcgis/rest/services/ElectricSubstations/FeatureServer/0",
        "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Substations/FeatureServer/0",
    ],
}

def pull(url, name):
    # envelope geometry filter = TX bbox, paginate by resultOffset
    feats = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*", "outSR": "4326", "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 2000,
        }
        r = requests.get(url + "/query", params=params, timeout=180)
        if r.status_code != 200:
            print(f"   {name}: HTTP {r.status_code}"); return None
        try:
            gj = r.json()
        except Exception:
            print(f"   {name}: non-JSON response"); return None
        batch = gj.get("features", [])
        if not batch:
            break
        feats.extend(batch)
        print(f"   {name}: {len(feats)} features so far")
        if len(batch) < 2000:
            break
        offset += 2000
        time.sleep(0.5)
    return feats

for name, urls in SERVICES.items():
    got = None
    for url in urls:
        print(f"\nTrying {name}: {url}")
        feats = pull(url, name)
        if feats:
            got = feats; break
    if not got:
        print(f"!! {name}: all endpoints failed - will need a manual fallback")
        continue
    gj = {"type": "FeatureCollection", "features": got}
    gdf = gpd.GeoDataFrame.from_features(gj, crs=4326)
    # clip precisely to TX
    gdf = gpd.clip(gdf, TX)
    out = OUTDIR / f"hifld_{name}_tx.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"   SAVED {len(gdf)} {name} features to {out}")
