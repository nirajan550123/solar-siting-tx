"""
14b_pull_substations.py
Pull TX substations from the public HIFLD electric substations REST endpoint.
"""
import requests, time
import geopandas as gpd
from pathlib import Path

OUTDIR = Path("data/raw")
TX = gpd.read_file("data/processed/tx_boundary.gpkg").to_crs(4326)
xmin, ymin, xmax, ymax = TX.total_bounds

URLS = [
    "https://services.arcgis.com/G4S1dGvn7PIgYd6Y/ArcGIS/rest/services/HIFLD_electric_power_substations/FeatureServer/0",
    "https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/Electric_Substations/FeatureServer/0",
]

def pull(url):
    feats, offset = [], 0
    while True:
        params = {"where":"1=1","geometry":f"{xmin},{ymin},{xmax},{ymax}",
                  "geometryType":"esriGeometryEnvelope","inSR":"4326",
                  "spatialRel":"esriSpatialRelIntersects","outFields":"*","outSR":"4326",
                  "f":"geojson","resultOffset":offset,"resultRecordCount":2000}
        r = requests.get(url+"/query", params=params, timeout=180)
        if r.status_code != 200:
            print("  HTTP", r.status_code); return None
        try: gj = r.json()
        except Exception: print("  non-JSON"); return None
        batch = gj.get("features", [])
        if not batch: break
        feats.extend(batch); print("  ", len(feats), "so far")
        if len(batch) < 2000: break
        offset += 2000; time.sleep(0.5)
    return feats

for url in URLS:
    print("Trying:", url)
    feats = pull(url)
    if feats:
        gj = {"type":"FeatureCollection","features":feats}
        gdf = gpd.GeoDataFrame.from_features(gj, crs=4326)
        gdf = gpd.clip(gdf, TX)
        out = OUTDIR / "hifld_substations_tx.gpkg"
        gdf.to_file(out, driver="GPKG")
        print("SAVED", len(gdf), "substations to", out)
        break
    print("  failed, trying next")
