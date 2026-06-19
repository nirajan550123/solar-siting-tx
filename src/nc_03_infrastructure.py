"""
nc_03_infrastructure.py
Pull NC transmission + substations from the same live HIFLD REST endpoints used for TX,
bbox-filtered to NC then clipped. (Roads come from TIGER FIPS 37, downloaded separately.)
"""
import requests, time
import geopandas as gpd
from pathlib import Path

OUTDIR = Path("data/raw")
NC = gpd.read_file("data/processed/nc_boundary.gpkg").to_crs(4326)
xmin,ymin,xmax,ymax = NC.total_bounds
print(f"NC bbox: {xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}")

SERVICES = {
    "transmission": "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0",
    "substations":  "https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/Electric_Substations/FeatureServer/0",
}

def pull(url):
    feats, off = [], 0
    while True:
        p = {"where":"1=1","geometry":f"{xmin},{ymin},{xmax},{ymax}",
             "geometryType":"esriGeometryEnvelope","inSR":"4326",
             "spatialRel":"esriSpatialRelIntersects","outFields":"*","outSR":"4326",
             "f":"geojson","resultOffset":off,"resultRecordCount":2000}
        r = requests.get(url+"/query", params=p, timeout=180)
        if r.status_code != 200: print("  HTTP",r.status_code); return None
        b = r.json().get("features",[])
        if not b: break
        feats += b; print("  ",len(feats),"so far")
        if len(b)<2000: break
        off += 2000; time.sleep(0.5)
    return feats

for name,url in SERVICES.items():
    print(f"\n{name}:")
    f = pull(url)
    if f:
        g = gpd.GeoDataFrame.from_features({"type":"FeatureCollection","features":f}, crs=4326)
        g = gpd.clip(g, NC)
        out = OUTDIR/f"hifld_{name}_nc.gpkg"
        g.to_file(out, driver="GPKG")
        print(f"  SAVED {len(g)} to {out}")
    else:
        print(f"  FAILED")
