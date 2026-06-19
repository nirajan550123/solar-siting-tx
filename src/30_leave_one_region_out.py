"""
30b_loro.py
Leave-one-region-out CV (Roberts et al. 2017; Wenger & Olden 2012; Valavi et al. 2019).
7 merged EPA L3 ecoregion folds. Hold out each region, train on the rest, test inside it,
loop over 10 PA sets. Report per-region AUC + Continuous Boyce Index + AUC-PR.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]

# merge map: sparse ecoregion -> partner
MERGE = {
    "South Central Plains": "East+SouthCentral",
    "East Central Texas Plains": "East+SouthCentral",
    "Southwestern Tablelands": "HighPlains+Tablelands",
    "High Plains": "HighPlains+Tablelands",
    "Central Great Plains": "CrossTimbers+CGP",
    "Cross Timbers": "CrossTimbers+CGP",
    "Edwards Plateau": "Edwards+SouthTexas",
    "Southern Texas Plains": "Edwards+SouthTexas",
    "Texas Blackland Prairies": "Blackland Prairies",
    "Western Gulf Coastal Plain": "Western Gulf Coastal Plain",
    "Chihuahuan Deserts": "Chihuahuan Deserts",
}

def boyce(pres_scores, bg_scores, nbins=10):
    # Continuous Boyce Index (Hirzel et al. 2006): Spearman corr of P/E ratio vs suitability bins
    alls = np.concatenate([pres_scores, bg_scores])
    edges = np.linspace(alls.min(), alls.max(), nbins+1)
    pe = []
    for i in range(nbins):
        lo,hi = edges[i],edges[i+1]
        pin = ((pres_scores>=lo)&(pres_scores<hi)).sum()/max(len(pres_scores),1)
        ein = ((alls>=lo)&(alls<hi)).sum()/max(len(alls),1)
        if ein>0: pe.append(pin/ein)
        else: pe.append(np.nan)
    pe=np.array(pe); mids=(edges[:-1]+edges[1:])/2
    ok=~np.isnan(pe)
    if ok.sum()<3: return np.nan
    from scipy.stats import spearmanr
    return spearmanr(mids[ok],pe[ok]).correlation

# assign every model-table row to a region via spatial join
eco = gpd.read_file("data/processed/tx_ecoregions_l3.gpkg")
m = gpd.read_file("data/processed/model_table.gpkg").to_crs(eco.crs)
m_pts = m.copy(); m_pts["geometry"]=m_pts.geometry.centroid
mj = gpd.sjoin(m_pts, eco, how="left", predicate="within").drop(columns="index_right")
# nearest-fill any that landed just outside a polygon
if mj["ecoregion"].isna().any():
    miss = mj[mj["ecoregion"].isna()]
    nn = gpd.sjoin_nearest(miss[["geometry"]], eco, how="left")
    mj.loc[mj["ecoregion"].isna(),"ecoregion"] = nn["ecoregion"].values
mj["region"] = mj["ecoregion"].map(MERGE)
m["region"] = mj["region"].values
print("Rows per merged region (presence / total):")
for r in sorted(m["region"].unique()):
    sub=m[m["region"]==r]; print(f"  {r:30s} pres={int((sub.label==1).sum()):3d}  total={len(sub)}")

REPS = sorted(m[m.label==0].replicate.unique())
regions = sorted(m["region"].unique())
rows=[]
for r in regions:
    in_r = m["region"]==r
    test_pres = m[in_r & (m.label==1)]
    if len(test_pres) < 10:
        rows.append((r,len(test_pres),np.nan,np.nan,np.nan)); continue
    aucs,boyces,prs=[],[],[]
    for rep in REPS:
        train = m[(~in_r) & ((m.label==1)|((m.label==0)&(m.replicate==rep)))]
        test  = m[( in_r) & ((m.label==1)|((m.label==0)&(m.replicate==rep)))]
        if test.label.nunique()<2: continue
        rf=RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",random_state=rep,n_jobs=-1).fit(train[FEATS].values,train.label.values)
        p=rf.predict_proba(test[FEATS].values)[:,1]; y=test.label.values
        aucs.append(roc_auc_score(y,p)); prs.append(average_precision_score(y,p))
        boyces.append(boyce(p[y==1],p[y==0]))
    rows.append((r,len(test_pres),np.nanmean(aucs),np.nanmean(boyces),np.nanmean(prs)))

print("\n=== LEAVE-ONE-REGION-OUT RESULTS (mean over 10 PA sets) ===")
print(f"{'Region':32s}{'n_pres':>7}{'AUC':>8}{'Boyce':>8}{'AUC-PR':>8}")
for r,n,a,b,pr in rows:
    print(f"{r:32s}{n:7d}{a:8.3f}{b:8.3f}{pr:8.3f}" if not np.isnan(a) else f"{r:32s}{n:7d}   n/a (sparse)")

valid=[(r,n,a,b,pr) for r,n,a,b,pr in rows if not np.isnan(a)]
ns=np.array([n for _,n,_,_,_ in valid]); aus=np.array([a for _,_,a,_,_ in valid])
wmean=np.average(aus,weights=ns)
print(f"\nPresence-weighted mean LORO AUC: {wmean:.3f}")
print(f"Unweighted mean: {aus.mean():.3f}  | SD across regions: {aus.std():.3f}  | range {aus.min():.3f}-{aus.max():.3f}")
print(f"\nGRADIENT: random CV ~0.93 > block CV 0.92 > LORO {wmean:.3f} > NC transfer 0.76")
