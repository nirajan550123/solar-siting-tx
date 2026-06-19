"""
28_gentie_full_sensitivity.py
Gen-tie circularity sensitivity. Historical transmission data unavailable (CEII-
restricted), so we bound the issue via staged feature-removal sensitivity (Bellemare et al. 2017;
Wu et al. 2026 precedent).
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

tx = gpd.read_file("data/processed/model_table.gpkg")
pres = tx[tx.label==1]; pa = tx[tx.label==0]
REPS = sorted(pa.replicate.unique())

FEATURE_SETS = {
    "Full (all 7)": ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"],
    "Drop transmission": ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_substation","dist_road"],
    "Drop transmission+substation": ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_road"],
    "No grid features (ex-ante only)": ["ghi_nsrdb","slope_pct","elevation","lc_majority"],
}

print("=== STAGED GEN-TIE SENSITIVITY (spatial CV, 10 reps x 5 folds) ===\n")
results = {}
for name, feats in FEATURE_SETS.items():
    rocs, prs = [], []
    for rep in REPS:
        sub = pd.concat([pres, pa[pa.replicate==rep]]).reset_index(drop=True)
        f = sub["fold_130"].values
        X = sub[feats].values; y = sub["label"].values
        for k in range(5):
            tr = np.where(f!=k)[0]; te = np.where(f==k)[0]
            if len(np.unique(y[te]))<2: continue
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=rep, n_jobs=-1).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:,1]
            rocs.append(roc_auc_score(y[te],p)); prs.append(average_precision_score(y[te],p))
    results[name] = (np.mean(rocs), np.mean(prs), len(feats))
    print(f"  {name:34s} ROC {np.mean(rocs):.3f}  PR {np.mean(prs):.3f}  (n_feat={len(feats)})")

floor = results["No grid features (ex-ante only)"][0]
print(f"\nFLOOR (ex-ante features only): ROC {floor:.3f}")
print("  >0.65 => siting signal survives even with ZERO interconnection-contaminable features")
print("  ~0.50 => siting predictable ONLY via grid features (lean harder on disclosure)")
