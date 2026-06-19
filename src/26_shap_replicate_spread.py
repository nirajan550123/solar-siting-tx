"""
27_shap_replicate_spread.py
Compute SHAP importance across all 10 replicates; report mean +/- SD
per feature (Nicodemus 2011 - single-model rankings unstable, average + report spread).
Also reports how often each feature holds its rank (ordering stability).
Outputs: importance table with error bars + figure.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
NICE = {"ghi_nsrdb":"Solar irradiance (GHI)","slope_pct":"Slope","elevation":"Elevation",
        "lc_majority":"Land cover","dist_transmission":"Dist. to transmission",
        "dist_substation":"Dist. to substation","dist_road":"Dist. to road"}

tx = gpd.read_file("data/processed/model_table.gpkg")
pres = tx[tx.label==1]; pa = tx[tx.label==0]

per_rep = []  # mean|SHAP| per feature, per replicate
for rep in sorted(pa.replicate.unique()):
    sub = pd.concat([pres, pa[pa.replicate==rep]]).reset_index(drop=True)
    X = sub[FEATS]; y = sub["label"].values
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=rep, n_jobs=-1).fit(X.values, y)
    sv = shap.TreeExplainer(rf).shap_values(X.values)
    svp = sv[1] if isinstance(sv, list) else sv[:,:,1] if sv.ndim==3 else sv
    per_rep.append(np.abs(svp).mean(axis=0))
    print(f"  replicate {rep} done")

M = np.array(per_rep)  # 10 x 7
mean_imp = M.mean(0); sd_imp = M.std(0)
order = np.argsort(mean_imp)[::-1]

print("\n=== SHAP IMPORTANCE ACROSS 10 REPLICATES (mean +/- SD) ===")
tbl = pd.DataFrame({"feature":[NICE[FEATS[i]] for i in order],
                    "mean_abs_shap":mean_imp[order], "sd":sd_imp[order]})
print(tbl.to_string(index=False))

# ordering stability: how often is each feature in its modal rank?
ranks = np.argsort(np.argsort(-M, axis=1), axis=1)  # rank per replicate (0=most important)
print("\n=== RANK STABILITY (rank per replicate, 0=top) ===")
for i in order:
    rk = ranks[:,i]
    print(f"  {NICE[FEATS[i]]:24s} modal rank {int(np.bincount(rk).argmax())}  (ranks seen: {sorted(set(rk))})")

# figure with error bars
plt.figure(figsize=(8,4.5))
yp = np.arange(len(order))[::-1]
plt.barh(yp, mean_imp[order], xerr=sd_imp[order], color="#c25b29", edgecolor="#3b3024",
         error_kw={"ecolor":"#3b3024","capsize":3})
plt.yticks(yp, [NICE[FEATS[i]] for i in order])
plt.xlabel("Mean |SHAP value|  (averaged over 10 replicates, +/- SD)")
plt.title("Feature importance for Texas solar siting\n(10 pseudo-absence replicates)")
plt.tight_layout(); plt.savefig(OUT/"shap_importance_replicates.png", dpi=170, bbox_inches="tight")
print("\nSaved shap_importance_replicates.png")
