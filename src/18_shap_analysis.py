"""
20_shap_analysis.py
SHAP interpretation of the RF siting model. Produces the headline figure:
which features drive Texas solar siting. Uses replicate 0 (representative; SHAP needs one model).
Outputs: beeswarm, importance bar, transmission dependence plot, and a printed ranking.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
from pathlib import Path

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
m = gpd.read_file("data/processed/model_table.gpkg")

FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority",
         "dist_transmission","dist_substation","dist_road"]
NICE = {"ghi_nsrdb":"Solar irradiance (GHI)","slope_pct":"Slope","elevation":"Elevation",
        "lc_majority":"Land cover","dist_transmission":"Dist. to transmission",
        "dist_substation":"Dist. to substation","dist_road":"Dist. to road"}

pres = m[m["label"]==1]
pa0  = m[(m["label"]==0)&(m["replicate"]==0)]
sub = pd.concat([pres, pa0]).reset_index(drop=True)
X = sub[FEATS].copy(); X.columns = [NICE[c] for c in FEATS]
y = sub["label"].values

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                            random_state=0, n_jobs=-1).fit(X, y)

print("Computing SHAP values (TreeExplainer)...")
expl = shap.TreeExplainer(rf)
sv = expl.shap_values(X)
# binary RF -> list of 2 arrays; take presence class (index 1)
sv_pres = sv[1] if isinstance(sv, list) else sv[:,:,1] if sv.ndim==3 else sv

# --- mean |SHAP| ranking ---
imp = np.abs(sv_pres).mean(axis=0)
rank = pd.DataFrame({"feature":X.columns,"mean_abs_shap":imp}).sort_values("mean_abs_shap",ascending=False)
print("\n===== FEATURE IMPORTANCE (mean |SHAP|) =====")
print(rank.to_string(index=False))

# --- beeswarm (headline) ---
plt.figure()
shap.summary_plot(sv_pres, X, show=False, max_display=7)
plt.title("What drives Texas utility-scale solar siting (SHAP)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT/"shap_beeswarm.png", dpi=160, bbox_inches="tight"); plt.close()

# --- importance bar ---
plt.figure(figsize=(7,4))
plt.barh(rank["feature"][::-1], rank["mean_abs_shap"][::-1], color="#c25b29", edgecolor="#3b3024")
plt.xlabel("Mean |SHAP value|"); plt.title("Feature importance for solar siting (SHAP)")
plt.tight_layout(); plt.savefig(OUT/"shap_importance.png", dpi=160, bbox_inches="tight"); plt.close()

# --- dependence: distance to transmission ---
plt.figure()
shap.dependence_plot("Dist. to transmission", sv_pres, X, show=False, interaction_index=None)
plt.tight_layout(); plt.savefig(OUT/"shap_dependence_transmission.png", dpi=160, bbox_inches="tight"); plt.close()

rank.to_csv("outputs/shap_importance.csv", index=False)
print("\nSaved: shap_beeswarm.png, shap_importance.png, shap_dependence_transmission.png")
print("       outputs/shap_importance.csv")
