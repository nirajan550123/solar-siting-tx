"""
21_robustness_no_transmission.py
Gen-tie robustness check: rerun CV + SHAP WITHOUT dist_transmission.
If AUC stays high and substation/road still dominate irradiance, the 'realized siting' thesis
holds without the circular interconnection feature.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import shap
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
m = gpd.read_file("data/processed/model_table.gpkg")

# FEATURES WITHOUT dist_transmission
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority",
         "dist_substation","dist_road"]
NICE = {"ghi_nsrdb":"Solar irradiance (GHI)","slope_pct":"Slope","elevation":"Elevation",
        "lc_majority":"Land cover","dist_substation":"Dist. to substation","dist_road":"Dist. to road"}

pres = m[m["label"]==1].copy(); pa = m[m["label"]==0].copy()
REPS = sorted(pa["replicate"].unique()); K = 5

def models(): return {
    "Logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=0, n_jobs=-1)}

def evalfold(tr, te, X, y, name):
    Xtr,Xte,ytr,yte = X[tr],X[te],y[tr],y[te]
    if name=="Logistic":
        sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    mdl=models()[name]; mdl.fit(Xtr,ytr); p=mdl.predict_proba(Xte)[:,1]
    if len(np.unique(yte))<2: return None,None
    return roc_auc_score(yte,p), average_precision_score(yte,p)

print("=== CV WITHOUT dist_transmission ===\n")
for name in ["Logistic","RandomForest","XGBoost"]:
    out={}
    for scheme in ["random","spatial"]:
        roc,pr=[],[]
        for rep in REPS:
            sub=pd.concat([pres,pa[pa["replicate"]==rep]]).reset_index(drop=True)
            X=sub[FEATS].values; y=sub["label"].values
            if scheme=="random":
                folds=list(StratifiedKFold(K,shuffle=True,random_state=rep).split(X,y))
            else:
                f=sub["fold_130"].values
                folds=[(np.where(f!=k)[0],np.where(f==k)[0]) for k in range(K)]
            for tr,te in folds:
                a,b=evalfold(tr,te,X,y,name)
                if a is not None: roc.append(a); pr.append(b)
        out[scheme]=(np.mean(roc),np.mean(pr))
    print(f"{name}: spatial ROC {out['spatial'][0]:.3f}  PR {out['spatial'][1]:.3f}  (random ROC {out['random'][0]:.3f})")

# --- SHAP without transmission ---
print("\n=== SHAP WITHOUT dist_transmission ===")
pa0=pa[pa["replicate"]==0]
sub=pd.concat([pres,pa0]).reset_index(drop=True)
X=sub[FEATS].copy(); X.columns=[NICE[c] for c in FEATS]; y=sub["label"].values
rf=RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",random_state=0,n_jobs=-1).fit(X,y)
sv=shap.TreeExplainer(rf).shap_values(X)
svp = sv[1] if isinstance(sv,list) else sv[:,:,1] if sv.ndim==3 else sv
rank=pd.DataFrame({"feature":X.columns,"mean_abs_shap":np.abs(svp).mean(0)}).sort_values("mean_abs_shap",ascending=False)
print(rank.to_string(index=False))

plt.figure(); shap.summary_plot(svp,X,show=False,max_display=6)
plt.title("Solar siting drivers WITHOUT transmission (robustness)",fontsize=10)
plt.tight_layout(); plt.savefig(OUT/"shap_beeswarm_no_transmission.png",dpi=160,bbox_inches="tight"); plt.close()
print("\nSaved shap_beeswarm_no_transmission.png")
