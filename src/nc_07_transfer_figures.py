"""
nc_08_transfer_figures_v2.py
Three standalone, portfolio-ready transfer figures:
  fig1_transfer_roc.png      - TX (cross-validated, honest) vs NC transfer ROC
  fig2_transfer_mechanism.png - transmission-distance distributions (regime shift)
  fig3_policy_paradox.png    - GHI vs facility count (policy over physics)
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
TXc, NCc = "#c25b29", "#2c6e8f"
plt.rcParams.update({"font.size":11,"axes.spline.top" if False else "axes.grid":False})

tx = gpd.read_file("data/processed/model_table.gpkg")
nc = gpd.read_file("data/processed/nc_eval_table.gpkg")

# ---------- FIG 1: honest ROC ----------
# TX curve from SPATIAL-CV out-of-fold predictions (matches 0.92), pooled over folds
pres = tx[tx.label==1]; pa = tx[tx.label==0]
oof_y, oof_p = [], []
f = tx["fold_130"].values
for k in range(5):
    tr = np.where(f!=k)[0]; te = np.where(f==k)[0]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(tx.iloc[tr][FEATS].values, tx.iloc[tr]["label"].values)
    oof_p.extend(rf.predict_proba(tx.iloc[te][FEATS].values)[:,1]); oof_y.extend(tx.iloc[te]["label"].values)
oof_y=np.array(oof_y); oof_p=np.array(oof_p)
tx_auc = roc_auc_score(oof_y, oof_p)

# NC transfer: full TX model -> NC
rf_full = RandomForestClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(tx[FEATS].values, tx["label"].values)
p_nc = rf_full.predict_proba(nc[FEATS].values)[:,1]
nc_auc = roc_auc_score(nc["label"].values, p_nc)

fig,ax=plt.subplots(figsize=(6,6))
for y,p,lab,c in [(oof_y,oof_p,f"Texas (spatial CV, AUC {tx_auc:.2f})",TXc),
                  (nc["label"].values,p_nc,f"North Carolina (transfer, AUC {nc_auc:.2f})",NCc)]:
    fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color=c,lw=2.4,label=lab)
ax.plot([0,1],[0,1],"--",color="#aaa",lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Solar-siting model: within-Texas vs transfer to North Carolina")
ax.legend(loc="lower right"); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(OUT/"fig1_transfer_roc.png",dpi=170,bbox_inches="tight"); plt.close()

# ---------- FIG 2: mechanism ----------
txd = tx[tx.label==1]["dist_transmission"].values/1000
ncd = nc[nc.label==1]["dist_transmission"].values/1000
fig,ax=plt.subplots(figsize=(7.5,5))
bins=np.linspace(0,15,40)
ax.hist(txd,bins=bins,color=TXc,alpha=0.75,label=f"Texas presences (median {np.median(txd):.2f} km)")
ax.hist(ncd,bins=bins,color=NCc,alpha=0.6,label=f"North Carolina presences (median {np.median(ncd):.2f} km)")
ax.set_xlabel("Distance to nearest transmission line (km)"); ax.set_ylabel("Number of facilities")
ax.set_title("Why the model transfers only partially:\nTexas plants hug transmission; NC plants sit farther out")
ax.legend()
plt.tight_layout(); plt.savefig(OUT/"fig2_transfer_mechanism.png",dpi=170,bbox_inches="tight"); plt.close()

# ---------- FIG 3: policy paradox (clean grouped bars, dual axis) ----------
ghi=[4.79,4.46]; nfac=[176,753]; states=["Texas","North Carolina"]
x=np.arange(2); w=0.38
fig,ax=plt.subplots(figsize=(7,5))
ax2=ax.twinx()
b1=ax.bar(x-w/2, ghi, w, color="#e8b54a", label="Median solar irradiance (GHI)")
b2=ax2.bar(x+w/2, nfac, w, color="#7a2d12", label="Number of solar facilities")
ax.set_xticks(x); ax.set_xticklabels(states)
ax.set_ylabel("Median GHI (kWh/m2/day)"); ax2.set_ylabel("Utility-scale solar facilities")
ax.set_ylim(0,6); ax2.set_ylim(0,850)
for i,v in enumerate(ghi): ax.text(i-w/2, v+0.1, f"{v:.2f}", ha="center", fontsize=10)
for i,v in enumerate(nfac): ax2.text(i+w/2, v+12, str(v), ha="center", fontsize=10)
ax.set_title("Policy over physics: North Carolina has lower sun\nbut four times as many solar facilities")
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, loc="upper center", fontsize=9)
plt.tight_layout(); plt.savefig(OUT/"fig3_policy_paradox.png",dpi=170,bbox_inches="tight"); plt.close()

print("Saved fig1_transfer_roc.png, fig2_transfer_mechanism.png, fig3_policy_paradox.png")
print(f"TX spatial-CV AUC (curve): {tx_auc:.3f}  | NC transfer AUC: {nc_auc:.3f}")
