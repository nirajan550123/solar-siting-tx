"""
29_aoa_lpd_check.py
AOA validation and local data-point density (LPD) check. Reproduce the AOA, then validate it:
  A. Reproduce current AOA (validate fidelity vs known 89.5%)
  B. Spatial-CV-derived threshold (Meyer & Pebesma 2021: DI from cross-fold distances, not
     same-set NN which underestimates) - the methodologically correct threshold
  C. Local data-point density (LPD, Schumacher et al. 2025): how many training points actually
     support each 'inside' grid cell (reveals isolated-vs-dense support)
  D. Unweighted-DI sanity check: does impurity-weighting drive permissiveness?
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path("outputs"); (OUT/"figures").mkdir(parents=True, exist_ok=True)
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]

m = gpd.read_file("data/processed/model_table.gpkg")
grid = gpd.read_file("data/interim/grid_features.gpkg")
Xtr = m[FEATS].values; ytr = m["label"].values
Xg = grid[FEATS].values
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1

rf = RandomForestClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(Xtr, ytr)
imp = rf.feature_importances_

def aoa(weights, threshold_mode, fold_col=None):
    W = weights
    Ztr = (Xtr-mu)/sd*W; Zg = (Xg-mu)/sd*W
    tree = cKDTree(Ztr)
    d_grid,_ = tree.query(Zg, k=1)
    if threshold_mode == "same_set":
        d_tr,_ = tree.query(Ztr, k=2); train_di_raw = d_tr[:,1]
    elif threshold_mode == "cross_fold":
        # each training pt's NN distance to points in OTHER folds (canonical CV-DI)
        folds = m[fold_col].values
        train_di_raw = np.empty(len(Ztr))
        for k in np.unique(folds):
            te = folds==k; tr = ~te
            t2 = cKDTree(Ztr[tr]); dd,_ = t2.query(Ztr[te], k=1)
            train_di_raw[te] = dd
    mean_d = train_di_raw.mean()
    DI_grid = d_grid/mean_d; DI_tr = train_di_raw/mean_d
    q75,q25 = np.percentile(DI_tr,[75,25]); thr = q75 + 1.5*(q75-q25)
    inside = DI_grid <= thr
    return inside, DI_grid, thr, Ztr, Zg, train_di_raw

# A. reproduce current (impurity weights, same-set threshold)
ins_cur,_,thr_cur,_,_,_ = aoa(imp, "same_set")
print(f"A. Reproduced current AOA: {ins_cur.mean()*100:.1f}%  (expected ~89.5% => fidelity check)")

# B. canonical spatial-CV threshold (impurity weights, cross-fold DI)
ins_cv, DI_cv, thr_cv, Ztr, Zg, train_di = aoa(imp, "cross_fold", fold_col="fold_130")
print(f"B. Spatial-CV-derived threshold AOA: {ins_cv.mean()*100:.1f}%  (canonical Meyer&Pebesma)")

# C. LPD: count training points within the AOA threshold distance of each grid cell
# (Schumacher 2025: density of similar training points, not just nearest)
thr_abs = thr_cv * train_di.mean()   # convert DI threshold back to raw scaled-space distance
tree_tr = cKDTree(Ztr)
lpd = tree_tr.query_ball_point(Zg, r=thr_abs, return_length=True)
grid["lpd"] = lpd
inside_cv = ins_cv
print(f"\nC. LPD (within-threshold training-point support) for INSIDE-AOA cells:")
li = lpd[inside_cv]
print(f"   median LPD: {np.median(li):.0f}  | mean: {li.mean():.1f}")
print(f"   inside-AOA cells with LPD>=5 (well-supported): {(li>=5).mean()*100:.0f}%")
print(f"   inside-AOA cells with LPD==1 (single-point support, fragile): {(li<=1).mean()*100:.0f}%")
print(f"   inside-AOA cells with LPD>=10: {(li>=10).mean()*100:.0f}%")

# D. unweighted sanity check
ins_unw,_,_,_,_,_ = aoa(np.ones(len(FEATS)), "cross_fold", fold_col="fold_130")
print(f"\nD. Unweighted-DI AOA: {ins_unw.mean()*100:.1f}%  (vs impurity-weighted {ins_cv.mean()*100:.1f}%)")
print(f"   if much SMALLER => impurity-weighting was inflating coverage")

# --- LPD map ---
fig,ax = plt.subplots(figsize=(11,11))
TX = gpd.read_file("data/processed/tx_boundary.gpkg")
TX.boundary.plot(ax=ax,color="#3b3024",linewidth=1.2,zorder=3)
gi = grid[inside_cv]; go = grid[~inside_cv]
go.plot(ax=ax,color="#dddddd",markersize=3,alpha=0.4,zorder=1)
cmap=LinearSegmentedColormap.from_list("lpd",["#f5e6c8","#d99441","#8a3a12","#3b1505"])
gi.plot(ax=ax,column="lpd",cmap=cmap,markersize=6,vmin=0,vmax=np.percentile(li,95),zorder=2)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,np.percentile(li,95)))
cb=fig.colorbar(sm,ax=ax,fraction=0.035,pad=0.02); cb.set_label("Local training-point density (LPD)")
ax.set_title("AOA local data-point density (LPD)\ndark = well-supported; pale = sparse/fragile support; grey = outside AOA",fontsize=11)
ax.set_axis_off(); plt.tight_layout()
plt.savefig(OUT/"figures"/"aoa_lpd_map.png",dpi=160,bbox_inches="tight")
print("\nSaved aoa_lpd_map.png")
