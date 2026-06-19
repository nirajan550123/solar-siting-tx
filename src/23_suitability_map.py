"""
24_suitability_map_v2.py
Corrected suitability map: train final RF on balanced (1:1) data, express output as a
RELATIVE siting-suitability percentile (honest for presence-background; uncompressed, readable).
Keep AOA mask. Re-validate that existing facilities fall in high-suitability cells.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path("outputs")
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]

m = gpd.read_file("data/processed/model_table.gpkg")
grid = gpd.read_file("data/interim/grid_features.gpkg")
pos = gpd.read_file("data/processed/positives_tx.gpkg")
TX = gpd.read_file("data/processed/tx_boundary.gpkg")

# --- balanced final model: average raw scores over the 10 replicates (each 176:176) ---
pres = m[m["label"]==1]; pa = m[m["label"]==0]
Xg = grid[FEATS].values
score = np.zeros(len(grid))
for rep in sorted(pa["replicate"].unique()):
    sub = pd.concat([pres, pa[pa["replicate"]==rep]])
    rf = RandomForestClassifier(n_estimators=400, random_state=rep, n_jobs=-1).fit(sub[FEATS].values, sub["label"].values)
    score += rf.predict_proba(Xg)[:,1]
score /= 10
grid["suit_raw"] = score
print("Balanced raw score range:", round(score.min(),3), "-", round(score.max(),3))

# --- relative suitability = percentile rank (0-1) ---
grid["suitability"] = grid["suit_raw"].rank(pct=True)

# --- AOA (importance-weighted DI), trained on all data for the mask ---
rf_full = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(m[FEATS].values, m["label"].values)
imp = rf_full.feature_importances_
Xtr = m[FEATS].values; mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
Ztr=(Xtr-mu)/sd*imp; Zg=(Xg-mu)/sd*imp
tree=cKDTree(Ztr); d_grid,_=tree.query(Zg,k=1); d_tr,_=tree.query(Ztr,k=2)
DI_g=d_grid/d_tr[:,1].mean(); DI_t=d_tr[:,1]/d_tr[:,1].mean()
q75,q25=np.percentile(DI_t,[75,25]); thr=q75+1.5*(q75-q25)
grid["in_aoa"]=DI_g<=thr
print(f"AOA inside: {grid['in_aoa'].mean()*100:.1f}%")

grid.to_file(OUT/"maps"/"tx_suitability_grid.gpkg", driver="GPKG")

# --- validate: facilities should now be HIGH relative suitability ---
gtree=cKDTree(np.c_[grid.geometry.x,grid.geometry.y])
_,gi=gtree.query(np.c_[pos.geometry.centroid.x,pos.geometry.centroid.y],k=1)
fac=grid["suitability"].values[gi]
print(f"Existing facilities median RELATIVE suitability: {np.median(fac):.2f} (grid median 0.50 by construction)")
print(f"  facilities in top 25%: {(fac>0.75).mean()*100:.0f}%  | top 10%: {(fac>0.90).mean()*100:.0f}%")

# --- render ---
cmap=LinearSegmentedColormap.from_list("suit",["#f0ead8","#e8c468","#c25b29","#7a2d12"])
fig,ax=plt.subplots(figsize=(11,11))
TX.boundary.plot(ax=ax,color="#3b3024",linewidth=1.2,zorder=3)
ins=grid[grid["in_aoa"]]; out=grid[~grid["in_aoa"]]
ins.plot(ax=ax,column="suitability",cmap=cmap,markersize=6,vmin=0,vmax=1,zorder=2)
out.plot(ax=ax,color="#cccccc",markersize=4,alpha=0.5,zorder=1)
pc=pos.copy(); pc["geometry"]=pc.geometry.centroid
pc.plot(ax=ax,color="#1a1a1a",markersize=5,marker="^",zorder=4)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1)); cb=fig.colorbar(sm,ax=ax,fraction=0.035,pad=0.02)
cb.set_label("Relative siting suitability (percentile)")
ax.set_title("Texas utility-scale solar siting suitability (Random Forest, spatial-CV ROC-AUC 0.92)\n"
             "relative percentile; grey = outside Area of Applicability; triangles = existing facilities",fontsize=11)
ax.set_axis_off(); plt.tight_layout()
plt.savefig(OUT/"figures"/"tx_suitability_map.png",dpi=160,bbox_inches="tight")
print("\nSaved corrected tx_suitability_map.png")
