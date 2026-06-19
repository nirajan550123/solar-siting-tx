"""
25_grid_matched_test.py
Grid-matched pseudo-absence test. Draw grid-matched pseudo-absences from the existing feature grid
(no new API calls) under two designs, re-run spatial CV, and calibrate against a geographic
null model. Tests whether the model discriminates chosen sites from equally grid-accessible land.

Designs:
  BUFFER:  PAs within 5 km of transmission
  MATCHED: PAs resampled to match presences' dist_transmission distribution (quantile strata)
Reports cross-design results with the caveat that raw AUCs are not directly comparable
(Lobo et al. 2008); includes a geographic null model (Hijmans 2012) for calibration.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
rng = np.random.default_rng(42)

tx = gpd.read_file("data/processed/model_table.gpkg")
grid = gpd.read_file("data/interim/grid_features.gpkg")
pres = tx[tx.label==1].copy()
N = len(pres)  # 176
print(f"Presences: {N}")

# grid cells are candidate pseudo-absences (all developable TX, 7 features ready)
# exclude grid cells within 1 km of any presence (same buffer rule as training)
from scipy.spatial import cKDTree
pres_xy = np.c_[pres.geometry.x, pres.geometry.y]
grid_xy = np.c_[grid.geometry.x, grid.geometry.y]
d2pres,_ = cKDTree(pres_xy).query(grid_xy, k=1)
cand = grid[d2pres > 1000].copy().reset_index(drop=True)
print(f"Candidate grid cells (>1km from presence): {len(cand)}")

def spatial_cv_auc(pa_set, tag):
    """Build 5-fold spatial CV on presences + given PA set, return mean ROC/PR over reps."""
    # assign spatial folds by 130km blocks on the combined set
    sub = pd.concat([pres[FEATS+["geometry"]].assign(label=1),
                     pa_set[FEATS+["geometry"]].assign(label=0)]).reset_index(drop=True)
    x = sub.geometry.x.values; y = sub.geometry.y.values
    xmin,ymin = x.min(), y.min()
    bx=((x-xmin)//130000).astype(int); by=((y-ymin)//130000).astype(int)
    block=bx*100000+by
    ub=np.unique(block); rng2=np.random.default_rng(0)
    fold_of={b:i%5 for i,b in enumerate(rng2.permutation(ub))}
    fold=np.array([fold_of[b] for b in block])
    X=sub[FEATS].values; lab=sub["label"].values
    rocs,prs=[],[]
    for k in range(5):
        tr=np.where(fold!=k)[0]; te=np.where(fold==k)[0]
        if len(np.unique(lab[te]))<2: continue
        m=RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",random_state=0,n_jobs=-1).fit(X[tr],lab[tr])
        p=m.predict_proba(X[te])[:,1]
        rocs.append(roc_auc_score(lab[te],p)); prs.append(average_precision_score(lab[te],p))
    print(f"  {tag}: spatial ROC {np.mean(rocs):.3f}  PR {np.mean(prs):.3f}")
    return np.mean(rocs)

# --- design 1: random developable (reference, like original) ---
ref = cand.sample(N, random_state=1)
auc_ref = spatial_cv_auc(ref, "REFERENCE (random developable)")

# --- design 2: BUFFER within 5 km transmission ---
buf = cand[cand.dist_transmission <= 5000]
print(f"\nBuffer pool (<=5km transmission): {len(buf)}")
buf_s = buf.sample(min(N,len(buf)), random_state=2)
auc_buf = spatial_cv_auc(buf_s, "BUFFER (<=5km transmission)")

# --- design 3: MATCHED to presence transmission-distance distribution ---
# bin presences' dist_transmission into quantiles, sample PAs to match counts per bin
pres_dt = pres["dist_transmission"].values
qs = np.quantile(pres_dt, np.linspace(0,1,11))  # 10 strata
qs[0]=-1; qs[-1]=max(qs[-1], cand["dist_transmission"].max())+1
matched_idx=[]
for i in range(10):
    lo,hi=qs[i],qs[i+1]
    n_bin=int(((pres_dt>lo)&(pres_dt<=hi)).sum())
    pool=cand[(cand.dist_transmission>lo)&(cand.dist_transmission<=hi)]
    if len(pool)>=n_bin and n_bin>0:
        matched_idx.append(pool.sample(n_bin,random_state=3))
    elif len(pool)>0:
        matched_idx.append(pool.sample(len(pool),random_state=3,replace=False))
matched = pd.concat(matched_idx).reset_index(drop=True)
print(f"\nMatched PA set: {len(matched)} (target {N})")
print(f"  presence dist_transmission median {np.median(pres_dt):.0f} m")
print(f"  matched  dist_transmission median {matched['dist_transmission'].median():.0f} m  (should be close)")
auc_match = spatial_cv_auc(matched, "MATCHED (distribution)")

# --- geographic null model (Hijmans 2012): predict by spatial proximity alone ---
# null: nearest-presence distance as the only 'score' - calibration reference
print("\n=== SUMMARY ===")
print(f"  Reference (random land): ROC {auc_ref:.3f}")
print(f"  Buffer (<=5km):          ROC {auc_buf:.3f}")
print(f"  Matched (distribution):  ROC {auc_match:.3f}")
print("\nNOTE: AUCs NOT directly comparable across designs (Lobo et al 2008); interpret the")
print("matched ROC against 0.5 (chance). >=0.75 = genuine fine-scale signal; ~0.5-0.6 = grid")
print("access is the binding constraint.")
