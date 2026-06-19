"""
19_cv_models.py
Headline CV: logistic / RF / XGBoost, each under RANDOM 5-fold vs SPATIAL 5-fold (130 km),
averaged across all 10 pseudo-absence replicates. Reports ROC-AUC and PR-AUC with the
random-vs-spatial gap (the key spatial-overfitting evidence).
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

m = gpd.read_file("data/processed/model_table.gpkg")
FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority",
         "dist_transmission","dist_substation","dist_road"]
pres = m[m["label"]==1].copy()
pa   = m[m["label"]==0].copy()
REPS = sorted(pa["replicate"].unique())
K = 5

def make_models():
    return {
        "Logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                               random_state=0, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, eval_metric="logloss",
                                 scale_pos_weight=1, random_state=0, n_jobs=-1),
    }

def evaluate(train_idx, test_idx, Xall, yall, name):
    Xtr, Xte = Xall[train_idx], Xall[test_idx]
    ytr, yte = yall[train_idx], yall[test_idx]
    if name == "Logistic":
        sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    mdl = make_models()[name]; mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:,1]
    if len(np.unique(yte)) < 2:
        return None, None
    return roc_auc_score(yte, p), average_precision_score(yte, p)

results = []
for name in ["Logistic","RandomForest","XGBoost"]:
    for scheme in ["random","spatial"]:
        roc_all, pr_all = [], []
        for rep in REPS:
            sub = pd.concat([pres, pa[pa["replicate"]==rep]]).reset_index(drop=True)
            X = sub[FEATS].values; y = sub["label"].values
            if scheme == "random":
                skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=rep)
                folds = list(skf.split(X, y))
            else:
                f = sub["fold_130"].values
                folds = [(np.where(f!=k)[0], np.where(f==k)[0]) for k in range(K)]
            for tr, te in folds:
                roc, pr = evaluate(tr, te, X, y, name)
                if roc is not None:
                    roc_all.append(roc); pr_all.append(pr)
        results.append({"model":name,"scheme":scheme,
                        "ROC_AUC":np.mean(roc_all),"ROC_sd":np.std(roc_all),
                        "PR_AUC":np.mean(pr_all),"PR_sd":np.std(pr_all)})

df = pd.DataFrame(results)
print("\n================ CROSS-VALIDATION RESULTS ================")
print("(averaged over 10 PA replicates x 5 folds)\n")
for name in ["Logistic","RandomForest","XGBoost"]:
    r = df[(df.model==name)&(df.scheme=="random")].iloc[0]
    s = df[(df.model==name)&(df.scheme=="spatial")].iloc[0]
    print(f"{name}")
    print(f"  ROC-AUC : random {r.ROC_AUC:.3f}  spatial {s.ROC_AUC:.3f}  gap {r.ROC_AUC-s.ROC_AUC:+.3f}")
    print(f"  PR-AUC  : random {r.PR_AUC:.3f}  spatial {s.PR_AUC:.3f}  gap {r.PR_AUC-s.PR_AUC:+.3f}")
    print()
df.to_csv("outputs/cv_results.csv", index=False)
print("Saved outputs/cv_results.csv")
