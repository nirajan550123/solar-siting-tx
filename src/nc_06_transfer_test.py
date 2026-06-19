"""
nc_06_transfer_test.py
THE TRANSFER TEST: train RF on ALL Texas data, apply to the NC eval set, measure discrimination.
Compares NC transfer performance to the within-Texas spatial-CV baseline (0.92).
Also reports the no-transmission transfer (robustness) and NC suitability-rank validation.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
FEATS_NOTX = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_substation","dist_road"]

tx = gpd.read_file("data/processed/model_table.gpkg")
nc = gpd.read_file("data/processed/nc_eval_table.gpkg")

def transfer(feats, tag):
    rf = RandomForestClassifier(n_estimators=800, class_weight="balanced_subsample",
                                random_state=0, n_jobs=-1).fit(tx[feats].values, tx["label"].values)
    p = rf.predict_proba(nc[feats].values)[:,1]
    roc = roc_auc_score(nc["label"].values, p)
    pr  = average_precision_score(nc["label"].values, p)
    print(f"{tag}: ROC-AUC {roc:.3f}  PR-AUC {pr:.3f}")
    return roc, pr, p

print("=== TX -> NC TRANSFER TEST ===")
print("(TX within-state spatial-CV baseline was ROC 0.922)\n")
roc_full, pr_full, p_full = transfer(FEATS, "Full model (7 features)")
roc_notx, pr_notx, _      = transfer(FEATS_NOTX, "No-transmission (robustness)")

# rank validation: do NC presences score higher than NC pseudo-absences?
nc["score"] = p_full
pres_med = nc[nc.label==1]["score"].median()
pa_med   = nc[nc.label==0]["score"].median()
print(f"\nNC presence median score {pres_med:.3f} vs pseudo-abs {pa_med:.3f}")

print("\n=== INTERPRETATION ===")
drop = 0.922 - roc_full
print(f"Transfer ROC {roc_full:.3f} vs TX baseline 0.922 -> degradation {drop:+.3f}")
if roc_full >= 0.85:
    print("STRONG transfer: grid-access siting logic generalizes across regimes.")
elif roc_full >= 0.70:
    print("PARTIAL transfer: same logic, regime-shifted thresholds reduce accuracy (predicted).")
else:
    print("WEAK transfer: siting is strongly regime-specific.")
