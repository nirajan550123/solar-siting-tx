"""
26_ghi_importance_validation.py
Confirm GHI's low importance is real, not a cardinality/resolution/variance
artifact. Three tests (Strobl et al. 2007; Kursa & Rudnicki 2010; Altmann et al. 2010):
  1. Permutation importance (cardinality-fair) vs impurity (cardinality-biased) - side by side
  2. Boruta shadow-feature test (Confirmed/Tentative/Rejected)
  3. Direct random-noise feature benchmark (does GHI beat pure noise?)
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings; warnings.filterwarnings("ignore")

FEATS = ["ghi_nsrdb","slope_pct","elevation","lc_majority","dist_transmission","dist_substation","dist_road"]
NICE = {"ghi_nsrdb":"GHI","slope_pct":"slope","elevation":"elevation","lc_majority":"land_cover",
        "dist_transmission":"dist_transmission","dist_substation":"dist_substation","dist_road":"dist_road"}

tx = gpd.read_file("data/processed/model_table.gpkg")
pres = tx[tx.label==1]; pa = tx[tx.label==0]

# use replicate 0 balanced set + add a random noise feature
sub = pd.concat([pres, pa[pa.replicate==0]]).reset_index(drop=True)
rng = np.random.default_rng(42)
sub["random_noise"] = rng.normal(size=len(sub))
ALL = FEATS + ["random_noise"]
X = sub[ALL].values; y = sub["label"].values

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(X, y)

# --- 1. impurity (biased) vs permutation (fair) ---
imp_gini = rf.feature_importances_
perm = permutation_importance(rf, X, y, n_repeats=30, random_state=0, n_jobs=-1)
imp_perm = perm.importances_mean
df = pd.DataFrame({"feature":[NICE.get(f,f) for f in ALL],
                   "impurity":imp_gini, "permutation":imp_perm,
                   "perm_std":perm.importances_std}).sort_values("permutation", ascending=False)
print("=== IMPURITY (cardinality-biased) vs PERMUTATION (fair) ===")
print(df.to_string(index=False))

noise_perm = df[df.feature=="random_noise"]["permutation"].iloc[0]
ghi_perm   = df[df.feature=="GHI"]["permutation"].iloc[0]
print(f"\nGHI permutation importance: {ghi_perm:.5f}")
print(f"Random-noise permutation importance: {noise_perm:.5f}")
print(f"GHI {'BEATS' if ghi_perm>noise_perm else 'DOES NOT BEAT'} pure noise")

# --- 2. Boruta ---
print("\n=== BORUTA (shadow-feature test) ===")
try:
    from boruta import BorutaPy
    rf_b = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    bor = BorutaPy(rf_b, n_estimators="auto", random_state=0, max_iter=80, verbose=0)
    bor.fit(X, y)
    res = pd.DataFrame({"feature":[NICE.get(f,f) for f in ALL],
                        "confirmed":bor.support_, "tentative":bor.support_weak_,
                        "rank":bor.ranking_}).sort_values("rank")
    print(res.to_string(index=False))
    ghi_status = "CONFIRMED" if res[res.feature=="GHI"]["confirmed"].iloc[0] else ("TENTATIVE" if res[res.feature=="GHI"]["tentative"].iloc[0] else "REJECTED")
    print(f"\nGHI Boruta status: {ghi_status}")
except Exception as e:
    print("Boruta error:", e)
