# Where Solar Gets Built: Utility-Scale Solar Siting in Texas

**A machine-learning analysis of utility-scale solar siting in Texas, with a North Carolina transfer test.**

Developers do not build solar farms where the sun is strongest. They build where the grid is reachable. This project shows that, quantifies it, and then demonstrates the rule is regime-specific by transferring the model to a state with the opposite policy environment.

![Texas solar siting suitability map](images/tx_suitability_map.png)

*Relative siting suitability across Texas. Warm corridors trace grid-accessible east and central Texas; grey areas fall outside the model's Area of Applicability. Triangles are existing facilities, which land overwhelmingly in the warm zones.*

## The headline

Using 176 operating utility-scale solar facilities across Texas, a Random Forest model trained on land, terrain, irradiance, and grid-access features finds:

- **Distance to transmission infrastructure is roughly an order of magnitude more important than solar irradiance** for predicting where solar actually gets sited (permutation importance ~9x; SHAP ~5x).
- **Irradiance is a real but minor predictor.** Across Texas's uniformly sunny range (4.5 to 5.87 kWh/m²/day), the marginal sunshine difference between one parcel and the next barely moves siting decisions.
- The model discriminates sited from unsited land at **spatial cross-validated ROC-AUC 0.92**.

Then the part that makes it more than a Texas story:

- **North Carolina built about four times as much solar as Texas despite lower irradiance.**
- The Texas-trained model transfers to North Carolina at **ROC-AUC 0.76 ± 0.01**, partial transfer. The siting logic generalizes in kind (grid access matters everywhere) but shifts in degree (what counts as "close enough" to the grid is set by each state's regulatory regime).

## Why this is interesting

Most solar-potential maps rank land by sunshine and slope. They describe *physical potential*. This project models *realized siting*: where developers actually chose to build, which turns out to be a different and more economically revealing question. The gap between the two is the contribution: irradiance barely matters, grid access dominates.

The transfer test sharpens it. Comparable siting studies rarely test whether their model generalizes to another state at all. By training on Texas and predicting North Carolina, this project shows the same siting logic holds in both states but is calibrated by each state’s policy regime, and the transfer test localizes where generalization breaks: at the state boundary.

## How it was built

**Problem framing.** Verified solar facilities give *presences* only; there is no such thing as a confirmed "never-solar" parcel. So the project borrows presence-background methodology from species distribution modeling (Barbet-Massin et al., 2012): real facilities versus pseudo-absences drawn from developable land, with 10 balanced replicate draws averaged for stability.

**Features.** Each facility is summarized over its actual footprint, not a single centroid point, so a 500 MW plant is represented by the land it really occupies (a change-of-support correction; Gotway & Young, 2002). Features span solar irradiance (NREL/NSRDB), slope and elevation (USGS 3DEP), land cover (NLCD), and distance to transmission lines, substations, and roads (HIFLD, TIGER).

**Models.** Random Forest (headline) and XGBoost, against a logistic-regression baseline. RF and XGBoost tie near ROC 0.92; logistic trails at 0.85.

**Validation** uses a full ladder rather than a single optimistic number, because siting features are spatially autocorrelated and naive random cross-validation inflates scores (Roberts et al., 2017):

| Validation | ROC-AUC | What it tests |
|------------|--------:|---------------|
| Random CV | ~0.93 | Interpolation (optimistic baseline) |
| Spatial block CV (130 km) | 0.92 | Spatial independence |
| Leave-one-region-out (EPA ecoregions) | 0.91 | Extrapolation to unseen Texas regions |
| Transfer to North Carolina | 0.76 | Cross-state, cross-regime extrapolation |

Block size was set from the data's own 12 km residual autocorrelation range.

## Robustness

The project stress-tests the obvious objections, each documented in the code with citations:

- **"Distance-to-transmission is circular."** With every grid feature removed and only interconnection-immune features left, the model still discriminates at ROC 0.72. About two-thirds of the signal survives, so grid access amplifies a real siting signal rather than being an artifact (Bellemare et al., 2017).
- **"Irradiance only looks unimportant because it's coarsely resolved."** Under the fair permutation metric it still ranks far below grid access, and a Boruta shadow-feature test confirms it is a genuine but minor predictor.
- **"The ranking could be one lucky run."** Across 10 replicate draws, transmission is the top feature in all 10 and substation second in all 10.
- **"The map claims applicability it doesn't have."** The Area of Applicability (94.8% of Texas) was validated with local data-point density (Meyer & Pebesma, 2021; Schumacher et al., 2025).

## Limitations

- Single training state and a modest sample (176 facilities). The North Carolina transfer probes generalization but does not replace multi-state training.
- The "irradiance is minor" claim is Texas-specific: within a narrow, uniformly high irradiance range there is little gradient for siting to track.
- Suitability outputs are relative percentiles, not calibrated probabilities (Guillera-Arroita et al., 2015).
- The panhandle (High Plains) is where the model extrapolates least well, reported transparently.

## Setup

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own NREL API key (free, from https://developer.nrel.gov/signup/) and Google Earth Engine project id. Earth Engine access requires `earthengine authenticate` once.

Scripts in `src/` are numbered in execution order; `nc_*` scripts run the North Carolina transfer arm. Raw data is not committed; sources are documented in each script's header.

## Selected references

Barbet-Massin, M., Jiguet, F., Albert, C. H., & Thuiller, W. (2012). Selecting pseudo-absences for species distribution models: How, where and how many? *Methods in Ecology and Evolution, 3*(2), 327-338.

Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution, 12*(9), 1620-1633.

Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913-929.

Wu, G. C., Min, Y., Deshmukh, R., et al. (2026). Factors shaping the siting of utility-scale solar and wind projects in the United States. *Environmental Research Letters, 21*(9), 094003.

---

*Built with Python, scikit-learn, GeoPandas, Google Earth Engine, and SHAP. Author: Nirajan Tripathi, M.S. Geography (GIS and remote sensing), Texas State University.*
