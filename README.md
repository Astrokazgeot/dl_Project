# Earthquake Magnitude Prediction (dl_Project)

## Abstract

This project evaluates classical and deep learning regressors to predict earthquake magnitudes from basic geophysical attributes (latitude, longitude, depth, event time). We clean a historical catalog, visualize temporal and spatial trends, and benchmark linear, regularized, ensemble, kernel, tree-based, and neural models. The accompanying notebook (`earthQuakePredictor.ipynb`) serves as a reproducible workflow for data preparation, exploratory analysis, model training, and evaluation.

## Repository Overview

- `earthQuakePredictor.ipynb`: end-to-end analysis and modeling notebook.
- `database.csv` (expected): earthquake catalog with columns `Date, Time, Latitude, Longitude, Depth, Magnitude, Type`.

## Problem Statement

Given an event’s geolocation and focal depth, estimate its magnitude. Magnitude is a continuous target; evaluation uses regression metrics (MSE, RMSE, MAE, R²). The intent is to compare model families under consistent preprocessing rather than to deploy an operational forecaster.

## Dataset

- Source: Provide/locate an earthquake catalog as `database.csv` (e.g., USGS ComCat export) containing at least the columns above.
- Scope in notebook: rows filtered to `Type == "Earthquake"`; rows with missing `date` or `magnitudo` dropped after parsing mixed date formats.
- Feature engineering: parse `date` to UTC, extract `year`, and retain numeric predictors `depth`, `latitude`, `longitude`. Magnitudes with non-positive values can be optionally excluded for cleaner regression targets.

## Methods

1. **Exploratory analysis**: summary stats, null inspection, event counts, temporal trend plots, average magnitude per year, and global spatial scatter using Basemap.
2. **Models compared** (scikit-learn unless noted):
   - Ordinary Least Squares (with and without non-positive magnitudes).
   - Ridge and Lasso regression for coefficient shrinkage.
   - Decision Tree Regressor (depth-limited) and Random Forest Regressor (sampled subset for speed).
   - Support Vector Regression with RBF kernel (scaled features, sampled subset).
   - Feed-forward Artificial Neural Network (Keras Sequential: 3 hidden layers with ReLU and dropout, early stopping).
3. **Sampling for efficiency**: RF uses ~10% of data, SVR ~1%, ANN ~5% to keep training tractable in a notebook context.
4. **Evaluation**: MSE, RMSE, MAE, and R² printed per model; diagnostic plots include predicted vs. actual and residual histograms. Tree/forest feature importances plotted; ANN training and validation loss curves tracked.

## Experimental Setup

- Train/test split: 75/25 with fixed `random_state=42` for reproducibility (per model cell).
- Features: `[depth, latitude, longitude]`; target: `magnitudo`.
- Scaling: StandardScaler applied for SVR and ANN.
- Example inference: final cell demonstrates single-event prediction using the trained decision tree model.

## How to Run

1. Ensure Python 3.9+ and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn basemap scikit-learn tensorflow
   ```
   Basemap may require system-level geospatial libraries; consult platform-specific install instructions.
2. Place `database.csv` in the repository root with the expected columns.
3. Open `earthQuakePredictor.ipynb` and run cells sequentially, or execute selected sections for specific models.

## Results (at a glance)

Metrics are computed at runtime; expect modest R² given limited feature richness. Ensemble (Random Forest) and ANN often outperform linear baselines on non-linear relationships, while SVR performance depends on kernel bandwidth and sampling. Residual plots highlight heteroscedasticity and remaining spatial-temporal structure not captured by the simple feature set.

## Literature Survey

Earthquake magnitude and occurrence modeling has evolved from statistical seismology to data-driven learning. Gutenberg–Richter magnitude-frequency relationships remain foundational for understanding event distributions [1]. Physical perspectives on rupture dynamics frame what signal features can inform predictability [2]. Modern catalogs (e.g., USGS ComCat) provide global event metadata that enable machine learning pipelines [3]. Deep neural approaches have demonstrated gains in detection and phase picking (e.g., generalized phase detection [4] and attention-based transformers [5]), while convolutional and recurrent models have been used for event characterization [6]. Tree ensembles have been applied for rapid intensity and ground-motion estimation leveraging mixed features [7]. Regularization methods such as ridge and lasso continue to be useful baselines and for interpretability in sparse-feature contexts [8]. This project positions a lightweight feature set within that spectrum, contrasting classical and neural regressors on catalog-level attributes.

## Limitations and Next Steps

- Only three spatial/depth predictors are used; incorporating focal mechanisms, station-derived features, or waveform data should improve accuracy.
- No temporal cross-validation; future work could evaluate robustness to time shifts.
- Hyperparameters are minimally tuned; systematic search (e.g., randomized search) could yield stronger baselines.
- Operational forecasting requires uncertainty quantification and rigorous prospective testing, which are out of scope here.

## Authors

- Karman Singh Sidhu — Roll No. 102215066
- Vishal Puri — Roll No. 102215142
- Shashwat Vashisht — Roll No. 102215119
- Aditya Prakash — Roll No. 102215072

## References

[1] Gutenberg, B. and Richter, C.F. (1944). Frequency of earthquakes in California. _Bulletin of the Seismological Society of America_, 34(4), 185–188.

[2] Kanamori, H. and Brodsky, E.E. (2004). The physics of earthquakes. _Reports on Progress in Physics_, 67(8), 1429–1496. doi:10.1088/0034-4885/67/8/R03.

[3] U.S. Geological Survey (USGS). ComCat Documentation: https://earthquake.usgs.gov/data/comcat/.

[4] Ross, Z.E., Meier, M.-A., and Hauksson, E. (2018). Generalized seismic phase detection with deep learning. _Bulletin of the Seismological Society of America_, 108(5A), 2894–2901. doi:10.1785/0120180080.

[5] Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., and Beroza, G.C. (2020). Earthquake Transformer: an attentive deep-learning model for simultaneous earthquake detection and phase picking. _Nature Communications_, 11, 3952. doi:10.1038/s41467-020-17591-w.

[6] Perol, T., Gharbi, M., and Denolle, M. (2018). Convolutional neural network for earthquake detection and location. _Science Advances_, 4(2), e1700578. doi:10.1126/sciadv.1700578.

[7] Breiman, L. (2001). Random forests. _Machine Learning_, 45(1), 5–32. doi:10.1023/A:1010933404324.

[8] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society: Series B_, 58(1), 267–288.
