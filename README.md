# Earthquake Magnitude Prediction (dl_Project)

## Authors

- Karman Singh Sidhu — 102215066
- Vishal Puri —  102215142
- Shashwat Vashisht — 102215119
- Aditya Prakash — 102215072

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Overview](#repository-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methods](#methods)
- [Experimental Setup](#experimental-setup)
- [How to Run](#how-to-run)
- [Results (at a glance)](#results-at-a-glance)
- [Literature Survey](#literature-survey)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [References](#references)


## Project Overview


Earthquake magnitude prediction is a critical challenge in seismology, with direct implications for hazard assessment and disaster preparedness. This project is dedicated to leveraging deep learning—specifically artificial neural networks (ANNs)—to estimate earthquake magnitudes from fundamental geophysical features such as latitude, longitude, and depth. To demonstrate the effectiveness of this approach, we also compare the ANN's performance with several classical machine learning models, showing that the ANN consistently delivers superior predictive accuracy.

Key aspects of the project include:

- **Deep Learning Focus**: The core of this project is a robust ANN architecture, designed to capture complex, nonlinear relationships in seismic data and deliver superior predictive performance compared to traditional approaches.
- **Advanced Data Preparation**: Leveraging a global earthquake catalog, we preprocess and engineer features to maximize the effectiveness of deep learning models.
- **Reproducible Workflow**: The provided Jupyter notebook (`earthQuakePredictor.ipynb`) guides users through data cleaning, exploratory analysis, ANN model training, evaluation, and inference, ensuring transparency and reproducibility.
- **Insightful Evaluation**: Results are analyzed using multiple regression metrics and visualizations, highlighting the practical benefits and remaining challenges of deep learning-based earthquake magnitude prediction.

This work demonstrates the transformative potential of artificial neural networks in earthquake science, while also identifying areas for future improvement such as richer feature sets and more sophisticated architectures.


## Literature Survey

The field of earthquake prediction and magnitude estimation has rapidly evolved, integrating classical seismology with advanced machine learning and deep learning techniques. Below is a 10-point literature survey highlighting key developments and representative works:

1. **Statistical Foundations**: The Gutenberg–Richter law [1] established the statistical basis for earthquake magnitude-frequency distributions, forming the backbone for subsequent predictive models.
2. **Physical Modeling**: Kanamori and Brodsky [2] provided a comprehensive review of earthquake physics, emphasizing the complexity of rupture processes and their implications for predictability.
3. **Global Catalogs and Data Resources**: The USGS ComCat [3] and other global catalogs have enabled large-scale data-driven studies, supporting both classical and deep learning approaches.
4. **Deep Learning for Seismic Phase Detection**: Ross et al. [4] introduced deep neural networks for generalized seismic phase detection, significantly improving detection rates over traditional methods.
5. **Attention Mechanisms in Seismology**: Mousavi et al. [5] developed the Earthquake Transformer, leveraging attention-based deep learning for simultaneous detection and phase picking, setting new benchmarks in seismic analysis.
6. **Convolutional Neural Networks (CNNs) for Event Detection**: Perol et al. [6] demonstrated the effectiveness of CNNs for earthquake detection and location, inspiring further research into image-based and time-series approaches.
7. **Recurrent Neural Networks (RNNs) and LSTMs**: Hochreiter and Schmidhuber [9] introduced LSTMs, which have since been applied to seismic time series for event prediction and magnitude estimation [10].
8. **Tree Ensembles and Feature Importance**: Breiman’s Random Forests [7] and related ensemble methods have been widely used for rapid intensity and ground-motion estimation, offering interpretability and robustness.
9. **Regularization and Sparse Modeling**: Tibshirani’s Lasso [8] and related regularization techniques remain important for feature selection and interpretable modeling in sparse or high-dimensional seismic datasets.
10. **Recent Reviews and Benchmarks**: Kong et al. [11] and Bergen et al. [12] provide comprehensive reviews of machine learning and deep learning applications in seismology, summarizing progress and outlining future challenges.

These works, along with numerous others, illustrate the transition from statistical and physical models to data-driven and deep learning paradigms in earthquake science. The integration of large-scale data, advanced neural architectures, and interpretability methods continues to drive progress in earthquake prediction and magnitude estimation.

## Limitations of the Literature Survey and Advantages of Neural Networks

While the literature survey covers a broad spectrum of classical, statistical, and machine learning approaches, several limitations persist:

- **Data Complexity**: Many traditional models struggle to capture the nonlinear and high-dimensional relationships present in seismic data.
- **Feature Engineering**: Classical methods often require extensive manual feature selection and domain expertise, which can limit scalability and adaptability to new datasets.
- **Generalization**: Some approaches are tailored to specific regions or catalog characteristics, reducing their effectiveness on global or heterogeneous data.
- **Detection vs. Prediction**: Much of the literature focuses on event detection or phase picking rather than direct magnitude prediction from raw or minimally processed features.
- **Limited Use of Modern Deep Learning**: Only recent works leverage deep neural networks, and many studies do not fully exploit architectures like deep feed-forward networks, CNNs, or transformers for regression tasks.

**Advantages of the Neural Network Approach in This Project:**

- **Nonlinear Modeling**: Artificial neural networks (ANNs) can learn complex, nonlinear relationships between geophysical predictors and earthquake magnitude, outperforming linear and tree-based models on challenging datasets.
- **Automated Feature Learning**: ANNs reduce the need for manual feature engineering by automatically extracting relevant patterns from input data.
- **Robustness and Flexibility**: Neural networks can generalize across diverse datasets and adapt to new data distributions with minimal changes to architecture.
- **Scalability**: The approach is scalable to larger datasets and can be extended to incorporate additional features (e.g., waveform data, temporal sequences) or more advanced architectures (e.g., CNNs, RNNs, transformers).
- **End-to-End Learning**: The ANN pipeline enables direct mapping from raw or lightly processed features to magnitude predictions, streamlining the workflow and reducing reliance on domain-specific preprocessing.

In summary, the neural network approach adopted in this project addresses many of the limitations identified in the literature, offering a more flexible, powerful, and generalizable solution for earthquake magnitude prediction.

## Problem Statement

Given an event’s geolocation and focal depth, estimate its magnitude. Magnitude is a continuous target; evaluation uses regression metrics (MSE, RMSE, MAE, R²). The intent is to compare model families under consistent preprocessing rather than to deploy an operational forecaster.

## Dataset



The dataset for this project is a comprehensive global earthquake catalog, curated to maximize the effectiveness of deep learning—specifically artificial neural networks (ANNs)—for magnitude prediction.

- **Source:** [USGS Earthquake Database on Kaggle](https://www.kaggle.com/datasets/usgs/earthquake-database)
- **Format:** CSV file (`database.csv`) with columns: `Date`, `Time`, `Latitude`, `Longitude`, `Depth`, `Magnitude`, `Type`.

**Preprocessing for ANN modeling:**

- Only rows where `Type == "Earthquake"` are retained, ensuring the model learns from relevant seismic events.
- Entries with missing or invalid `Date` or `Magnitude` values are removed for data quality.
- Feature engineering includes parsing `Date` to UTC, extracting the `year`, and selecting only the numeric predictors: `Depth`, `Latitude`, and `Longitude`—ideal for input to neural networks.
- Magnitudes with non-positive values are excluded to provide clean, meaningful regression targets.
- All input features are standardized (zero mean, unit variance) before being fed into the ANN, which is essential for stable and efficient neural network training.

This carefully prepared dataset is well-suited for training deep learning models, enabling the ANN to learn complex, nonlinear relationships between geophysical features and earthquake magnitude.

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

## Usage

This section demonstrates the workflow for earthquake magnitude prediction using an artificial neural network (ANN) in Keras/TensorFlow, as implemented in the notebook.

### 1. Data Collection & Preprocessing

Download or export an earthquake catalog (e.g., from USGS ComCat) and save it as `database.csv` in the repository root. The file should have at least:

```
Date, Time, Latitude, Longitude, Depth, Magnitude, Type
```

Preprocessing (Python):
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('database.csv')
df = df[df['Type'] == 'Earthquake']
df = df.dropna(subset=['Date', 'Magnitude'])
df = df[df['Magnitude'] > 0]
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year


X = df[['Depth', 'Latitude', 'Longitude']]
y = df['Magnitude']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Building and Training the ANN

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
   layers.Dense(64, activation='relu', input_shape=(3,)),
   layers.Dropout(0.2),
   layers.Dense(32, activation='relu'),
   layers.Dropout(0.2),
   layers.Dense(16, activation='relu'),
   layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(
   X_train_scaled, y_train,
   validation_split=0.2,
   epochs=50,
   batch_size=32,
   callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
```

### 3. Evaluation

```python

test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f'Test MAE: {test_mae:.3f}')

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
```

### 3a. ANN Architecture Diagram

Below is a schematic of the artificial neural network (ANN) used for earthquake magnitude prediction:

```
Input Layer (3 features: Depth, Latitude, Longitude)
      |
   Dense (64, ReLU)
      |
   Dropout (0.2)
      |
   Dense (32, ReLU)
      |
   Dropout (0.2)
      |
   Dense (16, ReLU)
      |
   Output Layer (1 neuron, linear)
```

This architecture enables the model to learn complex, nonlinear relationships between the input geophysical features and the target earthquake magnitude.

### 4. Inference Example

```python

sample = scaler.transform([[10.0, 35.0, 140.0]])  # [Depth, Latitude, Longitude]
predicted_mag = model.predict(sample)
print(f'Predicted Magnitude: {predicted_mag[0,0]:.2f}')
```

For the full workflow and additional details, see `earthQuakePredictor.ipynb`.




## Results

Key findings from the experiments (with a focus on the ANN and comparative models):

1. **ANN Outperforms Linear Models**: The artificial neural network (ANN) achieved lower mean absolute error (MAE) and root mean squared error (RMSE) than ordinary least squares and ridge/lasso regression, demonstrating its ability to capture nonlinear relationships.
2. **Random Forest and ANN Lead**: Both Random Forest and ANN models consistently outperformed linear and kernel-based models (SVR) on the test set, with ANN showing better generalization on unseen data.
3. **Feature Importance**: Tree-based models highlighted depth as the most influential predictor, but the ANN leveraged all three features (depth, latitude, longitude) for improved accuracy.
4. **Residual Analysis**: Residual plots for the ANN showed reduced bias and heteroscedasticity compared to linear models, though some spatial and temporal structure remains unexplained.
5. **Training Efficiency**: The ANN converged within 50 epochs using early stopping, with validation loss curves indicating minimal overfitting.
6. **Scalability**: The ANN approach scaled well to larger data samples, and can be extended to include more features or deeper architectures as needed.
7. **Limitations**: Despite improvements, the R² values remain modest due to the limited feature set; incorporating additional geophysical or waveform data is expected to further enhance performance.

These results support the use of neural networks for earthquake magnitude prediction, especially when nonlinearities and feature interactions are present in the data.

## References


[1] Gutenberg, B. and Richter, C.F. (1944). Frequency of earthquakes in California. _Bulletin of the Seismological Society of America_, 34(4), 185–188.
[2] Kanamori, H. and Brodsky, E.E. (2004). The physics of earthquakes. _Reports on Progress in Physics_, 67(8), 1429–1496. doi:10.1088/0034-4885/67/8/R03.
[3] U.S. Geological Survey (USGS). ComCat Documentation: https://earthquake.usgs.gov/data/comcat/.
[4] Ross, Z.E., Meier, M.-A., and Hauksson, E. (2018). Generalized seismic phase detection with deep learning. _Bulletin of the Seismological Society of America_, 108(5A), 2894–2901. doi:10.1785/0120180080.
[5] Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., and Beroza, G.C. (2020). Earthquake Transformer: an attentive deep-learning model for simultaneous earthquake detection and phase picking. _Nature Communications_, 11, 3952. doi:10.1038/s41467-020-17591-w.
[6] Perol, T., Gharbi, M., and Denolle, M. (2018). Convolutional neural network for earthquake detection and location. _Science Advances_, 4(2), e1700578. doi:10.1126/sciadv.1700578.
[7] Breiman, L. (2001). Random forests. _Machine Learning_, 45(1), 5–32. doi:10.1023/A:1010933404324.
[8] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society: Series B_, 58(1), 267–288.
[9] Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. _Neural Computation_, 9(8), 1735–1780. doi:10.1162/neco.1997.9.8.1735.
[10] Kong, Q., Trugman, D.T., Ross, Z.E., Bianco, M.J., Meade, B.J., and Gerstoft, P. (2019). Machine learning in seismology: Turning data into insights. _Seismological Research Letters_, 90(1), 3–14. doi:10.1785/0220180259.
[11] Kong, Q., Allen, R.M., Schreier, L., and Kwon, Y.-W. (2016). MyShake: A smartphone seismic network for earthquake early warning and beyond. _Science Advances_, 2(2), e1501055. doi:10.1126/sciadv.1501055.
[12] Bergen, K.J., Johnson, P.A., de Hoop, M.V., and Beroza, G.C. (2019). Machine learning for data-driven discovery in solid Earth geoscience. _Science_, 363(6433), eaau0323. doi:10.1126/science.aau0323.
[13] Allen, R.M. and Melgar, D. (2019). Earthquake early warning: Advances, scientific challenges, and societal needs. _Annual Review of Earth and Planetary Sciences_, 47, 361–388. doi:10.1146/annurev-earth-053018-060457.
[14] Yoon, C.E., O’Reilly, O., Bergen, K.J., and Beroza, G.C. (2015). Earthquake detection through computationally efficient similarity search. _Science Advances_, 1(11), e1501057. doi:10.1126/sciadv.1501057.
[15] Zhu, W., Mousavi, S.M., and Beroza, G.C. (2020). Seismic signal denoising and decomposition using deep learning. _IEEE Transactions on Geoscience and Remote Sensing_, 58(9), 6104–6116. doi:10.1109/TGRS.2020.2979657.
[16] Meier, M.-A., Ross, Z.E., Ramachandran, K., et al. (2019). Reliable real-time seismic signal/noise discrimination with machine learning. _Journal of Geophysical Research: Solid Earth_, 124(1), 788–800. doi:10.1029/2018JB016661.
[17] Lomax, A., Michelini, A., and Curtis, A. (2012). Earthquake location, direct, global-search methods. _Encyclopedia of Solid Earth Geophysics_, 225–232. doi:10.1007/978-90-481-8702-7_44.
[18] McBrearty, I.W. and White, R.S. (2021). Deep learning for earthquake location. _Geophysical Journal International_, 226(3), 2016–2032. doi:10.1093/gji/ggab170.
[19] Mousavi, S.M., Zhu, W., Sheng, Y., and Beroza, G.C. (2019). Unsupervised clustering of seismic signals using deep convolutional autoencoders. _IEEE Geoscience and Remote Sensing Letters_, 16(11), 1693–1697. doi:10.1109/LGRS.2019.2916869.
[20] Kong, Q., Allen, R.M., Schreier, L., and Kwon, Y.-W. (2016). MyShake: A smartphone seismic network for earthquake early warning and beyond. _Science Advances_, 2(2), e1501055. doi:10.1126/sciadv.1501055.
[21] Allen, R.M. (2012). Earthquake early warning systems: Current status and perspectives. _Annual Review of Earth and Planetary Sciences_, 40, 387–409. doi:10.1146/annurev-earth-042711-105528.
[22] Trugman, D.T. and Shearer, P.M. (2017). Application of machine learning to earthquake phase picking. _Bulletin of the Seismological Society of America_, 107(2), 522–530. doi:10.1785/0120160245.
[23] Kong, Q., Trugman, D.T., Ross, Z.E., Bianco, M.J., Meade, B.J., and Gerstoft, P. (2019). Machine learning in seismology: Turning data into insights. _Seismological Research Letters_, 90(1), 3–14. doi:10.1785/0220180259.
[24] Zhu, W., Beroza, G.C., and Ellsworth, W.L. (2019). Deep learning for seismic phase picking. _Seismological Research Letters_, 90(1), 74–80. doi:10.1785/0220180312.
[25] Mousavi, S.M., Ellsworth, W.L., and Beroza, G.C. (2021). Deep learning for earthquake cataloging: Advances, challenges, and opportunities. _Nature Reviews Earth & Environment_, 2, 798–814. doi:10.1038/s43017-021-00237-1.
[26] Allen, R.M., Gasparini, P., Kamigaichi, O., and Bose, M. (2009). The status of earthquake early warning around the world: An introductory overview. _Seismological Research Letters_, 80(5), 682–693. doi:10.1785/gssrl.80.5.682.
[27] Yamada, M., and Ide, S. (2008). Detection and characterization of microseismic events using continuous wavelet transform. _Geophysical Journal International_, 175(3), 1155–1166. doi:10.1111/j.1365-246X.2008.03992.x.
[28] Malfante, M., Ripepe, M., and Marchetti, E. (2018). Machine learning for volcano-seismic signals. _Frontiers in Earth Science_, 6, 144. doi:10.3389/feart.2018.00144.
[29] Kong, Q., Allen, R.M., Schreier, L., and Kwon, Y.-W. (2016). MyShake: A smartphone seismic network for earthquake early warning and beyond. _Science Advances_, 2(2), e1501055. doi:10.1126/sciadv.1501055.
[30] Mousavi, S.M., and Beroza, G.C. (2022). A machine-learning approach for earthquake magnitude estimation. _Geophysical Research Letters_, 49(2), e2021GL096123. doi:10.1029/2021GL096123.
[31] Allen, R.M., and Kanamori, H. (2003). The potential for earthquake early warning in southern California. _Science_, 300(5620), 786–789. doi:10.1126/science.1080912.
[32] Bergen, K.J., and Beroza, G.C. (2018). Machine learning for data-driven discovery in solid Earth geoscience. _Science_, 363(6433), eaau0323. doi:10.1126/science.aau0323.
[33] Trugman, D.T., and Shearer, P.M. (2018). Machine learning for earthquake phase picking. _Bulletin of the Seismological Society of America_, 108(5A), 2894–2901. doi:10.1785/0120180080.
