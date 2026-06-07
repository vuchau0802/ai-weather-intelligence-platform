# 🌍 Weather Trend Forecasting — Full Analysis Report

**PM Accelerator | Data Science Technical Assessment**
**Dataset: Global Weather Repository | 144,627 records | 211 Countries | 2024–2026**

---

## 🎯 PM Accelerator Mission

> *"Our mission is to break down financial barriers and achieve educational fairness. With the goal of establishing 200 schools worldwide over the next 20 years, we aim to empower more kids for a better future in their life and career, simultaneously fostering a diverse landscape in the tech industry."*

PM Accelerator — founded by Dr. Nancy Li — empowers aspiring and current Product Managers with certified programs, FAANG-level coaching, real-world project experience, and a global community of 1,500+ alumni. The organization also runs **PMA Kids**, a non-profit offering free Product Management education to teenagers from underserved families.

---

## 1. Executive Summary

This report presents a comprehensive analysis of global weather data, covering:

- **Data ingestion & cleaning** of a 144,627-row, 52-column dataset
- **Exploratory Data Analysis (EDA)** across temperature, precipitation, wind, and air quality dimensions
- **Six forecasting models** evaluated on daily temperature prediction (MAE, RMSE, R², MAPE)
- **Advanced analyses** including anomaly detection, climate patterns, air quality correlations, feature importance, and spatial/geographic breakdowns

**Best model: Random Forest** (RMSE = 1.051, R² = 0.823, MAPE = 4.16%)
**Best MAPE: Weighted Ensemble** (MAPE = 4.08%)
**Total runtime**: 53.4 seconds end-to-end

---

## 2. Dataset Description

### 2.1 Raw Dataset

| Attribute        | Value                           |
|------------------|---------------------------------|
| Source           | Global Weather Repository       |
| Rows             | 144,627                         |
| Columns          | 52                              |
| Countries        | 211                             |
| Cities           | 257                             |
| Start date       | 2024-05-16 01:45 UTC            |
| End date         | 2026-05-31 19:00 UTC            |

### 2.2 Key Features

**Meteorological:**
- `temperature_celsius` / `temperature_fahrenheit`
- `feels_like_celsius` / `feels_like_fahrenheit`
- `humidity` (%)
- `wind_mph` / `wind_kph`, `wind_degree`, `wind_direction`
- `pressure_mb` / `pressure_in`
- `precip_mm` / `precip_in`
- `visibility_km` / `visibility_miles`
- `cloud` (% cover)
- `uv_index`

**Air Quality:**
- `air_quality_Carbon_Monoxide`
- `air_quality_Ozone`
- `air_quality_Nitrogen_dioxide`
- `air_quality_Sulphur_dioxide`
- `air_quality_PM2.5`
- `air_quality_PM10`
- `air_quality_us-epa-index` (1–6 scale)
- `air_quality_gb-defra-index`

**Astronomical:**
- `sunrise`, `sunset`, `moonrise`, `moonset`
- `moon_phase`, `moon_illumination`

**Location:**
- `country`, `location_name`, `latitude`, `longitude`, `timezone`

---

## 3. Data Cleaning & Preprocessing

### 3.1 DateTime Parsing

The `last_updated` column was parsed into structured temporal features:

| Derived Feature | Description                       |
|-----------------|-----------------------------------|
| `year`          | 2024, 2025, 2026                  |
| `month`         | 1–12                              |
| `day`           | 1–31                              |
| `hour`          | 0–23                              |
| `dayofyear`     | 1–366                             |
| `week`          | ISO week number                   |
| `season`        | Spring / Summer / Autumn / Winter |

### 3.2 Missing Value Treatment

**Strategy:**
- Columns with ≥ 40% missing: **dropped entirely**
- Numeric columns: imputed with **median** (robust to remaining outliers)
- Categorical columns: imputed with **mode**

**Outcome:** 0 nulls remaining after imputation.

### 3.3 Outlier Treatment

IQR-based clipping applied to 10 key numeric columns:

| Column                  | Method | Action |
|-------------------------|--------|--------|
| `temperature_celsius`   | IQR    | Clip   |
| `wind_mph`              | IQR    | Clip   |
| `humidity`              | IQR    | Clip   |
| `pressure_mb`           | IQR    | Clip   |
| `precip_mm`             | IQR    | Clip   |
| `visibility_km`         | IQR    | Clip   |
| `feels_like_celsius`    | IQR    | Clip   |
| `dew_point_celsius`     | IQR    | Clip   |
| `windchill_celsius`     | IQR    | Clip   |
| `heat_index_celsius`    | IQR    | Clip   |

Values were **clipped** rather than removed to preserve time-series continuity.

### 3.4 Feature Engineering

Five new features were engineered:

**`temp_feels_diff`** = `temperature_celsius − feels_like_celsius`
Captures wind-chill and heat-index divergence from actual temperature.

**`discomfort_index`** = `wind_mph × humidity / 100`
A composite measure of perceived discomfort from wind and moisture.

**`heat_index_approx`** (Rothfusz NWS polynomial)
```
HI = −8.78 + 1.611T + 2.339H − 0.146TH − 0.0123T² − 0.0164H²
     + 0.00221T²H + 0.000725TH² − 0.00000358T²H²
```
where T = temperature_celsius, H = humidity (%).

**`pressure_category`** (5 bins):
Very Low (<980 mb) | Low (980–1000) | Normal (1000–1013) | High (1013–1030) | Very High (>1030)

**`visibility_category`** (5 bins):
Fog (<1 km) | Mist (1–5 km) | Moderate (5–10 km) | Good (10–50 km) | Excellent (>50 km)

### 3.5 Normalization

StandardScaler applied to all numeric features before ML model training. Year/month/day/hour excluded from scaling (used as categoricals).

**Final processed dataset:** 144,627 rows × 51 columns → saved to `outputs/processed_globalweather.csv`

---

## 4. Exploratory Data Analysis

### 4.1 Temperature Analysis
*Figure: `outputs/figures/01_temperature_analysis.png`*

- **Global distribution**: Approximately normal with slight right skew. Mean ≈ 18°C, std ≈ 12°C.
- **Bimodal hint**: reflects simultaneous Northern (summer) and Southern (winter) hemisphere readings.
- **Seasonal breakdown**: Summer mean ≈ 22°C; Winter mean ≈ 10°C globally.
- **Monthly trend**: Peaks July–August; troughs January–February.
- **Feels-like divergence**: Average |temp_feels_diff| ≈ 1.8°C; largest divergence in coastal/high-wind locations.

### 4.2 Precipitation Analysis
*Figure: `outputs/figures/02_precipitation_analysis.png`*

- **Zero-inflated distribution**: ~82% of records show 0 mm precipitation.
- **High-precip events**: Tail extends to 100+ mm/period in tropical zones.
- **Seasonal pattern**: Precipitation peaks in summer months across most regions.
- **High-precipitation countries**: Southeast Asian archipelagos and West African coast.

### 4.3 Correlation Analysis
*Figure: `outputs/figures/03_correlation_heatmap.png`*

**Strong positive correlations (r > 0.7):**
- `temperature_celsius` ↔ `feels_like_celsius` (r ≈ 0.99)
- `temperature_celsius` ↔ `dew_point_celsius` (r ≈ 0.83)
- `wind_mph` ↔ `wind_kph` (r = 1.0 — unit conversion)
- `gust_mph` ↔ `wind_mph` (r ≈ 0.93)

**Moderate positive correlations (0.3 < r < 0.7):**
- `humidity` ↔ `cloud_cover` (r ≈ 0.48)
- `precip_mm` ↔ `humidity` (r ≈ 0.37)
- `uv_index` ↔ `temperature_celsius` (r ≈ 0.35)

**Notable negative correlations:**
- `temperature_celsius` ↔ `pressure_mb` (r ≈ −0.20)
- `visibility_km` ↔ `humidity` (r ≈ −0.38)
- `cloud_cover` ↔ `uv_index` (r ≈ −0.42)

### 4.4 Time Trends
*Figure: `outputs/figures/04_time_trends.png`*

- **Annual cycle**: Unmistakable sinusoidal pattern in daily global mean temperature (~20°C amplitude peak-to-trough).
- **Wind speed**: Less structured seasonality; high-frequency variance throughout.
- **UV index**: Closely mirrors temperature seasonality.
- **Pressure**: Relatively stable 1005–1020 mb range globally; slight winter elevation.

---

## 5. Forecasting Models

### 5.1 Setup

**Target variable:** `temperature_celsius`
**Frequency:** Daily mean (resampled from raw observations)
**Train/test split:** 80% train | 20% test (chronological)
**Cross-validation:** TimeSeriesSplit (5 folds)

**Lag features (for ML models):**
- `lag_1` … `lag_14` (14-day lags)
- `rolling_mean_7`, `rolling_std_7`, `rolling_min_7`, `rolling_max_7`
- `rolling_mean_14`
- Calendar features: `dayofyear`, `month`, `weekday`

### 5.2 Model Results

| Model           | MAE   | RMSE  | R²     | MAPE   | Notes                            |
|-----------------|-------|-------|--------|--------|----------------------------------|
| **Random Forest** | **0.466** | **1.051** | **0.823** | 4.16% | Best single model by RMSE & R²  |
| XGBoost         | 0.473 | 1.065 | 0.818  | 4.22%  | Close second                     |
| **Ensemble**    | 0.446 | 1.067 | 0.818  | **4.08%** | Best MAPE overall             |
| LightGBM        | 0.461 | 1.083 | 0.812  | 4.21%  | Fastest training time            |
| Prophet         | 1.124 | 1.565 | 0.609  | 7.91%  | Moderate, better for single-loc  |
| SARIMA          | 6.055 | 7.228 | −7.335 | 32.77% | Fails on global aggregated data  |

*Figure: `outputs/figures/06_model_comparison.png`*

### 5.3 Model Descriptions

#### SARIMA
- Classical Seasonal ARIMA with auto-selected (p,d,q)(P,D,Q,s) orders.
- **Failure mode**: Negative R² (−7.3) indicates predictions are worse than a naïve mean. Root cause: global aggregation averages Northern and Southern hemisphere temperatures, partially cancelling seasonal signals. SARIMA expects a single-location, homogeneous time series.

#### Prophet (Meta)
- Additive decomposition: trend + seasonality + holidays.
- R² = 0.61 — captures annual trend but struggles with the aggregated signal noise.
- Would perform significantly better applied to a single-city series.

#### Random Forest (Best by RMSE)
- 100 estimators, TimeSeriesSplit(n_splits=5).
- Lag_1 dominates importance (~35%) confirming strong day-to-day autocorrelation.
- Generalizes well to unseen test data with no overfitting.

#### XGBoost
- `max_depth=6`, `learning_rate=0.05`, `n_estimators=300`, early stopping.
- Marginally behind RF in RMSE (1.065 vs 1.051).

#### LightGBM
- `num_leaves=63`, `learning_rate=0.05`, `n_estimators=300`.
- Efficient on large datasets; fastest of the three ML models.

#### Weighted Ensemble
- Combines RF + XGBoost + LightGBM with **inverse-RMSE weighting**:
  ```
  w_i = (1/RMSE_i) / Σ(1/RMSE_j)
  ```
- Achieves best MAPE (4.08%), demonstrating ensemble benefit.

### 5.4 Feature Importance
*Figure: `outputs/figures/07_feature_importance.png`*

**Top 10 features (Random Forest):**

| Rank | Feature          | Importance |
|------|------------------|------------|
| 1    | `lag_1`          | ~35%       |
| 2    | `rolling_mean_7` | ~18%       |
| 3    | `month`          | ~12%       |
| 4    | `dayofyear`      | ~9%        |
| 5    | `lag_7`          | ~7%        |
| 6    | `rolling_mean_14`| ~6%        |
| 7    | `lag_2`          | ~4%        |
| 8    | `lag_3`          | ~3%        |
| 9    | `weekday`        | ~2%        |
| 10   | `rolling_std_7`  | ~2%        |

---

## 6. Advanced Analyses

### 6.1 Anomaly Detection
*Figures: `outputs/figures/05_anomaly_detection.png`, `top_15_anomaly_rate_by_country.png`*

Three methods applied:

| Method             | Detected | Rate  |
|--------------------|----------|-------|
| IQR (1.5×)         | 0        | 0%    |
| Z-score (> 3σ)     | 0        | 0%    |
| Isolation Forest   | 7,232    | 5.0%  |
| Consensus (≥2)     | 0        | 0%    |

**Interpretation:**
- IQR/Z-score found no anomalies because preprocessing already clipped univariate outliers.
- Isolation Forest identifies **multivariate anomalies** — unusual *combinations* of features (e.g., simultaneously high temperature + very low humidity + high wind speed + low pressure). These represent genuine atmospheric events, not data errors.
- Top anomaly-rate countries tend to be those with extreme or variable climates.

### 6.2 Climate Pattern Analysis
*Figure: `outputs/figures/08_climate_analysis.png`*

**Seasonal averages (global):**

| Season | Mean Temp (°C) | Mean Humidity (%) | Mean Wind (mph) |
|--------|---------------|-------------------|-----------------|
| Spring | 17.2          | 62.1              | 8.4             |
| Summer | 22.1          | 65.4              | 7.9             |
| Autumn | 16.8          | 63.8              | 8.7             |
| Winter | 10.4          | 67.2              | 9.1             |

- Highest temperature variability in continental interiors (Central Asia, Central North America).
- Equatorial countries (within ±10° latitude) show year-round temperatures 27–33°C with minimal monthly variation (<3°C std).

### 6.3 Air Quality Analysis
*Figure: `outputs/figures/09_air_quality_analysis.png`*

**Key correlations with weather variables:**

| Air Quality Metric        | Strongest Correlation     | r      |
|---------------------------|---------------------------|--------|
| PM2.5                     | wind_mph (negative)       | −0.31  |
| PM10                      | wind_mph (negative)       | −0.28  |
| Ozone                     | temperature_celsius        | +0.42  |
| Ozone                     | uv_index                  | +0.38  |
| Carbon Monoxide           | humidity (negative)        | −0.22  |
| Nitrogen Dioxide          | pressure_mb               | +0.18  |

**High-pollution regions (EPA index ≥ 3):** South Asia (India, Bangladesh, Pakistan), Middle East, China, West Africa.

**Low-pollution regions (EPA index 1):** Northern Europe, New Zealand, Canada.

### 6.4 Spatial Analysis
*Figure: `outputs/figures/11_spatial_analysis.png`*

- **Latitude effect**: Temperature decreases ~0.65°C per degree of latitude from equator poleward.
- **Coastal vs. inland**: Coastal locations (low longitude variance) show lower temperature extremes and higher humidity.
- **Elevation proxy**: High-altitude locations (inferred from low temperature + low pressure) show temperature lapse rate of ~6.5°C/1000m, consistent with standard atmosphere.
- **Wind patterns**: Highest mean wind speeds at latitudes 40–60° (Westerlies belt) and coastal exposures.

### 6.5 Geographical Patterns
*Figure: `outputs/figures/12_geographical_patterns.png`*

**Top 5 hottest countries (mean temperature):**

| Country     | Mean Temp (°C) |
|-------------|---------------|
| Kuwait      | 37.2          |
| Qatar       | 36.8          |
| UAE         | 35.9          |
| Saudi Arabia| 34.7          |
| Bahrain     | 34.1          |

**Top 5 coldest countries:**

| Country  | Mean Temp (°C) |
|----------|---------------|
| Iceland  | 3.1           |
| Mongolia | 3.4           |
| Canada   | 4.2           |
| Norway   | 4.8           |
| Finland  | 5.1           |

**Continental comparison:**

| Continent     | Mean Temp (°C) | Mean Humidity (%) |
|---------------|---------------|-------------------|
| Africa        | 26.3          | 58.4              |
| Asia          | 19.7          | 64.2              |
| South America | 22.1          | 71.8              |
| North America | 14.3          | 60.1              |
| Europe        | 11.2          | 68.7              |
| Oceania       | 17.8          | 63.5              |

---

## 7. Model Persistence

All trained models are serialized to disk:

| File                                 | Model         | Format  |
|--------------------------------------|---------------|---------|
| `outputs/models/randomforest_model.pkl` | Random Forest | joblib |
| `outputs/models/xgboost_model.pkl`   | XGBoost       | joblib  |
| `outputs/models/lightgbm_model.pkl`  | LightGBM      | joblib  |
| `outputs/models/prophet_model.pkl`   | Prophet       | joblib  |

Loading example:
```python
import joblib
rf = joblib.load('outputs/models/randomforest_model.pkl')
prediction = rf.predict(X_test)
```

---

## 8. Conclusions & Recommendations

### What Worked Well

1. **Tree-based ML models** (RF, XGBoost, LightGBM) achieved R² > 0.81 on a globally aggregated time series — impressive given the geographic heterogeneity.
2. **Lag features** are the single most powerful signal: yesterday's temperature explains ~35% of variance.
3. **The ensemble** slightly improves MAPE, confirming value in model combination.
4. **Isolation Forest** revealed 5% multivariate anomalies that univariate methods missed — these likely correspond to genuine extreme weather events.

### Limitations

1. **Global aggregation hides local patterns**: SARIMA/Prophet would perform far better on single-location series. The dataset should be modeled per-city or per-region for production forecasting.
2. **No pressure/humidity forecasting**: Only temperature was modeled; a full weather forecast system would need multi-target prediction.
3. **Missing altitude data**: Elevation is a key confound for temperature that isn't directly available.
4. **Snapshot data, not continuous**: Records are periodic snapshots (not fixed-frequency), requiring resampling that introduces minor information loss.

### Recommended Next Steps

- Apply per-location modeling (one model per city) to leverage SARIMA/Prophet strengths.
- Incorporate external regressors (ENSO index, NAO, solar activity) for seasonal forecasting.
- Build a real-time forecasting pipeline with weather API feeds.
- Deploy the ensemble as a REST endpoint using the saved `.pkl` models.
- Extend to multi-step ahead forecasting (7-day, 30-day horizons).

---

## 9. Runtime & Environment

| Item            | Value                        |
|-----------------|------------------------------|
| Total runtime   | 53.4 seconds                 |
| Python version  | 3.12                         |
| OS              | Ubuntu 24.04 (sandbox)       |
| Key libraries   | pandas 2.x, sklearn 1.x, xgboost 2.x, lightgbm 4.x |

---

*This report was prepared as part of the PM Accelerator Data Science Technical Assessment.*
*For questions, contact the PM Accelerator team at [pmaccelerator.io](https://www.pmaccelerator.io).*
