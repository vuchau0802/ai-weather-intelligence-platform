# 🌍 Weather Trend Forecasting — Global Weather Repository Analysis

**PM Accelerator Data Science / Analyst Technical Assessment**

---

## 🎯 PM Accelerator Mission

> *"Our mission is to break down financial barriers and achieve educational fairness. With the goal of establishing 200 schools worldwide over the next 20 years, we aim to empower more kids for a better future in their life and career, simultaneously fostering a diverse landscape in the tech industry."*
>
> — [pmaccelerator.io](https://www.pmaccelerator.io/about-us)

PM Accelerator (founded by Dr. Nancy Li) helps aspiring and current Product Managers land roles at top-tier tech companies — even with zero prior PM experience — through certified programs, 1:1 coaching, and a global alumni community of 1,500+ professionals.

---

## 📋 Project Overview

This project analyzes the **Global Weather Repository** dataset (144,627 observations × 52 features, spanning 211 countries, 257 cities, and dates from May 2024 to May 2026) to:

1. Clean and preprocess raw weather data.
2. Perform exploratory data analysis (EDA) on temperature, precipitation, air quality, and spatial patterns.
3. Build and evaluate multiple time-series forecasting models.
4. Conduct advanced analyses including anomaly detection, climate pattern analysis, and geographic comparisons.
5. Document findings with visualizations and actionable insights.

---

## 📁 Project Structure

```
WeatherForecastingReport/
├── data/
│   └── GlobalWeatherRepository.csv       # Raw input dataset (144,627 rows × 52 cols)
├── src/
│   ├── preprocess.py                     # Data cleaning & feature engineering pipeline
│   ├── explorer.py                       # EDA & visualization functions
│   ├── forecasting.py                    # Forecasting models (SARIMA, Prophet, ML)
│   └── unique_analyses.py               # Advanced analyses (climate, air quality, spatial)
├── outputs/
│   ├── processed_globalweather.csv       # Cleaned & feature-engineered dataset
│   ├── model_metrics.csv                 # Model evaluation table
│   ├── analysis_report.md               # Concise auto-generated report
│   ├── figures/                          # All visualizations (14 PNG files)
│   └── models/                           # Saved model artifacts (.pkl files)
├── main.py                               # Entry-point: runs full pipeline
├── app.py                                # Flask web app for interactive exploration
├── weather_forecasting.ipynb            # Jupyter notebook (executed)
├── FULL_REPORT.md                        # 📄 Comprehensive analysis report (this project)
└── README.md                             # This file
```

---

## 🔧 Setup & Installation

```bash
# 1. Clone or unzip the project
cd WeatherForecastingReport

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
statsmodels>=0.14
prophet>=1.1
scipy>=1.11
joblib>=1.3
flask>=3.0
```

---

## 🚀 Running the Project

### Full pipeline (all analyses + models):

```bash
python main.py --data data/GlobalWeatherRepository.csv
```

### Quick mode (faster local check):

```bash
python main.py --data data/GlobalWeatherRepository.csv --quick
```

### Interactive web dashboard:

```bash
python app.py
# Open http://localhost:5000
```

### Jupyter Notebook (pre-executed):

Open `weather_forecasting.ipynb` in JupyterLab or Jupyter Notebook.

---

## 📊 Dataset Summary

| Attribute         | Value                                 |
|-------------------|---------------------------------------|
| Total rows        | 144,627                               |
| Total columns     | 52                                    |
| Countries covered | 211                                   |
| Cities covered    | 257                                   |
| Date range        | 2024-05-16 → 2026-05-31               |
| Target variable   | `temperature_celsius`                 |
| Key features      | Temperature, Humidity, Wind, Pressure, Precipitation, Air Quality indices, UV index, Moon phase |

---

## 🧹 Data Cleaning & Preprocessing (`src/preprocess.py`)

### Steps Performed

1. **DateTime parsing** — Extracted `year`, `month`, `day`, `hour`, `dayofyear`, `week`, and `season` from the `last_updated` timestamp.

2. **Missing value handling**:
   - Columns with ≥ 40% missing values were dropped.
   - Numeric columns: imputed with **median**.
   - Categorical columns: imputed with **mode**.

3. **Outlier treatment** (IQR clipping on key columns):
   - `temperature_celsius`, `wind_mph`, `humidity`, `pressure_mb`, `precip_mm`, `visibility_km`, `feels_like_celsius`
   - Values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR] were clipped (not removed) to preserve time-series continuity.

4. **Feature engineering**:

   | New Feature            | Formula / Logic                                      |
   |------------------------|------------------------------------------------------|
   | `temp_feels_diff`      | `temperature_celsius − feels_like_celsius`           |
   | `discomfort_index`     | `wind_mph × humidity / 100`                          |
   | `heat_index_approx`    | Rothfusz polynomial (NWS standard)                   |
   | `pressure_category`    | Binned: Very Low / Low / Normal / High / Very High   |
   | `visibility_category`  | Binned: Fog / Mist / Moderate / Good / Excellent     |

5. **Normalization**: StandardScaler applied to numeric features (for ML models).

---

## 🔍 Exploratory Data Analysis (`src/explorer.py`)

### Temperature Analysis (`figures/01_temperature_analysis.png`)

- Global mean temperature: ~**16–20°C** with a bimodal distribution reflecting both hemispheres.
- Seasonal patterns confirm expected warming (Summer) and cooling (Winter) cycles.
- Monthly breakdowns show peak temperatures in July–August for Northern Hemisphere locations.

### Precipitation Analysis (`figures/02_precipitation_analysis.png`)

- Most observations record **0 mm precipitation** (dry weather dominates).
- Tropical and monsoon regions show sporadic high-precipitation events (>50 mm).
- Seasonal distribution: precipitation peaks in Southern Hemisphere winter (June–August).

### Correlation Heatmap (`figures/03_correlation_heatmap.png`)

Strong correlations found:
- `temperature_celsius` ↔ `feels_like_celsius` (r ≈ 0.99)
- `temperature_celsius` ↔ `dew_point_celsius` (r ≈ 0.83)
- `humidity` ↔ `cloud_cover` (r ≈ 0.48)
- Negative: `temperature_celsius` ↔ `pressure_mb` (r ≈ −0.20)

### Time Trends (`figures/04_time_trends.png`)

- Daily average global temperature oscillates with a ~365-day cycle (annual seasonality clearly visible).
- Wind speeds show less pronounced seasonality but higher variance in winter months.
- UV index peaks align with local summer periods.

---

## 🤖 Forecasting Models (`src/forecasting.py`)

All models forecast daily-mean `temperature_celsius` using an 80/20 train/test split.

### Model Evaluation Results

| Model         | MAE   | RMSE  | R²     | MAPE  |
|---------------|-------|-------|--------|-------|
| **RandomForest** | **0.466** | **1.051** | **0.823** | **4.16%** |
| XGBoost       | 0.473 | 1.065 | 0.818  | 4.22% |
| **Ensemble**  | 0.446 | 1.067 | 0.818  | **4.08%** |
| LightGBM      | 0.461 | 1.083 | 0.812  | 4.21% |
| Prophet       | 1.124 | 1.565 | 0.609  | 7.91% |
| SARIMA        | 6.055 | 7.228 | −7.335 | 32.77% |

### Model Descriptions

**SARIMA** — Classical statsmodels seasonal ARIMA. Underperformed significantly (R² < 0) likely due to global dataset heterogeneity (locations from all hemispheres cancel out seasonal signals when aggregated).

**Prophet (Meta)** — Trend + seasonality decomposition model. Moderate performance (R² = 0.61). Better suited to single-location univariate series.

**Random Forest** — Best single model by RMSE (1.051) and R² (0.823). 100 trees, lag + rolling features, TimeSeriesSplit cross-validation.

**XGBoost** — Close second (RMSE 1.065). Gradient-boosted trees with 14-day lag features and rolling statistics.

**LightGBM** — Efficient gradient boosting (RMSE 1.083). Slightly lower accuracy but fastest training time.

**Weighted Ensemble** — Combines RF + XGBoost + LightGBM with inverse-RMSE weights. Best MAPE (4.08%), confirming ensemble benefit for generalization.

### Feature Engineering for ML Models

- Lag features: `lag_1` through `lag_14`
- Rolling statistics: 7-day mean, std, min, max; 14-day mean
- Calendar features: `dayofyear`, `month`, `weekday`

---

## 🔬 Advanced Analyses (`src/unique_analyses.py`)

### Anomaly Detection (`figures/05_anomaly_detection.png`)

Three complementary methods applied to `temperature_celsius`:

| Method             | Anomalies Detected |
|--------------------|--------------------|
| IQR (1.5×)         | 0                  |
| Z-score (> 3σ)     | 0                  |
| Isolation Forest   | 7,232 (5.0%)       |

- IQR and Z-score found no outliers after preprocessing (clipping handled them).
- Isolation Forest identified **7,232 multivariate anomalies** — records with unusual combinations of temperature, humidity, pressure, and wind that don't match global norms.
- Top countries by anomaly rate are visualized in `figures/top_15_anomaly_rate_by_country.png`.

### Climate Pattern Analysis (`figures/08_climate_analysis.png`)

- **Seasonal temperatures**: Summer average ~22°C vs. Winter ~10°C globally (Northern Hemisphere bias due to more countries).
- **Monthly variability**: July highest; January lowest.
- **Regional patterns**: Equatorial locations show minimal seasonal variance; high-latitude locations show extreme swings (>30°C annual range).

### Air Quality & Weather Correlations (`figures/09_air_quality_analysis.png`)

Key findings:
- **PM2.5** and **PM10** correlate positively with lower wind speeds (stagnant air traps particles).
- **Ozone** increases with temperature and UV index.
- **Carbon Monoxide** shows inverse correlation with humidity.
- South/Southeast Asia and Middle East locations consistently show higher air quality index (EPA scale 3–5).

### Feature Importance (`figures/10_feature_importance_analysis.png`)

Top predictors of `temperature_celsius`:
1. `lag_1` (yesterday's temperature) — dominant predictor (~35% importance)
2. `rolling_mean_7` — weekly trend
3. `month` — seasonality signal
4. `dayofyear` — annual cycle
5. `lag_7` — weekly lag

### Spatial Analysis (`figures/11_spatial_analysis.png`)

- Strong latitudinal gradient: temperature decreases ~0.65°C per degree of latitude (poleward).
- Longitudinal variation minimal at mid-latitudes but significant at tropical latitudes (ocean vs. land effects).
- Wind speeds highest at coastal and island locations.

### Geographical Patterns (`figures/12_geographical_patterns.png`)

- **Hottest countries on average**: Middle Eastern nations (Qatar, UAE, Kuwait) — avg >35°C.
- **Coldest**: Iceland, Mongolia, Canada — avg <5°C.
- **Highest precipitation**: Southeast Asia, West Africa.
- **Continent comparison**: Africa highest mean temp; Europe widest seasonal swing.

---

## 📈 Key Insights & Conclusions

1. **Tree-based ML models vastly outperform classical time-series models** (SARIMA, Prophet) for a globally aggregated weather dataset. This is because the dataset mixes both hemispheres (opposing seasons) and many climatic zones, making additive decomposition misleading.

2. **Lag features are the most powerful predictors** — yesterday's temperature alone explains ~35% of variance. This confirms strong temporal autocorrelation in weather data.

3. **The ensemble model achieves the best MAPE (4.08%)**, suggesting that combining RF, XGBoost, and LightGBM provides meaningful complementary signal.

4. **Isolation Forest detected 5% multivariate anomalies** — these correspond to unusual weather events (heat waves, cold snaps, extreme wind-humidity combinations) rather than data errors.

5. **Air quality is strongly weather-dependent**: high temperature + low wind = elevated ozone and particulate matter. This has public health implications for urban planning and emissions policy.

6. **The Global Weather Repository provides excellent coverage** (211 countries) but is biased toward hourly snapshots of capital cities, limiting rural and regional granularity.

---

## 🗂️ Outputs Summary

| File                                           | Description                             |
|------------------------------------------------|-----------------------------------------|
| `outputs/processed_globalweather.csv`          | Cleaned, feature-engineered dataset     |
| `outputs/model_metrics.csv`                    | Model evaluation table (6 models)       |
| `outputs/analysis_report.md`                  | Auto-generated concise report           |
| `outputs/figures/01_temperature_analysis.png` | Temperature distribution & trends       |
| `outputs/figures/02_precipitation_analysis.png` | Precipitation patterns               |
| `outputs/figures/03_correlation_heatmap.png`  | Feature correlation matrix              |
| `outputs/figures/04_time_trends.png`           | Time-series trends (temp, wind, UV)     |
| `outputs/figures/05_anomaly_detection.png`     | Anomaly detection results               |
| `outputs/figures/06_model_comparison.png`      | Forecast vs. actual (all models)        |
| `outputs/figures/07_feature_importance.png`    | RF & XGBoost feature importance         |
| `outputs/figures/08_climate_analysis.png`      | Seasonal & monthly climate patterns     |
| `outputs/figures/09_air_quality_analysis.png`  | Air quality vs. weather correlations    |
| `outputs/figures/10_feature_importance_analysis.png` | Permutation importance         |
| `outputs/figures/11_spatial_analysis.png`      | Lat/Lon weather patterns                |
| `outputs/figures/12_geographical_patterns.png` | Country & continent comparisons         |
| `outputs/models/randomforest_model.pkl`        | Saved Random Forest model               |
| `outputs/models/xgboost_model.pkl`             | Saved XGBoost model                     |
| `outputs/models/lightgbm_model.pkl`            | Saved LightGBM model                    |
| `outputs/models/prophet_model.pkl`             | Saved Prophet model                     |

---

## 📚 References

- [Global Weather Repository — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
- [Prophet — Meta Research](https://facebook.github.io/prophet/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [PM Accelerator](https://www.pmaccelerator.io/)

---

*Report generated as part of the PM Accelerator Data Science Technical Assessment.*
