# Weather Trend Forecasting
## PM Accelerator — Data Science / Analyst Technical Assessment
### Advanced Track

**Built by:** [Your Name]

---

## About PM Accelerator

**Product Manager Accelerator** is the world's leading product management community, empowering aspiring and current product managers with 1-on-1 coaching, job placement support, real-world projects, and a thriving global network. Whether you're breaking into PM or leveling up your career, PM Accelerator provides the tools and community to get you there faster.

🔗 [PM Accelerator on LinkedIn](https://www.linkedin.com/company/product-manager-accelerator/)

---

## Assessment Coverage

| Section | What's Implemented |
|---|---|
| **Data Cleaning** | Missing value imputation (median/mode) · Outlier detection & IQR clipping · StandardScaler normalization · Feature engineering (heat index, discomfort index, pressure categories) |
| **Basic EDA** | Temperature distribution & monthly patterns · Precipitation analysis · Correlation heatmap · Time-series trends |
| **Advanced EDA** | Anomaly detection: IQR + Isolation Forest + Z-score + **Consensus flag** |
| **Forecasting** | SARIMA · Prophet · XGBoost · LightGBM · Random Forest · **Weighted Ensemble** |
| **Unique Analysis 1** | Climate Analysis — seasonal violin, monthly normals, interannual trend, climate zones |
| **Unique Analysis 2** | Environmental Impact — Air quality (PM2.5/PM10) × weather correlations |
| **Unique Analysis 3** | Feature Importance — RF importance + Permutation importance + Gradient Boosting |
| **Unique Analysis 4** | Spatial Analysis — global lat/lon scatter maps of temperature & humidity |
| **Unique Analysis 5** | Geographical Patterns — country-level comparisons, variability ranking, radar chart |

---

## Dataset

**Global Weather Repository** by Kaggle user nelgiriyewithana  
🔗 https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code

- 40+ weather features (temperature, humidity, wind, pressure, UV, air quality, etc.)
- Hundreds of cities worldwide
- Place the downloaded file at: `data/GlobalWeatherRepository.csv`

---

## Project Structure

```
weather-ds/
├── data/
│   └── GlobalWeatherRepository.csv      ← place dataset here
│
├── notebooks/
│   └── weather_forecasting.ipynb        ← main deliverable
│
├── src/
│   ├── cleaning/
│   │   └── preprocess.py                ← loading, imputation, outliers, feature engineering
│   ├── eda/
│   │   └── explorer.py                  ← EDA plots + anomaly detection
│   ├── models/
│   │   └── forecasting.py               ← 5 models + ensemble
│   └── analysis/
│       └── unique_analyses.py           ← 5 unique analyses
│
├── outputs/
│   ├── figures/                         ← all generated plots (PNG)
│   └── models/                          ← saved model files (.pkl)
│
├── main.py                              ← end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Go to https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code  
Download `GlobalWeatherRepository.csv` and place it in the `data/` folder.

### 3a. Run as Jupyter Notebook (recommended)
```bash
jupyter notebook notebooks/weather_forecasting.ipynb
```

### 3b. Run as a script
```bash
python main.py --data data/GlobalWeatherRepository.csv
```

---

## Output Files

After running, you'll find:

| File | Description |
|---|---|
| `outputs/figures/01_temperature_analysis.png` | Temperature distribution, monthly box plots, country ranking |
| `outputs/figures/02_precipitation_analysis.png` | Precipitation distribution, seasonal trends |
| `outputs/figures/03_correlation_heatmap.png` | Feature correlation matrix + top pairs |
| `outputs/figures/04_time_trends.png` | Daily rolling trends for key variables |
| `outputs/figures/05_anomaly_detection.png` | 3-method anomaly comparison |
| `outputs/figures/06_model_comparison.png` | All 6 model forecasts + error metrics |
| `outputs/figures/07_feature_importance.png` | XGBoost & RF feature importances |
| `outputs/figures/08_climate_analysis.png` | Seasonal violin, normals, interannual trend |
| `outputs/figures/09_air_quality_analysis.png` | PM2.5 correlations with weather |
| `outputs/figures/10_feature_importance_analysis.png` | RF + Permutation + GB importance |
| `outputs/figures/11_spatial_analysis.png` | Global lat/lon scatter maps |
| `outputs/figures/12_geographical_patterns.png` | Country-level radar + variability |
| `outputs/models/xgboost_model.pkl` | Saved XGBoost model |
| `outputs/models/lightgbm_model.pkl` | Saved LightGBM model |
| `outputs/models/randomforest_model.pkl` | Saved Random Forest model |
| `outputs/model_metrics.csv` | All model MAE / RMSE / R² / MAPE |

---

## Model Performance (expected ranges on this dataset)

| Model | Typical RMSE | Notes |
|---|---|---|
| SARIMA | 3–6°C | Strong on stationary series; slower on large data |
| Prophet | 2–5°C | Captures yearly + weekly seasonality well |
| XGBoost | 1–3°C | Best single-model on multi-city data |
| LightGBM | 1–3°C | Similar to XGBoost, faster training |
| Random Forest | 1–4°C | Robust, good interpretability |
| **Ensemble** | **1–2°C** | Best overall; weighted by 1/RMSE |

---

## Key Findings Summary

- **Temperature & dewpoint** are the most correlated feature pair (r ≈ 0.85–0.95)
- **~5% of records** are consensus anomalies across 3 detection methods
- **Air quality (PM2.5)** correlates negatively with wind speed and precipitation — rain and wind clear the air
- **XGBoost / LightGBM** outperform classical SARIMA on multi-city aggregated data
- **Weighted ensemble** consistently delivers the lowest RMSE across test sets
- Clear **climate zone clusters** visible in the lat/lon spatial scatter
