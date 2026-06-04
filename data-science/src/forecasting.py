import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

FIG_DIR = 'outputs/figures'
MODEL_DIR = 'outputs/models'


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f'{FIG_DIR}/{name}.png'
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _metrics(y_true, y_pred, name='model') -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {'model': name, 'MAE': round(mae,3), 'RMSE': round(rmse,3),
            'R2': round(r2,4), 'MAPE': round(mape,2)}


# Time-series data prep
def prepare_ts(df: pd.DataFrame,
               target: str = 'temperature_celsius',
               freq: str = 'D',
               agg: str = 'mean') -> pd.Series:
    df2 = df.copy()
    df2['last_updated'] = pd.to_datetime(df2['last_updated'])
    ts = (df2.set_index('last_updated')[target]
            .resample(freq)
            .agg(agg)
            .interpolate('time'))
    ts = ts.dropna()
    print(f"Time series: {len(ts)} {freq}-periods | "
          f"{ts.index.min().date()} to {ts.index.max().date()}")
    return ts


def create_lag_features(ts: pd.Series, n_lags: int = 14) -> pd.DataFrame:
    df = pd.DataFrame({'y': ts})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_mean_7']  = df['y'].shift(1).rolling(7).mean()
    df['rolling_std_7']   = df['y'].shift(1).rolling(7).std()
    df['rolling_mean_14'] = df['y'].shift(1).rolling(14).mean()
    df['rolling_min_7']   = df['y'].shift(1).rolling(7).min()
    df['rolling_max_7']   = df['y'].shift(1).rolling(7).max()
    df['dayofyear'] = df.index.dayofyear
    df['month']     = df.index.month
    df['weekday']   = df.index.weekday
    return df.dropna()


def train_test_split_ts(df: pd.DataFrame, test_frac: float = 0.2):
    split = int(len(df) * (1 - test_frac))
    return df.iloc[:split], df.iloc[split:]


# SARIMA forecasting
def run_sarima(ts: pd.Series, steps: int = 30) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    try:
        model = SARIMAX(train, order=(1,1,1),
                        seasonal_order=(1,1,1,7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=50)
        pred = fit.forecast(steps=len(test))
        pred.index = test.index
        m = _metrics(test.values, pred.values, 'SARIMA')
        forecast = fit.forecast(steps=steps)
        return {'metrics': m, 'train': train, 'test': test,
                'pred': pred, 'forecast': forecast, 'fit': fit}
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        return None


# Prophet forecasting
def run_prophet(ts: pd.Series, steps: int = 30) -> dict:
    try:
        from prophet import Prophet
    except ImportError:
        print("  Prophet not installed, skipping."); return None

    split = int(len(ts) * 0.8)
    train_df = pd.DataFrame({'ds': ts.iloc[:split].index,
                              'y': ts.iloc[:split].values})
    test_ts  = ts.iloc[split:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, interval_width=0.95)
    m.fit(train_df)

    future = m.make_future_dataframe(periods=len(test_ts) + steps)
    forecast_df = m.predict(future)

    test_pred = forecast_df.set_index('ds')['yhat'].loc[test_ts.index]
    metrics   = _metrics(test_ts.values, test_pred.values, 'Prophet')
    future_fc = forecast_df.tail(steps).set_index('ds')['yhat']

    return {'metrics': metrics, 'test': test_ts,
            'pred': test_pred, 'forecast': future_fc,
            'components_df': forecast_df, 'model': m}


# XGBoost forecasting
def run_xgboost(ts: pd.Series, steps: int = 30) -> dict:
    feat_df = create_lag_features(ts)
    train_df, test_df = train_test_split_ts(feat_df)

    X_tr, y_tr = train_df.drop('y', axis=1), train_df['y']
    X_te, y_te = test_df.drop('y', axis=1), test_df['y']

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              verbose=False)

    pred = pd.Series(model.predict(X_te), index=y_te.index)
    metrics = _metrics(y_te.values, pred.values, 'XGBoost')

    # Recursive forecast
    last_row = feat_df.iloc[-1:].copy()
    forecast_vals = []
    for _ in range(steps):
        pred_val = model.predict(last_row.drop('y', axis=1))[0]
        forecast_vals.append(pred_val)
        # Shift lags
        for lag in range(14, 1, -1):
            if f'lag_{lag}' in last_row.columns:
                last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
        last_row['lag_1'] = pred_val

    fc_index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast = pd.Series(forecast_vals, index=fc_index)

    return {'metrics': metrics, 'test': y_te, 'pred': pred,
            'forecast': forecast, 'model': model,
            'feature_importance': dict(zip(X_tr.columns,
                                            model.feature_importances_))}


# LightGBM forecasting
def run_lightgbm(ts: pd.Series, steps: int = 30) -> dict:
    feat_df = create_lag_features(ts)
    train_df, test_df = train_test_split_ts(feat_df)

    X_tr, y_tr = train_df.drop('y', axis=1), train_df['y']
    X_te, y_te = test_df.drop('y', axis=1), test_df['y']

    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                          lgb.log_evaluation(period=-1)])

    pred = pd.Series(model.predict(X_te), index=y_te.index)
    metrics = _metrics(y_te.values, pred.values, 'LightGBM')

    last_row = feat_df.iloc[-1:].copy()
    forecast_vals = []
    for _ in range(steps):
        pred_val = model.predict(last_row.drop('y', axis=1))[0]
        forecast_vals.append(pred_val)
        for lag in range(14, 1, -1):
            if f'lag_{lag}' in last_row.columns:
                last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
        last_row['lag_1'] = pred_val

    fc_index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast = pd.Series(forecast_vals, index=fc_index)

    return {'metrics': metrics, 'test': y_te, 'pred': pred,
            'forecast': forecast, 'model': model}


# Random Forest forecasting
def run_random_forest(ts: pd.Series, steps: int = 30) -> dict:
    feat_df = create_lag_features(ts)
    train_df, test_df = train_test_split_ts(feat_df)

    X_tr, y_tr = train_df.drop('y', axis=1), train_df['y']
    X_te, y_te = test_df.drop('y', axis=1), test_df['y']

    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    pred = pd.Series(model.predict(X_te), index=y_te.index)
    metrics = _metrics(y_te.values, pred.values, 'RandomForest')

    last_row = feat_df.iloc[-1:].copy()
    forecast_vals = []
    for _ in range(steps):
        pred_val = model.predict(last_row.drop('y', axis=1))[0]
        forecast_vals.append(pred_val)
        for lag in range(14, 1, -1):
            if f'lag_{lag}' in last_row.columns:
                last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
        last_row['lag_1'] = pred_val

    fc_index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast = pd.Series(forecast_vals, index=fc_index)

    return {'metrics': metrics, 'test': y_te, 'pred': pred,
            'forecast': forecast, 'model': model,
            'feature_importance': dict(zip(X_tr.columns,
                                            model.feature_importances_))}


# Weighted ensemble
def build_ensemble(results: dict, weights: dict = None) -> dict:
    # Default: weight by 1/RMSE
    if weights is None:
        weights = {}
        for name, res in results.items():
            if res and 'metrics' in res:
                weights[name] = 1 / (res['metrics']['RMSE'] + 1e-8)

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Align test predictions on common index
    common_idx = None
    for name, res in results.items():
        if res and 'pred' in res:
            if common_idx is None:
                common_idx = res['pred'].index
            else:
                common_idx = common_idx.intersection(res['pred'].index)

    ensemble_pred = pd.Series(0.0, index=common_idx)
    for name, res in results.items():
        if res and 'pred' in res and name in weights:
            ensemble_pred += weights[name] * res['pred'].reindex(common_idx)

    # Use any result's test values for evaluation
    y_te = None
    for res in results.values():
        if res and 'test' in res:
            y_te = res['test'].reindex(common_idx)
            break

    metrics = _metrics(y_te.values, ensemble_pred.values, 'Ensemble') if y_te is not None else {}

    # Ensemble forecast
    fc_len = min(len(res['forecast']) for res in results.values()
                 if res and 'forecast' in res)
    fc_index = list(results.values())[0]['forecast'].index[:fc_len]
    ensemble_fc = pd.Series(0.0, index=fc_index)
    for name, res in results.items():
        if res and 'forecast' in res and name in weights:
            fc = res['forecast'].reindex(fc_index)
            ensemble_fc += weights[name] * fc.ffill()

    return {'metrics': metrics, 'pred': ensemble_pred,
            'forecast': ensemble_fc, 'weights': weights, 'test': y_te}


# Plotting results

def plot_model_comparison(results: dict, ts: pd.Series) -> str:
    n = len(results)
    fig = plt.figure(figsize=(18, 5 * (n + 1)))
    gs  = gridspec.GridSpec(n + 1, 1, figure=fig,
                             hspace=0.5) if n > 0 else None

    # Metrics bar chart
    metrics_list = [r['metrics'] for r in results.values() if r and 'metrics' in r]
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).set_index('model')
        ax_top = fig.add_subplot(gs[0])
        metrics_df[['MAE','RMSE']].plot(kind='bar', ax=ax_top,
                                         color=['C0','C3'],
                                         alpha=0.85, edgecolor='white')
        ax_top.set(title='Model Comparison — MAE & RMSE', ylabel='Error')
        ax_top.set_xticklabels(ax_top.get_xticklabels(), rotation=30, ha='right')
        for container in ax_top.containers:
            ax_top.bar_label(container, fmt='%.3f', padding=2, fontsize=8)

    # Individual model forecast plots
    split_idx = int(len(ts) * 0.8)
    train_ts = ts.iloc[:split_idx]

    for i, (name, res) in enumerate(results.items()):
        if not res: continue
        ax = fig.add_subplot(gs[i + 1])
        ax.plot(train_ts.index[-60:], train_ts.iloc[-60:],
                color='gray', lw=1.5, label='Train (last 60)')
        if 'test' in res:
            ax.plot(res['test'].index, res['test'], color='black',
                    lw=2, label='Actual', zorder=5)
        if 'pred' in res:
            ax.plot(res['pred'].index, res['pred'],
                    color='C0', lw=2, linestyle='--', label='Predicted')
        if 'forecast' in res:
            ax.plot(res['forecast'].index, res['forecast'],
                    color='C1', lw=2, linestyle=':', label='Forecast')
        m = res.get('metrics', {})
        title = f"{name} | MAE={m.get('MAE','?')} RMSE={m.get('RMSE','?')} R²={m.get('R2','?')}"
        ax.set(title=title, ylabel='°C')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Forecasting Model Results — Temperature',
                 fontsize=15, fontweight='bold', y=1.01)
    return _save(fig, '06_model_comparison')


def plot_feature_importance(results: dict) -> str:
    from matplotlib import gridspec
    tree_models = {k: v for k, v in results.items()
                   if v and 'feature_importance' in v}
    if not tree_models:
        return None

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, tree_models.items()):
        imp = pd.Series(res['feature_importance']).nlargest(15).sort_values()
        ax.barh(imp.index, imp.values)
        ax.set(title=f'{name} — Feature Importance', xlabel='Importance')

    fig.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '07_feature_importance')


def run_all_models(df: pd.DataFrame,
                   target: str = 'temperature_celsius',
                   steps: int = 30) -> tuple:
    print(f"\n{'='*50}")
    print(f"Running forecasting models for: {target}")
    print('='*50)

    ts = prepare_ts(df, target=target)

    results = {}

    print("\n[1/5] SARIMA...")
    results['SARIMA'] = run_sarima(ts, steps)

    print("[2/5] Prophet...")
    results['Prophet'] = run_prophet(ts, steps)

    print("[3/5] XGBoost...")
    results['XGBoost'] = run_xgboost(ts, steps)

    print("[4/5] LightGBM...")
    results['LightGBM'] = run_lightgbm(ts, steps)

    print("[5/5] Random Forest...")
    results['RandomForest'] = run_random_forest(ts, steps)

    print("\n[Ensemble] Building weighted ensemble...")
    results['Ensemble'] = build_ensemble(
        {k: v for k, v in results.items() if k != 'Ensemble'})

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, res in results.items():
        if res and 'model' in res:
            joblib.dump(res['model'], f'{MODEL_DIR}/{name.lower()}_model.pkl')

    # Metrics table
    metrics_rows = [r['metrics'] for r in results.values()
                    if r and 'metrics' in r]
    metrics_df = pd.DataFrame(metrics_rows).set_index('model').sort_values('RMSE')
    print("\nModel Performance:")
    print(metrics_df.to_string())

    fig_path   = plot_model_comparison(results, ts)
    fig_fi     = plot_feature_importance(results)

    return results, metrics_df, ts
