import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PALETTE = 'viridis'
FIG_DIR = 'outputs/figures'

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
})


# Helpers

def _save(fig, name: str):
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f'{FIG_DIR}/{name}.png'
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# Dataset overview
def dataset_overview(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    return {
        'shape': df.shape,
        'n_countries': df['country'].nunique() if 'country' in df.columns else None,
        'n_cities': df['location_name'].nunique() if 'location_name' in df.columns else None,
        'date_range': (str(df['last_updated'].min()), str(df['last_updated'].max()))
                       if 'last_updated' in df.columns else None,
        'numeric_summary': numeric.describe().round(2),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
    }


# Temperature analysis
def plot_temperature(df: pd.DataFrame) -> str:
    
    col = 'temperature_celsius'
    if col not in df.columns:
        print("No temperature_celsius column"); return

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[col].dropna(), bins=60, alpha=0.8, edgecolor='white')
    ax1.axvline(df[col].mean(), color='red', lw=2, linestyle='--', label=f"Mean {df[col].mean():.1f}°C")
    ax1.set(title='Temperature Distribution', xlabel='°C', ylabel='Count')
    ax1.legend()

    # Monthly box
    ax2 = fig.add_subplot(gs[0, 1])
    if 'month' in df.columns:
        month_data = [df[df.month == m][col].dropna().values for m in range(1, 13)]
        bp = ax2.boxplot(month_data, patch_artist=True,
                         medianprops=dict(color='white', lw=2))
        for i, patch in enumerate(bp['boxes']):
            patch.set_alpha(0.8)
        ax2.set(title='Monthly Temperature Distribution',
                xlabel='Month', ylabel='°C',
                xticks=range(1, 13),
                xticklabels=['Jan','Feb','Mar','Apr','May','Jun',
                             'Jul','Aug','Sep','Oct','Nov','Dec'])

    # Temperature heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    if 'season' in df.columns and 'month' in df.columns:
        pivot = df.groupby(['season', 'month'])[col].mean().unstack(fill_value=0)
        sns.heatmap(pivot, ax=ax3, cmap='RdYlBu_r', annot=True, fmt='.1f',
                    linewidths=0.5, cbar_kws={'label': '°C'})
        ax3.set_title('Avg Temperature: Season × Month')

    # Top hottest countries
    ax4 = fig.add_subplot(gs[1, 1])
    if 'country' in df.columns:
        top = df.groupby('country')[col].mean().nlargest(20)
        bars = ax4.barh(top.index, top.values)
        ax4.set(title='Top 20 Countries by Avg Temperature', xlabel='°C')
        ax4.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)

    fig.suptitle('Temperature Analysis — Global Weather Repository',
                 fontsize=15, fontweight='bold', y=1.01)
    return _save(fig, '01_temperature_analysis')


# Precipitation analysis
def plot_precipitation(df: pd.DataFrame) -> str:
    col = 'precip_mm'
    if col not in df.columns:
        print("No precip_mm column"); return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Distribution (log-transformed, right-skewed)
    ax = axes[0, 0]
    data = df[col].dropna()
    data_log = np.log1p(data)
    ax.hist(data_log, bins=50, alpha=0.8, edgecolor='white')
    ax.set(title='Precipitation Distribution (log1p)', xlabel='log1p(mm)', ylabel='Count')

    # Monthly average
    ax = axes[0, 1]
    if 'month' in df.columns:
        monthly = df.groupby('month')[col].mean()
        monthly.plot(kind='bar', ax=ax, edgecolor='white')
        ax.set(title='Average Monthly Precipitation', xlabel='Month', ylabel='mm')
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45)

    # Top rainy countries
    ax = axes[1, 0]
    if 'country' in df.columns:
        top = df.groupby('country')[col].mean().nlargest(20)
        top.sort_values().plot(kind='barh', ax=ax, alpha=0.8)
        ax.set(title='Top 20 Rainy Countries (avg mm)', xlabel='mm')

    # Temperature vs precipitation scatter
    ax = axes[1, 1]
    if 'temperature_celsius' in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        scatter = ax.scatter(sample['temperature_celsius'], sample[col],
                             alpha=0.3, c=sample.get('humidity', 50), s=10)
        plt.colorbar(scatter, ax=ax, label='Humidity %')
        ax.set(title='Temperature vs Precipitation', xlabel='°C', ylabel='mm')

    fig.suptitle('Precipitation Analysis', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, '02_precipitation_analysis')


# Correlation heatmap
def plot_correlation(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    focus = ['temperature_celsius', 'feelslike_celsius', 'dewpoint_celsius',
             'humidity', 'wind_mph', 'pressure_mb', 'precip_mm',
             'vis_km', 'uv_index', 'air_quality_PM2.5', 'air_quality_PM10']
    cols = [c for c in focus if c in df.columns]
    if not cols:
        cols = num_cols[:15]

    corr = df[cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Full heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[0], mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')

    # Top correlations bar chart
    ax = axes[1]
    pairs = (corr.where(~np.eye(len(corr), dtype=bool))
                 .stack()
                 .reset_index()
                 .rename(columns={0: 'corr',
                                   'level_0': 'feat_a',
                                   'level_1': 'feat_b'}))
    pairs['abs_corr'] = pairs['corr'].abs()
    top = pairs.nlargest(15, 'abs_corr')
    top['pair'] = top['feat_a'].str[:15] + ' × ' + top['feat_b'].str[:15]
    ax.barh(top['pair'], top['corr'], alpha=0.85)
    ax.axvline(0, color='black', lw=0.8)
    ax.set(title='Top 15 Feature Correlations', xlabel='Pearson r')

    fig.suptitle('Correlation Analysis', fontsize=15, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '03_correlation_heatmap')


# Time-series trends
def plot_time_trends(df: pd.DataFrame) -> str:
    if 'last_updated' not in df.columns:
        return

    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['last_updated']).dt.date

    metrics = ['temperature_celsius', 'humidity', 'wind_mph', 'precip_mm']
    metrics = [m for m in metrics if m in df2.columns]

    daily = df2.groupby('date')[metrics].mean()
    daily.index = pd.to_datetime(daily.index)
    daily_smooth = daily.rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)),
                              sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, col in zip(axes, metrics):
        ax.fill_between(daily.index, daily[col], alpha=0.15)
        ax.plot(daily.index, daily_smooth[col], lw=2,
                label='7-day rolling mean')
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Date')
    fig.suptitle('Global Daily Weather Trends Over Time',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '04_time_trends')


# Anomaly detection
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05, consensus_threshold: int = 2) -> tuple:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df2 = df.copy()

    feature_cols = ['temperature_celsius', 'humidity', 'wind_mph',
                    'pressure_mb', 'precip_mm', 'vis_km']
    feature_cols = [c for c in feature_cols if c in df2.columns]

    X = df2[feature_cols].fillna(df2[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Method 1: IQR
    iqr_flags = pd.Series(False, index=df2.index)
    for col in feature_cols:
        Q1, Q3 = df2[col].quantile(0.25), df2[col].quantile(0.75)
        IQR = Q3 - Q1
        flag = (df2[col] < Q1 - 1.5 * IQR) | (df2[col] > Q3 + 1.5 * IQR)
        iqr_flags |= flag
    df2['anomaly_iqr'] = iqr_flags.astype(int)

    # Method 2: Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df2['anomaly_isoforest'] = (iso.fit_predict(X_scaled) == -1).astype(int)
    df2['anomaly_score_iso'] = -iso.score_samples(X_scaled)

    # Method 3: Z-score (|z| > 3)
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    df2['anomaly_zscore'] = (z_scores > 3).any(axis=1).astype(int)

    # Consensus flag
    df2['anomaly_consensus'] = (
        df2['anomaly_iqr'] + df2['anomaly_isoforest'] + df2['anomaly_zscore'] >= consensus_threshold
    ).astype(int)

    # Plot results
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

    # Method counts
    ax = fig.add_subplot(gs[0, 0])
    counts = {
        'IQR': df2['anomaly_iqr'].sum(),
        'Isolation Forest': df2['anomaly_isoforest'].sum(),
        'Z-Score > 3 std': df2['anomaly_zscore'].sum(),
        'Consensus >= 2 methods': df2['anomaly_consensus'].sum(),
    }
    bars = ax.bar(counts.keys(), counts.values(), alpha=0.85, edgecolor='white')
    ax.bar_label(bars, fmt='{:,.0f}', padding=3)
    ax.set(title='Anomaly Counts by Method', ylabel='Count')

    # Anomaly score distribution
    ax = fig.add_subplot(gs[0, 1])
    normal_scores = df2.loc[df2['anomaly_isoforest'] == 0, 'anomaly_score_iso']
    anomaly_scores = df2.loc[df2['anomaly_isoforest'] == 1, 'anomaly_score_iso']
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal')
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly')
    ax.set(title='Isolation Forest Anomaly Score Distribution',
           xlabel='Anomaly Score', ylabel='Count')
    ax.legend()

    # Use consensus anomalies when available; after preprocessing clips IQR
    # outliers, consensus can be zero, so fall back to Isolation Forest.
    anomaly_plot_col = (
        'anomaly_consensus'
        if df2['anomaly_consensus'].sum() > 0
        else 'anomaly_isoforest'
    )
    anomaly_plot_label = (
        'Consensus Anomaly'
        if anomaly_plot_col == 'anomaly_consensus'
        else 'Isolation Forest Anomaly'
    )

    # Scatter: temp vs humidity, colored by anomaly
    ax = fig.add_subplot(gs[1, :])
    if 'temperature_celsius' in df2.columns and 'humidity' in df2.columns:
        sample = df2.sample(min(8000, len(df2)), random_state=42)
        colors_scatter = ['C3' if a else 'C0' for a in sample[anomaly_plot_col]]
        ax.scatter(sample['temperature_celsius'], sample['humidity'],
                   c=colors_scatter, alpha=0.4, s=8)
        ax.set(title=f'Temp vs Humidity (Red = {anomaly_plot_label})',
               xlabel='Temperature (°C)', ylabel='Humidity (%)')

    fig.suptitle('Anomaly Detection Analysis — 3 Methods + Consensus',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig_path = _save(fig, '05_anomaly_detection')

    summary = {
        'total_records': len(df2),
        'anomalies_iqr': int(df2['anomaly_iqr'].sum()),
        'anomalies_isolation_forest': int(df2['anomaly_isoforest'].sum()),
        'anomalies_zscore_gt_3std': int(df2['anomaly_zscore'].sum()),
        f'anomalies_consensus_at_least_{consensus_threshold}_methods': int(df2['anomaly_consensus'].sum()),
    }
    return df2, fig_path, summary


def plot_anomaly_countries(df: pd.DataFrame, analysis_flag: str = None, top_n: int = 15) -> str:
    if analysis_flag is None:
        if 'anomaly_consensus' in df.columns and df['anomaly_consensus'].sum() > 0:
            analysis_flag = 'anomaly_consensus'
        else:
            analysis_flag = 'anomaly_isoforest'

    if analysis_flag not in df.columns:
        print(f"No anomaly column '{analysis_flag}' in DataFrame")
        return None

    if 'country' not in df.columns:
        print('No country column'); return None

    country_anomaly = (df.groupby('country')[analysis_flag].mean() * 100).nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    country_anomaly.plot(kind='barh', ax=ax, alpha=0.85, color='C3')
    label = 'Consensus Anomaly' if analysis_flag == 'anomaly_consensus' else 'Isolation Forest Anomaly'
    ax.set(title=f'Top {top_n} Countries by {label} Rate', xlabel='% of Records')
    for i, value in enumerate(country_anomaly.values):
        ax.text(value, i, f' {value:.1f}%', va='center', fontsize=10)

    return _save(fig, '05_anomaly_detection_countries')
