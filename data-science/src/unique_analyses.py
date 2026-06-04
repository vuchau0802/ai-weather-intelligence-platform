import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

FIG_DIR = 'outputs/figures'

def add_continent_region(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if 'continent_region' in df2.columns:
        return df2

    def infer_region(row):
        timezone = str(row.get('timezone', ''))
        latitude = row.get('latitude', np.nan)
        longitude = row.get('longitude', np.nan)

        if timezone.startswith('Africa/'):
            return 'Africa'
        if timezone.startswith('Europe/'):
            return 'Europe'
        if timezone.startswith('Asia/'):
            return 'Asia'
        if timezone.startswith(('Australia/', 'Pacific/')):
            return 'Oceania'
        if timezone.startswith('Antarctica/'):
            return 'Antarctica'
        if timezone.startswith('America/'):
            if pd.notna(latitude) and pd.notna(longitude):
                if -90 < longitude < -30 and latitude < 13:
                    return 'South America'
            return 'North America'
        if timezone in {'UTC', 'Etc/UTC'}:
            return 'Global/UTC'
        return 'Other'

    df2['continent_region'] = df2.apply(infer_region, axis=1)
    return df2

def _save(fig, name):
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f'{FIG_DIR}/{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# Climate analysis

def climate_analysis(df: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Seasonal temperature violin
    ax = fig.add_subplot(gs[0, 0])
    if 'season' in df.columns and 'temperature_celsius' in df.columns:
        order = ['Winter','Spring','Summer','Autumn']
        order = [s for s in order if s in df['season'].unique()]
        data_by_season = [df.loc[df['season']==s,'temperature_celsius'].dropna()
                          for s in order]
        vp = ax.violinplot(data_by_season, showmedians=True, showextrema=True)
        for i, pc in enumerate(vp['bodies']):
            pc.set_alpha(0.8)
        ax.set(title='Temperature by Season', ylabel='°C',
               xticks=range(1, len(order)+1), xticklabels=order)

    # Monthly climate normals
    ax1 = fig.add_subplot(gs[0, 1])
    if 'month' in df.columns:
        monthly_temp  = df.groupby('month')['temperature_celsius'].mean() \
                          if 'temperature_celsius' in df.columns else None
        monthly_prec  = df.groupby('month')['precip_mm'].mean() \
                          if 'precip_mm' in df.columns else None
        months = ['J','F','M','A','M','J','J','A','S','O','N','D']
        all_months = pd.Series(range(1, 13), name='month')
        if monthly_temp is not None:
            monthly_temp = monthly_temp.reindex(range(1, 13))
            ax1.plot(range(1,13), monthly_temp, 'o-',
                     lw=2, label='Avg Temp (°C)')
            ax1.fill_between(range(1,13), monthly_temp.values,
                             alpha=0.15)
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_xticks(range(1,13)); ax1.set_xticklabels(months)
        ax2 = ax1.twinx()
        if monthly_prec is not None:
            monthly_prec = monthly_prec.reindex(range(1, 13)).fillna(0)
            ax2.bar(range(1,13), monthly_prec.values, alpha=0.4,
                    label='Avg Precip (mm)')
            ax2.set_ylabel('Precipitation (mm)')
        ax1.set_title('Monthly Climate Normals')
        ax1.legend(loc='upper left'); ax2.legend(loc='upper right')

    # Climate zones
    ax = fig.add_subplot(gs[0, 2])
    if 'country' in df.columns and 'temperature_celsius' in df.columns:
        country_temp = df.groupby('country')['temperature_celsius'].mean()
        bins = country_temp.quantile([0, 0.25, 0.5, 0.75, 1.0])
        zones = pd.cut(country_temp,
                       bins=[bins.iloc[0]-0.1, bins.iloc[1], bins.iloc[2],
                              bins.iloc[3], bins.iloc[4]+0.1],
                       labels=['Cold','Cool-Temperate','Warm-Temperate','Hot'])
        zone_counts = zones.value_counts()
        zone_counts.plot(kind='pie', ax=ax,
                         autopct='%1.1f%%', startangle=90)
        ax.set(title='Country Climate Zone Distribution', ylabel='')

    # Temperature variability
    ax = fig.add_subplot(gs[1, 0])
    if 'year' in df.columns and 'temperature_celsius' in df.columns:
        yearly = df.groupby('year')['temperature_celsius'].agg(['mean','std'])
        if len(yearly) > 1:
            ax.errorbar(yearly.index, yearly['mean'],
                        yerr=yearly['std'], fmt='o-',
                        capsize=4, lw=2)
            z = np.polyfit(yearly.index, yearly['mean'], 1)
            trend_line = np.polyval(z, yearly.index)
            ax.plot(yearly.index, trend_line, '--',
                    lw=1.5, label=f'Trend: {z[0]:+.3f}°C/yr')
            ax.set(title='Annual Temperature with Trend', ylabel='°C')
            ax.legend()

    # Humidity vs temperature
    ax = fig.add_subplot(gs[1, 1])
    if 'temperature_celsius' in df.columns and 'humidity' in df.columns:
        if 'season' in df.columns:
            for season in ['Winter','Spring','Summer','Autumn']:
                if season not in df['season'].unique():
                    continue
                mask = df['season'] == season
                sample = df[mask].sample(min(800, mask.sum()), random_state=42)
                ax.scatter(sample['temperature_celsius'], sample['humidity'],
                           alpha=0.4, s=8, label=season)
            ax.legend(markerscale=3, fontsize=9)
        ax.set(title='Temperature vs Humidity by Season',
               xlabel='°C', ylabel='Humidity %')

    # Wind by month
    ax = fig.add_subplot(gs[1, 2], projection='polar') \
        if 'wind_direction' not in df.columns else fig.add_subplot(gs[1, 2])
    if 'wind_mph' in df.columns and 'month' in df.columns:
        monthly_wind = df.groupby('month')['wind_mph'].mean()
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        radii = monthly_wind.values
        width = 2*np.pi / 12
        try:
            bars = ax.bar(theta, radii, width=width, alpha=0.7)
            ax.set_xticks(theta)
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax.set_title('Monthly Wind Speed (mph)', pad=20)
        except Exception:
            monthly_wind.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set(title='Monthly Wind Speed (mph)')

    fig.suptitle('Climate Analysis — Long-term Patterns & Seasonal Variations',
                 fontsize=15, fontweight='bold')
    return _save(fig, '08_climate_analysis')


# Air quality analysis

def air_quality_analysis(df: pd.DataFrame) -> str:
    aq_cols = [c for c in df.columns if 'air_quality' in c.lower()]
    if not aq_cols:
        print("  No air quality columns found; skipping."); return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    weather_feats = ['temperature_celsius','humidity','wind_mph',
                     'pressure_mb','precip_mm','vis_km']
    weather_feats = [c for c in weather_feats if c in df.columns]
    primary_aq = aq_cols[0]  # e.g. air_quality_PM2.5

    # AQ distribution
    ax = axes[0, 0]
    df[primary_aq].dropna().hist(ax=ax, bins=50, alpha=0.8,
                                   edgecolor='white')
    ax.set(title=f'{primary_aq} Distribution', xlabel=primary_aq, ylabel='Count')
    ax.axvline(df[primary_aq].mean(), color='black', lw=2, linestyle='--',
               label=f"Mean: {df[primary_aq].mean():.1f}")
    ax.legend()

    # AQ vs weather
    ax = axes[0, 1]
    corrs = df[weather_feats + [primary_aq]].corr()[primary_aq].drop(primary_aq).abs()
    top3 = corrs.nlargest(3).index.tolist()
    if len(top3) >= 2:
        scatter = ax.scatter(df[top3[0]].fillna(0), df[primary_aq].fillna(0),
                             c=df[top3[1]].fillna(0), cmap='RdYlGn_r',
                             alpha=0.3, s=10)
        plt.colorbar(scatter, ax=ax, label=top3[1])
        ax.set(title=f'Top Correlate: {top3[0]} vs {primary_aq}',
               xlabel=top3[0], ylabel=primary_aq)

    # Correlation bar
    ax = axes[1, 0]
    corr_vals = df[weather_feats + [primary_aq]].corr()[primary_aq].drop(primary_aq)
    ax.barh(corr_vals.index, corr_vals.values, alpha=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set(title=f'{primary_aq} Correlation with Weather Features',
           xlabel='Pearson r')

    # AQ heatmap
    ax = axes[1, 1]
    if 'month' in df.columns and 'hour' in df.columns:
        pivot = df.pivot_table(values=primary_aq, index='hour',
                               columns='month', aggfunc='mean')
        if not pivot.empty:
            sns.heatmap(pivot, ax=ax, cmap='YlOrRd', fmt='.0f',
                        cbar_kws={'label': primary_aq})
            ax.set(title=f'{primary_aq} by Hour × Month',
                   xlabel='Month', ylabel='Hour')

    fig.suptitle('Environmental Impact — Air Quality × Weather Analysis',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '09_air_quality_analysis')


# Feature importance analysis
def feature_importance_analysis(df: pd.DataFrame,
                                 target: str = 'temperature_celsius') -> str:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target, 'year', 'month', 'day', 'hour', 'week', 'dayofyear',
               'last_updated'] + [c for c in numeric_cols if '_norm' in c]
    feature_cols = [c for c in numeric_cols if c not in exclude and c in df.columns]
    if not feature_cols:
        print("  No features found"); return None

    data = df[feature_cols + [target]].dropna()
    X, y = data[feature_cols], data[target]
    sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
    X_s, y_s = X.iloc[sample_idx], y.iloc[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_s, y_s)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(15).sort_values()
    axes[0].barh(rf_imp.index, rf_imp.values)
    axes[0].set(title='Random Forest\nFeature Importance', xlabel='Importance')

    # Permutation importance
    perm = permutation_importance(rf, X_s, y_s, n_repeats=10, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=feature_cols).nlargest(15).sort_values()
    axes[1].barh(perm_imp.index, perm_imp.values,
                 xerr=perm.importances_std[
                     [feature_cols.index(f) for f in perm_imp.index]],
                 alpha=0.8, capsize=3)
    axes[1].set(title='Permutation Importance\n(±std)', xlabel='Mean Decrease in R²')

    # Gradient Boosting importance
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_s, y_s)
    gb_imp = pd.Series(gb.feature_importances_, index=feature_cols).nlargest(15).sort_values()
    axes[2].barh(gb_imp.index, gb_imp.values)
    axes[2].set(title='Gradient Boosting\nFeature Importance', xlabel='Importance')

    fig.suptitle(f'Feature Importance Analysis — Target: {target}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '10_feature_importance_analysis')


# Spatial analysis

def spatial_analysis(df: pd.DataFrame) -> str:
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    temp_col = 'temperature_celsius'

    if not (lat_col and lon_col and temp_col in df.columns):
        print("  Missing lat/lon/temp columns; skipping."); return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Temperature scatter
    ax = axes[0]
    sample = df.dropna(subset=[lat_col, lon_col, temp_col]) \
               .sample(min(10000, len(df)), random_state=42)
    scatter = ax.scatter(sample[lon_col], sample[lat_col],
                         c=sample[temp_col], cmap='RdYlBu_r',
                         alpha=0.5, s=6, vmin=-20, vmax=45)
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)', shrink=0.7)
    ax.set(title='Global Temperature Distribution (Scatter)',
           xlabel='Longitude', ylabel='Latitude',
           xlim=(-180, 180), ylim=(-90, 90))
    ax.axhline(0, color='gray', lw=0.5, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2)

    # Humidity scatter
    ax = axes[1]
    hum_col = 'humidity'
    if hum_col in df.columns:
        sample2 = df.dropna(subset=[lat_col, lon_col, hum_col]) \
                    .sample(min(10000, len(df)), random_state=99)
        scatter2 = ax.scatter(sample2[lon_col], sample2[lat_col],
                              c=sample2[hum_col], cmap='Blues',
                              alpha=0.5, s=6)
        plt.colorbar(scatter2, ax=ax, label='Humidity (%)', shrink=0.7)
        ax.set(title='Global Humidity Distribution',
               xlabel='Longitude', ylabel='Latitude',
               xlim=(-180, 180), ylim=(-90, 90))
        ax.axhline(0, color='gray', lw=0.5, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Spatial Analysis — Geographic Weather Patterns',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    return _save(fig, '11_spatial_analysis')


# Geographical patterns

def geographical_patterns(df: pd.DataFrame) -> str:
    if 'country' not in df.columns:
        return None

    df = add_continent_region(df)

    fig = plt.figure(figsize=(18, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    metrics = ['temperature_celsius','humidity','wind_mph','precip_mm']
    metrics = [m for m in metrics if m in df.columns]

    # Country metrics
    ax = fig.add_subplot(gs[0, :])
    top_countries = df['country'].value_counts().head(20).index.tolist()
    ctry_df = df[df['country'].isin(top_countries)].groupby('country')[metrics].mean()

    x = np.arange(len(top_countries))
    width = 0.8 / len(metrics)

    for i, (col) in enumerate(metrics):
        if col in ctry_df.columns:
            vals = ctry_df.loc[top_countries, col].values
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            ax.bar(x + i * width, vals_norm, width, label=col.replace('_',' ').title(),
                   alpha=0.8, edgecolor='white')

    ax.set(title='Top 20 Countries — Normalized Weather Metrics',
           ylabel='Normalized (0–1)', xticks=x + width * len(metrics) / 2,
           xticklabels=top_countries)
    ax.set_xticklabels(top_countries, rotation=45, ha='right', fontsize=9)
    ax.legend()

    # Temperature variance
    ax = fig.add_subplot(gs[1, 0])
    if 'temperature_celsius' in df.columns:
        var_df = (df.groupby('country')['temperature_celsius']
                    .std()
                    .nlargest(25)
                    .sort_values())
        ax.barh(var_df.index, var_df.values)
        ax.set(title='Top 25 Countries — Temperature Variability (std)',
               xlabel='Std Dev (°C)')

    # Continent comparison
    ax = fig.add_subplot(gs[1, 1])
    region_df = df.groupby('continent_region')[metrics].mean().sort_index()
    region_norm = (region_df - region_df.min()) / (region_df.max() - region_df.min() + 1e-8)
    region_norm.plot(kind='bar', ax=ax, alpha=0.85)
    ax.set(title='Continent/Region Weather Comparison',
           xlabel='Continent/Region', ylabel='Normalized (0-1)')
    ax.set_xticklabels(region_norm.index, rotation=30, ha='right')
    ax.legend(fontsize=8)

    # Weather radar
    top6 = df['country'].value_counts().head(6).index.tolist()
    cntry_profiles = df[df['country'].isin(top6)].groupby('country')[metrics].mean()
    # Normalize
    cntry_norm = (cntry_profiles - cntry_profiles.min()) / \
                 (cntry_profiles.max() - cntry_profiles.min() + 1e-8)

    theta = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])  # close polygon

    try:
        ax_polar = fig.add_subplot(gs[2, :], projection='polar')
        for i, country in enumerate(top6):
            if country in cntry_norm.index:
                vals = cntry_norm.loc[country, metrics].values
                vals = np.concatenate([vals, [vals[0]]])
                ax_polar.plot(theta, vals, lw=2, label=country)
                ax_polar.fill(theta, vals, alpha=0.1)
        ax_polar.set_xticks(theta[:-1])
        ax_polar.set_xticklabels([m.replace('_celsius','').replace('_',' ').title()
                                   for m in metrics], size=9)
        ax_polar.set_title('Weather Profile Radar\n(Top 6 Countries)', pad=20)
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    except Exception:
        # Fallback: simple grouped bar
        ax = fig.add_subplot(gs[2, :])
        cntry_norm.T.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set(title='Weather Profiles — Top 6 Countries',
               xlabel='Metric', ylabel='Normalized')
        ax.set_xticklabels([m.replace('_celsius','') for m in metrics], rotation=30)

    fig.suptitle('Geographical Patterns - Country and Continent/Region Analysis',
                 fontsize=15, fontweight='bold')
    return _save(fig, '12_geographical_patterns')
