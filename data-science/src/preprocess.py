try:
    import pandas as pd  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        'pandas is required for preprocess.py. Install it with `pip install pandas`.'
    ) from exc
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Data loading
def load_data(path: str = "data/GlobalWeatherRepository.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


# Parse datetime columns
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
    df['date']   = df['last_updated'].dt.date
    df['year']   = df['last_updated'].dt.year
    df['month']  = df['last_updated'].dt.month
    df['day']    = df['last_updated'].dt.day
    df['hour']   = df['last_updated'].dt.hour
    df['dayofyear'] = df['last_updated'].dt.dayofyear
    df['week']   = df['last_updated'].dt.isocalendar().week.astype(int)
    df['season'] = df['month'].map({
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Autumn',10:'Autumn',11:'Autumn'
    })
    return df


# Handle missing values
def handle_missing(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    miss_pct = df.isnull().mean()

    if verbose:
        missing = miss_pct[miss_pct > 0].sort_values(ascending=False)
        print(f"\nMissing values ({len(missing)} columns affected):")
        print(missing.head(20).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols     = df.select_dtypes(include=['object', 'category']).columns

    # Drop high-missing columns (>=40%)
    drop_cols = miss_pct[miss_pct >= 0.40].index.tolist()
    if drop_cols and verbose:
        print(f"\nDropping {len(drop_cols)} high-missing columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Impute numerics
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Impute categoricals
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) else 'Unknown', inplace=True)

    print(f"\nAfter imputation: {df.isnull().sum().sum()} total nulls remain")
    return df


# Handle outliers
def handle_outliers(df: pd.DataFrame,
                    cols: list = None,
                    method: str = 'iqr',
                    action: str = 'clip') -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = ['temperature_celsius', 'wind_mph', 'humidity',
                'pressure_mb', 'precip_mm', 'vis_km',
                'feelslike_celsius', 'dewpoint_celsius',
                'windchill_celsius', 'heatindex_celsius']
        cols = [c for c in cols if c in df.columns]

    outlier_counts = {}
    for col in cols:
        if method == 'iqr':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:  # zscore
            z = np.abs(stats.zscore(df[col].dropna()))
            lower = df[col].mean() - 3 * df[col].std()
            upper = df[col].mean() + 3 * df[col].std()

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_counts[col] = mask.sum()

        if action == 'clip':
            df[col] = df[col].clip(lower, upper)
        else:
            df = df[~mask]

    print("\nOutlier counts (before handling):")
    for col, cnt in outlier_counts.items():
        print(f"  {col}: {cnt:,}")

    return df


# Normalization
def normalize(df: pd.DataFrame,
              cols: list = None,
              method: str = 'standard') -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude id/year/month/day columns
        cols = [c for c in cols if c not in
                ['year','month','day','hour','week','dayofyear']]

    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    scaled = scaler.fit_transform(df[cols].fillna(0))
    for i, col in enumerate(cols):
        df[f'{col}_norm'] = scaled[:, i]

    print(f"\nNormalized {len(cols)} columns using {method} scaling")
    return df, scaler


# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temperature range (diurnal spread proxy)
    if 'temperature_celsius' in df.columns and 'feelslike_celsius' in df.columns:
        df['temp_feels_diff'] = df['temperature_celsius'] - df['feelslike_celsius']

    # Wind × humidity discomfort index
    if 'wind_mph' in df.columns and 'humidity' in df.columns:
        df['discomfort_index'] = df['wind_mph'] * df['humidity'] / 100

    # Heat index approximation
    if 'temperature_celsius' in df.columns and 'humidity' in df.columns:
        T = df['temperature_celsius']
        H = df['humidity']
        df['heat_index_approx'] = (-8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * H
            - 0.14611605 * T * H
            - 0.012308094 * T**2
            - 0.0164248277778 * H**2
            + 0.002211732 * T**2 * H
            + 0.00072546 * T * H**2
            - 0.000003582 * T**2 * H**2)

    # Pressure tendency category
    if 'pressure_mb' in df.columns:
        df['pressure_category'] = pd.cut(
            df['pressure_mb'],
            bins=[0, 980, 1000, 1013, 1030, 9999],
            labels=['Very Low','Low','Normal','High','Very High']
        )

    # Visibility category
    if 'vis_km' in df.columns:
        df['visibility_category'] = pd.cut(
            df['vis_km'],
            bins=[0, 1, 5, 10, 50, 9999],
            labels=['Fog','Mist','Moderate','Good','Excellent']
        )

    print(f"\nEngineered features. Shape: {df.shape}")
    return df


# Full preprocessing pipeline
def full_pipeline(path: str = "data/GlobalWeatherRepository.csv") -> pd.DataFrame:
    df = load_data(path)
    df = parse_datetime(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    print(f"\nPreprocessing complete. Final shape: {df.shape}")
    return df
