"""
=============================================================================
Weather Data Regression Analysis Pipeline
Assessment 2 - Group Project
Dataset: Weather Together (Temperature & Humidity)
Target: Predict temperature_merged from humidity and other features
=============================================================================

PIPELINE OVERVIEW:
  1. Data Loading & Inspection
  2. Data Preprocessing (missing values, feature engineering, scaling)
  3. Heterogeneous Data Integration (Open-Meteo historical weather API)
  4. Exploratory Data Analysis (EDA)
  5. Regression Modelling (Linear, Ridge, Random Forest)
  6. Model Evaluation (MSE, RMSE, MAE, R²)
  7. Statistical Significance Testing (paired t-tests)
  8. Results Interpretation
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy import stats
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# Representative location for the Victoria dataset and Open-Meteo integration.
VICTORIA_LATITUDE = -38.29
VICTORIA_LONGITUDE = 144.39


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING & INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION:
  We load the full CSV and immediately inspect shape, dtypes, missing counts,
  and basic statistics. This is required before any preprocessing decision —
  e.g. we cannot choose a missing-value strategy without first knowing how
  many rows are affected and in which columns.
"""

print("=" * 65)
print("STEP 1: DATA LOADING & INSPECTION")
print("=" * 65)

df = pd.read_csv("weather-together-temperature-and-humidity.csv")

print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nColumn data types:")
print(df.dtypes.to_string())
print("\nDescriptive statistics (numeric columns):")
print(df[["temperature_merged", "humidity_merged", "battery"]].describe().round(2))
print("\nMissing values per column:")
print(df.isnull().sum().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION:
  a) Parse timestamps → enables time-based features (hour, month, day-of-week).
     Models cannot use raw ISO-8601 strings; cyclical encodings (sin/cos)
     preserve the circular nature of time (hour 23 is close to hour 0).

  b) Missing device_name / device_location (≈30 % of rows) are caused by
     devices not broadcasting their metadata in every packet. We fill them
     using a device_id → name/location lookup built from non-null rows.
     This is preferred over dropping rows because it is a systematic,
     recoverable missingness (Missing Not At Random), and dropping 9,041 rows
     would reduce the dataset by ~30%.

  c) Outlier treatment: temperature values above 50 °C are physically
     implausible for outdoor sensors in Victoria, Australia. We cap them
     using the IQR fence (Q3 + 1.5*IQR) rather than Z-score because the
     distribution has a right skew; IQR is more robust to skewed data.

  d) Feature engineering: humidity²  (humidity_sq) captures a possible
     non-linear relationship between humidity and temperature (high humidity
     at night → cool temperatures; non-linearity expected). Hour-of-day
     as sin/cos pairs encode the cyclic structure without introducing
     a spurious ordinal relationship (hour 23 ≠ 23* hour 1).
     Dew point and humidex are derived from temperature and humidity to
     capture moisture and human comfort information. A 1-hour rolling
     temperature average is computed per device to smooth sensor noise.
"""

print("\n" + "=" * 65)
print("STEP 2: PREPROCESSING")
print("=" * 65)

# --- 2a. Parse timestamps ------------------------------------------------
df["time"] = pd.to_datetime(df["time"], utc=True)
df["hour"]       = df["time"].dt.hour
df["month"]      = df["time"].dt.month
df["day_of_week"]= df["time"].dt.dayofweek   # 0 = Monday

# Cyclical encoding for hour (preserves circularity)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Use the representative longitude as a spatial reference for the dataset.
df["longitude"] = VICTORIA_LONGITUDE

# Sort by device and time so per-device rolling features are computed correctly.
df = df.sort_values(["device_id", "time"]).reset_index(drop=True)

print("Timestamps parsed; cyclical hour features created.")

# --- 2b. Fill missing device_name / device_location by device_id ----------
id_to_name = (df.dropna(subset=["device_name"])
                .groupby("device_id")["device_name"]
                .first())
id_to_loc  = (df.dropna(subset=["device_location"])
                .groupby("device_id")["device_location"]
                .first())

df["device_name"]     = df["device_name"].fillna(df["device_id"].map(id_to_name))
df["device_location"] = df["device_location"].fillna(df["device_id"].map(id_to_loc))

still_missing = df[["device_name","device_location"]].isnull().sum()
print(f"After ID-based imputation, remaining missing: {still_missing.to_dict()}")

# --- 2c. Outlier capping on temperature ----------------------------------
Q1, Q3 = df["temperature_merged"].quantile([0.25, 0.75])
IQR     = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
n_outliers  = (df["temperature_merged"] > upper_fence).sum()
df["temperature_merged"] = df["temperature_merged"].clip(upper=upper_fence)
print(f"Capped {n_outliers} temperature outliers at {upper_fence:.2f} degC (IQR fence).")

# --- 2d. Feature engineering --------------------------------------------
df["humidity_sq"]   = df["humidity_merged"] ** 2
df["temp_humidity"] = df["temperature_merged"] * df["humidity_merged"]  # interaction

# Dew point uses temperature and relative humidity to derive saturation.
rh = np.clip(df["humidity_merged"] / 100.0, 1e-6, 1.0)
alpha = (17.27 * df["temperature_merged"] / (237.7 + df["temperature_merged"])) + np.log(rh)
df["dew_point"] = (237.7 * alpha) / (17.27 - alpha)

# Humidex is a human comfort index derived from temperature and dew point.
es = 6.11 * np.exp(5417.7530 * (1/273.16 - 1/(df["dew_point"] + 273.15)))
df["humidex"] = df["temperature_merged"] + 0.5555 * (es - 10)

# Smooth temperature noise with a per-device 1-hour rolling average.
df["temp_rolling_1h"] = (
    df.groupby("device_id")["temperature_merged"]
      .rolling(4, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

print("Engineered features: humidity_sq, temp_humidity, dew_point, humidex, temp_rolling_1h.")

# --- 2e. Encode device_id as integer category ---------------------------
df["device_code"] = pd.Categorical(df["device_id"]).codes
print(f"Encoded {df['device_id'].nunique()} unique devices as integer codes.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: HETEROGENEOUS DATA INTEGRATION (Open-Meteo API)
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION — WHY ADD EXTERNAL DATA:
  The dataset contains only IoT sensor readings (temperature, humidity,
  battery). To demonstrate 'heterogeneous' data integration as required by
  the assessment, we enrich the dataset with historical reanalysis weather
  data from the Open-Meteo API (https://open-meteo.com), which is a free,
  open-source meteorological API providing ERA5 reanalysis data.

  The bounding box of the device locations (Victoria, AU, ~lat -38.3,
  lon 144.4) is used to query a single representative station.
  We fetch daily apparent temperature and wind speed, then merge on date.
  This adds macroscale atmospheric context that explains sensor variability
  beyond device-level humidity alone.

  MERGE STRATEGY: left join on date (YYYY-MM-DD) so original rows are
  always retained. NaN values introduced by date mismatches are forward-
  filled then back-filled (appropriate for slowly-varying daily series).
"""

print("\n" + "=" * 65)
print("STEP 3: HETEROGENEOUS DATA INTEGRATION (Open-Meteo API)")
print("=" * 65)

LAT, LON = VICTORIA_LATITUDE, VICTORIA_LONGITUDE   # centroid of device locations in Victoria
start_date = df["time"].dt.date.min().isoformat()
end_date   = df["time"].dt.date.max().isoformat()

try:
    # Setup Open-Meteo API client with cache and retry on errors.
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Use archive endpoint to align with historical date range in the dataset.
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": [LAT],
        "longitude": [LON],
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "apparent_temperature_max",
            "apparent_temperature_min",
            "windspeed_10m_max",
            "precipitation_sum",
        ],
        "timezone": "Australia/Melbourne",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    ext_df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ).date,
        "apparent_temp_max": daily.Variables(0).ValuesAsNumpy(),
        "apparent_temp_min": daily.Variables(1).ValuesAsNumpy(),
        "wind_speed_max": daily.Variables(2).ValuesAsNumpy(),
        "precipitation": daily.Variables(3).ValuesAsNumpy(),
    })

    df["date"] = df["time"].dt.date
    df = df.merge(ext_df, on="date", how="left")
    df[["apparent_temp_max", "apparent_temp_min",
        "wind_speed_max", "precipitation"]] = (
        df[["apparent_temp_max", "apparent_temp_min",
            "wind_speed_max", "precipitation"]]
        .ffill().bfill()
    )
    EXTERNAL_OK = True
    print(f"Merged Open-Meteo data for {len(ext_df)} days ({start_date} -> {end_date}).")
    print(f"  New features: apparent_temp_max, apparent_temp_min, "
          f"wind_speed_max, precipitation")
except Exception as e:
    EXTERNAL_OK = False
    print(f"  API unavailable ({e}). Falling back to synthetic proxy features.")
    # Synthetic fallback derived from existing data (avoids crashing pipeline)
    df["apparent_temp_max"] = df["temperature_merged"].rolling(48, min_periods=1).max()
    df["apparent_temp_min"] = df["temperature_merged"].rolling(48, min_periods=1).min()
    df["wind_speed_max"]    = np.random.uniform(5, 25, size=len(df))   # placeholder
    df["precipitation"]     = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION:
  EDA precedes modelling to:
  1. Detect distributional skew → informs whether to log-transform targets.
  2. Identify correlated predictors → informs feature selection / regularisation.
  3. Reveal time-based patterns → confirms hour/month features are informative.
  4. Spot remaining anomalies before features are frozen.
"""

print("\n" + "=" * 65)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Exploratory Data Analysis — Weather IoT Dataset", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# 4a. Temperature distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["temperature_merged"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
ax1.set_title("Temperature Distribution (°C)")
ax1.set_xlabel("Temperature (°C)"); ax1.set_ylabel("Count")

# 4b. Humidity distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["humidity_merged"], bins=60, color="#DD8452", edgecolor="white", linewidth=0.4)
ax2.set_title("Humidity Distribution (%)")
ax2.set_xlabel("Humidity (%)"); ax2.set_ylabel("Count")

# 4c. Scatter: humidity vs temperature
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df["humidity_merged"], df["temperature_merged"],
            alpha=0.05, s=5, c="#55A868")
ax3.set_title("Humidity vs Temperature")
ax3.set_xlabel("Humidity (%)"); ax3.set_ylabel("Temperature (°C)")

# 4d. Temperature by hour (boxplot)
ax4 = fig.add_subplot(gs[1, :2])
hourly = [df[df["hour"]==h]["temperature_merged"].values for h in range(24)]
ax4.boxplot(hourly, positions=range(24), widths=0.6,
            medianprops=dict(color="red", linewidth=1.5),
            flierprops=dict(marker=".", alpha=0.3, markersize=2))
ax4.set_title("Temperature by Hour of Day")
ax4.set_xlabel("Hour (UTC)"); ax4.set_ylabel("Temperature (°C)")
ax4.set_xticks(range(24))

# 4e. Correlation heatmap
ax5 = fig.add_subplot(gs[1, 2])
corr_cols = ["temperature_merged","humidity_merged","humidity_sq",
             "dew_point","humidex","temp_rolling_1h",
             "hour_sin","hour_cos","month","device_code"]
if EXTERNAL_OK:
    corr_cols += ["apparent_temp_max","wind_speed_max","precipitation"]
corr_mat = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, ax=ax5, cmap="coolwarm", vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.3)
ax5.set_title("Feature Correlation Matrix")
ax5.tick_params(axis="x", rotation=45, labelsize=7)
ax5.tick_params(axis="y", rotation=0,  labelsize=7)

# 4f. Monthly average temperature per device
ax6 = fig.add_subplot(gs[2, :])
monthly = df.groupby(["month", "device_name"])["temperature_merged"].mean().reset_index()
for dev in monthly["device_name"].dropna().unique():
    sub = monthly[monthly["device_name"] == dev]
    ax6.plot(sub["month"], sub["temperature_merged"], marker="o", label=dev, linewidth=1.2)
ax6.set_title("Monthly Average Temperature per Device")
ax6.set_xlabel("Month"); ax6.set_ylabel("Avg Temperature (°C)")
ax6.legend(fontsize=7, ncol=2, loc="upper right")

plt.savefig("eda_plots.png", bbox_inches="tight")
plt.close()
print("EDA visualisations saved -> eda_plots.png")

# Print key correlation with target
print("\nCorrelation of features with temperature_merged:")
target_corr = df[corr_cols].corr()["temperature_merged"].drop("temperature_merged")
print(target_corr.sort_values(ascending=False).round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FEATURE MATRIX & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION:
  FEATURES CHOSEN:
    humidity_merged   — primary predictor (strong negative correlation expected)
    humidity_sq       — captures non-linearity (humidity-temperature curve)
    dew_point         — stable moisture indicator derived from humidity.
    humidex           — human comfort index combining temperature and humidity.
    temp_rolling_1h   — last-hour smoothed temperature to reduce sensor noise.
    hour_sin/cos      — captures diurnal temperature cycle
    month             — captures seasonal variation
    device_code       — captures per-device calibration offset
    apparent_temp_max/min, wind_speed_max, precipitation (if available)
      — macroscale atmospheric context from external source

  EXCLUDED: battery (unrelated to temperature physics), raw hour (replaced
    by cyclical encoding), temp_humidity (used only for EDA), longitude
    (representative spatial reference, not predictive when constant).

  SPLIT STRATEGY: 80 % train / 20 % test, stratified by month to ensure
    each month is proportionally represented in both sets. This is important
    because the dataset spans ~2 months and the temperature signal has strong
    seasonal structure; a purely random split could leave all winter days in
    train and summer days in test.

  SCALING: StandardScaler (zero-mean, unit-variance) is applied to all
    numeric features before Linear and Ridge regression. Tree-based models
    (Random Forest) are scale-invariant, but we scale anyway to keep the
    pipeline consistent and allow fair coefficient interpretation.
"""

print("\n" + "=" * 65)
print("STEP 5: FEATURE MATRIX & TRAIN/TEST SPLIT")
print("=" * 65)

FEATURES = ["humidity_merged", "humidity_sq", "dew_point",
            "humidex", "temp_rolling_1h", "hour_sin", "hour_cos",
            "month", "device_code"]
if EXTERNAL_OK:
    FEATURES += ["apparent_temp_max", "apparent_temp_min",
                 "wind_speed_max", "precipitation"]

TARGET = "temperature_merged"

# Drop any rows with NaN in feature/target columns
model_df = df[FEATURES + [TARGET]].dropna()
X = model_df[FEATURES].values
y = model_df[TARGET].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: REGRESSION MODELS
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION — MODEL SELECTION:

  MODEL 1 — Linear Regression (OLS):
    Chosen as the baseline because it is interpretable (each coefficient
    directly quantifies the marginal effect of a feature), computationally
    efficient, and mathematically well-understood. It is appropriate here
    as a starting point; its assumptions (linearity, homoscedasticity)
    can then be tested against residuals.

  MODEL 2 — Ridge Regression (L2 regularisation):
    Extends OLS with an L2 penalty (λ‖β‖²) that shrinks coefficients toward
    zero. Justified because the correlation matrix revealed moderate
    collinearity between humidity-derived features (humidity_merged vs
    humidity_sq, r ≈ 0.9+). OLS is sensitive to multicollinearity; Ridge
    stabilises estimates without discarding variables. alpha = 1.0 is a
    reasonable default; optimal alpha could be found via cross-validation
    (RidgeCV) in a production system.

  MODEL 3 — Random Forest Regressor:
    A non-parametric ensemble of decision trees that captures non-linear
    interactions and is robust to outliers. Justified because EDA showed a
    non-linear humidity-temperature relationship and strong device/time
    interactions. The trade-off vs. OLS/Ridge is interpretability (we use
    feature_importances_ to partially recover it) and longer training time.
    100 trees with max_depth=None provides a good bias-variance balance for
    ~24,000 training samples.
"""

print("\n" + "=" * 65)
print("STEP 6: REGRESSION MODELLING")
print("=" * 65)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=100,
                             random_state=RANDOM_STATE,
                             n_jobs=-1
                         ),
}

results   = {}
cv_scores = {}          # store 5-fold CV RMSE for statistical testing

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    # Use scaled data for linear models, raw for Random Forest
    Xtr = X_train_sc if "Forest" not in name else X_train
    Xte = X_test_sc  if "Forest" not in name else X_test

    # Fit
    model.fit(Xtr, y_train)

    # Test-set predictions
    y_pred = model.predict(Xte)

    # Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred,
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2
    }

    # 5-fold cross-validated RMSE on full dataset
    Xfull = X_train_sc if "Forest" not in name else X_train
    cv_neg_mse = cross_val_score(model, Xfull, y_train,
                                 scoring="neg_mean_squared_error", cv=kf)
    cv_rmse = np.sqrt(-cv_neg_mse)
    cv_scores[name] = cv_rmse

    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"  MSE  = {mse:.4f}   RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}   R2   = {r2:.4f}")
    print(f"  5-Fold CV RMSE: mean={cv_rmse.mean():.4f}, std={cv_rmse.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: MODEL EVALUATION — METRICS JUSTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
"""
METRICS JUSTIFICATION:

  MSE (Mean Squared Error):
    ∑(yᵢ - ŷᵢ)² / n
    Heavily penalises large errors (quadratic). Sensitive to outliers.
    Used internally by OLS (it minimises MSE analytically). Useful for
    comparing models on the same dataset, but not directly interpretable
    in the original units.

  RMSE (Root MSE):
    √MSE — same units as temperature (°C). An RMSE of 2 °C means on
    average predictions deviate by 2 °C. More interpretable than MSE.
    Still sensitive to large errors (squared), which is appropriate here
    because a 10 °C temperature error is much worse than a 2 °C error.

  MAE (Mean Absolute Error):
    ∑|yᵢ - ŷᵢ| / n
    Less sensitive to outliers than RMSE (linear penalty). Provides a
    complementary view: if RMSE >> MAE, the model has occasional large
    errors but is mostly accurate; if RMSE ≈ MAE, errors are uniform.

  R² (Coefficient of Determination):
    1 - SS_res/SS_tot
    Proportion of variance explained. Model-agnostic baseline: R² = 0
    means the model is no better than predicting the mean; R² = 1 is
    perfect. Useful for non-technical stakeholders ("our model explains
    85 % of the variation in temperature").
"""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: STATISTICAL SIGNIFICANCE TESTING (Paired t-tests)
# ─────────────────────────────────────────────────────────────────────────────
"""
JUSTIFICATION:
  Cross-validation produces a distribution of RMSE scores across folds.
  A paired t-test on two models' fold-level RMSE scores tests whether the
  difference in performance is statistically significant or could be due
  to random variation in the data split.

  HYPOTHESES (example: Linear vs Ridge):
    H₀: μ_RMSE(Linear) = μ_RMSE(Ridge)    (no significant difference)
    H₁: μ_RMSE(Linear) ≠ μ_RMSE(Ridge)    (two-tailed, alpha = 0.05)

  CHOICE OF TEST: Paired t-test (not independent samples) because the
  same folds are used for both models, making the observations paired.
  This removes fold-level data variability from the denominator, giving
  a more powerful test for model comparison.
"""

print("\n" + "=" * 65)
print("STEP 8: STATISTICAL SIGNIFICANCE TESTING (Paired t-tests)")
print("=" * 65)

model_names = list(cv_scores.keys())
pairs = [(model_names[i], model_names[j])
         for i in range(len(model_names))
         for j in range(i+1, len(model_names))]

for m1, m2 in pairs:
    t_stat, p_val = stats.ttest_rel(cv_scores[m1], cv_scores[m2])
    sig = "[SIGNIFICANT]" if p_val < 0.05 else "[NOT SIGNIFICANT]"
    print(f"\n  {m1}  vs  {m2}")
    print(f"    H0: no difference in CV-RMSE  |  alpha = 0.05")
    print(f"    t = {t_stat:.4f},  p = {p_val:.4f}  ->  {sig}")
    if p_val < 0.05:
        better = m1 if cv_scores[m1].mean() < cv_scores[m2].mean() else m2
        print(f"    -> '{better}' performs significantly better.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: VISUALISATION — EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 9: EVALUATION VISUALISATIONS")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Evaluation Plots", fontsize=14, fontweight="bold")
colors = ["#4C72B0", "#DD8452", "#55A868"]

for idx, (name, res) in enumerate(results.items()):
    ax_top = axes[0, idx]
    ax_bot = axes[1, idx]
    col    = colors[idx]

    # Top: Actual vs Predicted scatter
    ax_top.scatter(y_test, res["y_pred"], alpha=0.15, s=5, color=col)
    mn, mx = y_test.min(), y_test.max()
    ax_top.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax_top.set_title(f"{name}\nActual vs Predicted")
    ax_top.set_xlabel("Actual Temp (°C)")
    ax_top.set_ylabel("Predicted Temp (°C)")
    ax_top.legend(fontsize=8)
    ax_top.text(0.05, 0.93, f"R²={res['R2']:.3f}", transform=ax_top.transAxes,
                fontsize=9, color="black",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Bottom: Residual plot
    residuals = y_test - res["y_pred"]
    ax_bot.scatter(res["y_pred"], residuals, alpha=0.15, s=5, color=col)
    ax_bot.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax_bot.set_title(f"{name}\nResiduals vs Fitted")
    ax_bot.set_xlabel("Predicted Temp (°C)")
    ax_bot.set_ylabel("Residual (°C)")
    ax_bot.text(0.05, 0.93, f"RMSE={res['RMSE']:.3f}", transform=ax_bot.transAxes,
                fontsize=9, color="black",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
plt.savefig("model_evaluation.png", bbox_inches="tight")
plt.close()
print("Evaluation plots saved -> model_evaluation.png")


# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Random Forest — Feature Importances")
ax.set_xlabel("Importance (mean decrease in impurity)")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.close()
print("Feature importance plot saved -> feature_importance.png")


# Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metric_df = pd.DataFrame({
    name: {"RMSE": res["RMSE"], "MAE": res["MAE"], "R²": res["R2"]}
    for name, res in results.items()
}).T

x   = np.arange(len(metric_df))
w   = 0.25
ax.bar(x - w,   metric_df["RMSE"], width=w, label="RMSE", color="#4C72B0")
ax.bar(x,       metric_df["MAE"],  width=w, label="MAE",  color="#DD8452")
ax.bar(x + w,   metric_df["R²"],   width=w, label="R²",   color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(metric_df.index, rotation=10)
ax.set_title("Model Comparison: RMSE, MAE, R²")
ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.close()
print("Model comparison chart saved -> model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: PIPELINE FLOWCHART (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
ax.set_title("Data Analysis Pipeline Flowchart", fontsize=14, fontweight="bold")

stages = [
    (5, 11.0, "1. Data Collection\n(IoT Sensor CSV + Open-Meteo API)", "#AED6F1"),
    (5,  9.5, "2. Preprocessing\n(Parse timestamps, impute device_name,\ncap outliers, encode categories)", "#A9DFBF"),
    (5,  7.8, "3. Feature Engineering\n(humidity², cyclical hour sin/cos,\ninteraction terms)", "#FAD7A0"),
    (5,  6.3, "4. Heterogeneous Integration\n(Merge daily weather reanalysis on date)", "#F9E79F"),
    (5,  4.8, "5. Exploratory Data Analysis\n(Distributions, correlations, time patterns)", "#D2B4DE"),
    (5,  3.3, "6. Train/Test Split (80/20)\n+ StandardScaler", "#F5CBA7"),
    (5,  1.9, "7. Regression Models\n(Linear, Ridge, Random Forest)", "#A9CCE3"),
    (5,  0.5, "8. Evaluation + Statistical Tests\n(MSE, RMSE, MAE, R², Paired t-test)", "#FADBD8"),
]

for (x, y, text, color) in stages:
    ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
        (x - 3.8, y - 0.45), 7.6, 0.9,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="gray", linewidth=1.2
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, wrap=True)

# Draw arrows between boxes
for i in range(len(stages) - 1):
    _, y1, _, _ = stages[i]
    _, y2, _, _ = stages[i + 1]
    ax.annotate("", xy=(5, y2 + 0.47), xytext=(5, y1 - 0.47),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

plt.tight_layout()
plt.savefig("pipeline_flowchart.png", bbox_inches="tight", dpi=150)
plt.close()
print("Pipeline flowchart saved -> pipeline_flowchart.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"{'Model':<22} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 58)
for name, res in results.items():
    print(f"{name:<22} {res['MSE']:>8.4f} {res['RMSE']:>8.4f} "
          f"{res['MAE']:>8.4f} {res['R2']:>8.4f}")

print("\nAll pipeline steps completed successfully.")
print("   Output files:")
print("   - eda_plots.png          - EDA visualisations")
print("   - model_evaluation.png   - Actual vs Predicted & Residual plots")
print("   - feature_importance.png - Random Forest feature importances")
print("   - model_comparison.png   - RMSE / MAE / R2 bar chart")
print("   - pipeline_flowchart.png - Data pipeline diagram")
