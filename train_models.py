"""
Step 2: Train AI Models on Physics-Based Data
===============================================
We train 3 XGBoost models:
  1. Solar output predictor
  2. Wind output predictor
  3. Load demand predictor

WHY XGBoost?
- It's the gold standard for tabular data in industry
- Handles non-linear relationships (like solar's bell curve) perfectly
- Fast to train and fast to predict
- Won more Kaggle competitions than any other algorithm
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

print("=" * 60)
print("  Training AI Models on Physics-Based Dataset")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/processed_dataset.csv", index_col="timestamp", parse_dates=True)
print(f"\n✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── DEFINE FEATURES & TARGETS ─────────────────────────────────────────────────
# FEATURES = the inputs the model uses to make predictions
# Think of these as "clues" the model uses to guess the output

FEATURE_COLS = [
    # Weather inputs (main drivers of solar and wind)
    "solar_irradiance",        # How sunny it is right now
    "wind_speed",              # How windy it is right now
    "temperature",             # Temperature (affects solar efficiency)
    "humidity",                # Humidity
    "pressure",                # Atmospheric pressure

    # Grid state (what's happening in the grid right now)
    "grid_frequency",
    "grid_voltage",
    "grid_exchange",
    "battery_soc",
    "battery_charge",
    "battery_discharge",

    # Time features (the model needs to know WHEN it is)
    "hour",                    # Hour of day (0-23) — critical for solar!
    "day_of_week",             # Day of week (0=Mon, 6=Sun)
    "month",                   # Month (1-12) — critical for seasonal patterns!
    "day_of_year",             # Day of year (1-365)

    # Lag features (what happened 1 hour ago)
    "load_demand_lag1",        # Last hour's demand
    "solar_irradiance_lag1",   # Last hour's sun
    "wind_speed_lag1",         # Last hour's wind

    # Rolling averages (what's the recent trend)
    "wind_speed_roll_3h",      # Average wind speed over last 3 hours
    "solar_irradiance_roll_3h",# Average irradiance over last 3 hours
    "load_roll_3h",            # Average load over last 3 hours
]

# TARGETS = what we want to predict
TARGETS = ["solar_output", "wind_output", "load_demand"]

print(f"\n Features used: {len(FEATURE_COLS)}")
print(f" Targets to predict: {TARGETS}")

# ── TRAIN/TEST SPLIT ──────────────────────────────────────────────────────────
# Use first 9 months for training, last 3 months for testing
# This is called "time-series split" — IMPORTANT: never shuffle time series data!
# If you shuffle, the model sees "future" data during training — that's cheating!

split_date = "2023-10-01"
df_train   = df[df.index < split_date]
df_test    = df[df.index >= split_date]

print(f"\n Train set: {len(df_train)} rows ({df_train.index[0].date()} → {df_train.index[-1].date()})")
print(f" Test set:  {len(df_test)} rows ({df_test.index[0].date()} → {df_test.index[-1].date()})")

X_train = df_train[FEATURE_COLS]
X_test  = df_test[FEATURE_COLS]
y_train = df_train[TARGETS]
y_test  = df_test[TARGETS]

# ── SCALE THE FEATURES ────────────────────────────────────────────────────────
# StandardScaler makes all features have mean=0 and std=1
# This prevents large-valued features (like irradiance ~500) from dominating
# small-valued features (like hour ~12)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)   # Fit AND transform training
X_test_sc   = scaler.transform(X_test)        # ONLY transform test (no fit!)

print("\n✅ Features scaled with StandardScaler")

# ── TRAIN MODELS ─────────────────────────────────────────────────────────────
# XGBoost hyperparameters explained:
#   n_estimators=500     → Build 500 decision trees
#   learning_rate=0.05   → Each tree contributes a small amount (prevents overfitting)
#   max_depth=6          → How deep each tree can go
#   subsample=0.8        → Use 80% of rows per tree (regularization)
#   colsample_bytree=0.8 → Use 80% of features per tree (regularization)
#   early_stopping_rounds=30 → Stop if no improvement after 30 rounds

XGBOOST_PARAMS = dict(
    n_estimators        = 500,
    learning_rate       = 0.05,
    max_depth           = 6,
    min_child_weight    = 3,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    gamma               = 0.1,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    random_state        = 42,
    n_jobs              = -1,
    early_stopping_rounds = 30,
)

trained_models = {}
results        = []

print("\n" + "=" * 60)
for target in TARGETS:
    print(f"\n🚀 Training model for: {target.upper()}")
    print("-" * 40)

    y_tr = y_train[target]
    y_te = y_test[target]

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train_sc, y_tr,
        eval_set  = [(X_test_sc, y_te)],
        verbose   = 100,  # Print progress every 100 trees
    )

    y_pred = model.predict(X_test_sc)
    y_pred = np.maximum(0, y_pred)  # Outputs can't be negative

    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    print(f"\n  📊 {target} Results:")
    print(f"     R² Score : {r2:.4f}  (1.0 = perfect, 0 = useless, <0 = worse than average)")
    print(f"     MAE      : {mae:.2f} kW  (average prediction error)")
    print(f"     RMSE     : {rmse:.2f} kW  (penalises large errors more)")

    trained_models[target] = model
    results.append({"Target": target, "R² Score": r2, "MAE (kW)": mae, "RMSE (kW)": rmse})

# ── RESULTS SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results)
results_df["R² Score"] = results_df["R² Score"].map("{:.4f}".format)
results_df["MAE (kW)"] = results_df["MAE (kW)"].map("{:.2f}".format)
results_df["RMSE (kW)"] = results_df["RMSE (kW)"].map("{:.2f}".format)
print(results_df.to_string(index=False))

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
print("\n Top 5 most important features per model:")
for target in TARGETS:
    model  = trained_models[target]
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    top5   = importance.nlargest(5)
    print(f"\n  {target}:")
    for feat, imp in top5.items():
        bar = "█" * int(imp * 100)
        print(f"    {feat:<35} {imp:.3f}  {bar}")

# ── SAVE MODELS ───────────────────────────────────────────────────────────────
print("\n Saving models...")
joblib.dump(trained_models["solar_output"], "/home/claude/model_solar.joblib")
joblib.dump(trained_models["wind_output"],  "/home/claude/model_wind.joblib")
joblib.dump(trained_models["load_demand"],  "/home/claude/model_load.joblib")
joblib.dump(scaler,                         "/home/claude/scaler.joblib")

# Also save training metadata for the app to display
metadata = {
    "feature_cols": FEATURE_COLS,
    "targets":      TARGETS,
    "train_end":    str(df_train.index[-1]),
    "test_start":   str(df_test.index[0]),
    "n_train":      len(df_train),
    "n_test":       len(df_test),
}
import json
with open("/home/claude/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved: model_solar.joblib")
print("✅ Saved: model_wind.joblib")
print("✅ Saved: model_load.joblib")
print("✅ Saved: scaler.joblib")
print("✅ Saved: model_metadata.json")
print("\n" + "=" * 60)
print("  Step 2 COMPLETE! Models trained and saved.")
print("=" * 60)
