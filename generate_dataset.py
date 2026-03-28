"""
Step 1: Physics-Based Dataset Generator
========================================
This script creates realistic synthetic data for the hybrid microgrid.

WHY THIS IS DIFFERENT FROM YOUR OLD DATA:
- Old data: random numbers with no relationship to anything
- New data: follows real physics rules
    - Solar power depends on the sun angle (time of day + season)
    - Wind power follows a real power curve (cubic law)
    - Load demand peaks in morning and evening like real households
    - All values have realistic noise added (weather isn't perfectly predictable)
"""

import numpy as np
import pandas as pd
import os

# Set a seed so we get the same data every time we run this
np.random.seed(42)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# One full year of hourly data = 8760 rows
HOURS_PER_YEAR = 8760
START_DATE      = "2023-01-01"

# System capacity constants (realistic for a small community microgrid)
SOLAR_PANEL_CAPACITY_KW   = 500   # Maximum possible solar output
WIND_TURBINE_CAPACITY_KW  = 300   # Maximum possible wind output
BATTERY_CAPACITY_KWH      = 1000  # How much energy the battery can store
BATTERY_MAX_CHARGE_RATE   = 200   # Max kW to charge per hour
BATTERY_MAX_DISCHARGE_RATE = 200  # Max kW to discharge per hour
BASE_LOAD_KW              = 250   # Average electricity demand

print("=" * 60)
print("  Generating Physics-Based Microgrid Dataset")
print("=" * 60)

# ── TIME AXIS ──────────────────────────────────────────────────────────────────
timestamps = pd.date_range(start=START_DATE, periods=HOURS_PER_YEAR, freq="h")
df = pd.DataFrame(index=timestamps)
df.index.name = "timestamp"

hour_of_day = df.index.hour          # 0 to 23
day_of_year = df.index.dayofyear     # 1 to 365
month       = df.index.month         # 1 to 12
day_of_week = df.index.dayofweek     # 0=Mon, 6=Sun

# ── WEATHER: SOLAR IRRADIANCE (W/m²) ──────────────────────────────────────────
# Real physics:
# - Sun is only up between roughly hour 6 and hour 20
# - Peak irradiance happens at solar noon (~hour 12-13)
# - Summer (day 172) has much more sun than winter (day 355)
# - We add random cloud cover noise

# Solar elevation angle (simplified)
# Seasonal factor: peaks in summer, lowest in winter
seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

# Daily bell curve: peaks at noon, zero at night
daily_factor = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 14))

# Base clear-sky irradiance
clear_sky_irradiance = 900 * seasonal_factor * daily_factor

# Add cloud cover variability (weather noise) — some days are cloudy
cloud_cover = np.random.beta(2, 3, HOURS_PER_YEAR)  # Skewed towards clear sky
solar_irradiance = clear_sky_irradiance * (1 - 0.6 * cloud_cover)
solar_irradiance = np.maximum(0, solar_irradiance)

df["solar_irradiance"] = solar_irradiance.values
print("✅ Solar irradiance generated")

# ── WEATHER: WIND SPEED (m/s) ──────────────────────────────────────────────────
# Real physics:
# - Wind tends to be stronger at night and in winter
# - Weibull distribution is used to model wind speed (industry standard)
# - Scale factor varies by season
k_shape    = 2.0    # Weibull shape (typical for temperate regions)
c_scale_base = 8.0  # Average wind speed m/s

# Seasonal variation: windier in winter
seasonal_wind = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 355) / 365)

# Daily variation: calmer midday, windier at night
daily_wind = 1.0 - 0.2 * np.sin(np.pi * (hour_of_day - 3) / 12)

c_scale = c_scale_base * seasonal_wind * daily_wind

# Generate Weibull-distributed wind speeds
wind_speed = c_scale * np.random.weibull(k_shape, HOURS_PER_YEAR)
wind_speed = np.clip(wind_speed, 0, 30)  # Cap at 30 m/s (storm cutoff)

df["wind_speed"] = wind_speed
print("✅ Wind speed generated")

# ── WEATHER: TEMPERATURE (°C) ──────────────────────────────────────────────────
# Seasonal: peaks in July (~day 196), lowest in January (~day 14)
# Daily: warmest at 2-3pm, coldest before sunrise
seasonal_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
daily_temp    = 5  *  np.sin(np.pi * (hour_of_day - 6) / 12)
noise_temp    = np.random.normal(0, 2, HOURS_PER_YEAR)
df["temperature"] = seasonal_temp + daily_temp + noise_temp

# ── WEATHER: HUMIDITY (%) ─────────────────────────────────────────────────────
# Inversely related to temperature (hot → dry, cool → humid)
df["humidity"] = 60 - 0.8 * df["temperature"] + np.random.normal(0, 8, HOURS_PER_YEAR)
df["humidity"] = df["humidity"].clip(10, 100)

# ── WEATHER: PRESSURE (hPa) ───────────────────────────────────────────────────
df["pressure"] = 1013 + np.random.normal(0, 8, HOURS_PER_YEAR)
df["pressure"] = df["pressure"].clip(980, 1045)

print("✅ Temperature, humidity, pressure generated")

# ── SOLAR OUTPUT (kW) ──────────────────────────────────────────────────────────
# Real physics: Solar output ∝ irradiance, slightly reduced by high temperature
# (Solar panels lose ~0.4% efficiency per °C above 25°C)
temp_coeff    = 0.004  # 0.4% per °C
efficiency    = 0.18   # 18% panel efficiency (modern standard)
temp_derating = 1 - temp_coeff * np.maximum(0, df["temperature"] - 25)
panel_area    = SOLAR_PANEL_CAPACITY_KW * 1000 / (efficiency * 900)  # m²

solar_output = (df["solar_irradiance"] * efficiency * temp_derating * panel_area) / 1000
solar_output = solar_output.clip(0, SOLAR_PANEL_CAPACITY_KW)
# Add small inverter noise
solar_output += np.random.normal(0, 2, HOURS_PER_YEAR)
solar_output = solar_output.clip(0, SOLAR_PANEL_CAPACITY_KW)
df["solar_output"] = solar_output
print("✅ Solar output generated (with temperature derating)")

# ── WIND OUTPUT (kW) ───────────────────────────────────────────────────────────
# Real physics: Wind power follows the CUBIC LAW (power ∝ wind³)
# But turbines have a cut-in speed (3 m/s) and a cut-out speed (25 m/s)
CUT_IN  = 3.0   # m/s — turbine starts spinning
RATED   = 12.0  # m/s — turbine reaches full power
CUT_OUT = 25.0  # m/s — turbine shuts down for safety

ws = df["wind_speed"].values
wind_output = np.zeros(HOURS_PER_YEAR)

# Between cut-in and rated: cubic power curve
mask_partial = (ws >= CUT_IN) & (ws < RATED)
wind_output[mask_partial] = WIND_TURBINE_CAPACITY_KW * ((ws[mask_partial] - CUT_IN) / (RATED - CUT_IN)) ** 3

# Between rated and cut-out: full power
mask_rated = (ws >= RATED) & (ws < CUT_OUT)
wind_output[mask_rated] = WIND_TURBINE_CAPACITY_KW

# Above cut-out: turbine is off (safety shutdown)
wind_output[ws >= CUT_OUT] = 0

# Add mechanical noise
wind_output += np.random.normal(0, 3, HOURS_PER_YEAR)
wind_output = np.clip(wind_output, 0, WIND_TURBINE_CAPACITY_KW)
df["wind_output"] = wind_output
print("✅ Wind output generated (real cubic power curve)")

# ── LOAD DEMAND (kW) ───────────────────────────────────────────────────────────
# Real patterns:
# - Morning peak: 7-9am (people wake up, cook, shower)
# - Evening peak: 6-9pm (cooking, TV, heating/cooling)
# - Low at night: 11pm-5am
# - Slightly lower on weekends
# - Higher demand in summer (AC) and winter (heating)

# Convert Index to numpy arrays for arithmetic
hour_arr    = df.index.hour.to_numpy()
dow_arr     = df.index.dayofweek.to_numpy()
doy_arr     = df.index.dayofyear.to_numpy()

# Daily profile: two-hump camel curve
morning_peak = np.exp(-0.5 * ((hour_arr - 8)  / 1.5) ** 2)
evening_peak = np.exp(-0.5 * ((hour_arr - 19) / 2.0) ** 2)
daily_load_profile = BASE_LOAD_KW * (0.4 + 0.8 * morning_peak + 1.0 * evening_peak)

# Seasonal: higher in summer and winter (AC and heating), lower in spring/autumn
seasonal_load = 1.0 + 0.25 * np.abs(np.sin(2 * np.pi * (doy_arr - 80) / 365))

# Weekend reduction (people go out, factories closed)
is_weekend = (dow_arr >= 5).astype(float)
weekend_factor = 1.0 - 0.15 * is_weekend

# Random demand fluctuations
demand_noise = np.random.normal(0, 20, HOURS_PER_YEAR)

load_demand = (daily_load_profile * seasonal_load * weekend_factor) + demand_noise
load_demand = load_demand.clip(50, 800)
df["load_demand"] = load_demand
print("✅ Load demand generated (morning + evening peaks)")

# ── GRID PARAMETERS ────────────────────────────────────────────────────────────
df["grid_frequency"] = 50 + np.random.normal(0, 0.1, HOURS_PER_YEAR)
df["grid_voltage"]   = 230 + np.random.normal(0, 5, HOURS_PER_YEAR)

# ── BATTERY & GRID LOGIC (Smart Controller Simulation) ─────────────────────────
print("\n Running smart controller simulation...")
battery_soc       = np.zeros(HOURS_PER_YEAR)  # State of Charge in kWh
battery_charge    = np.zeros(HOURS_PER_YEAR)
battery_discharge = np.zeros(HOURS_PER_YEAR)
grid_exchange     = np.zeros(HOURS_PER_YEAR)
total_renewable   = np.zeros(HOURS_PER_YEAR)

# Start battery at 50% charge
current_soc = BATTERY_CAPACITY_KWH * 0.5

SOC_MIN = BATTERY_CAPACITY_KWH * 0.10  # Never go below 10%
SOC_MAX = BATTERY_CAPACITY_KWH * 0.90  # Never charge above 90% (battery health)

for i in range(HOURS_PER_YEAR):
    gen  = df["solar_output"].iloc[i] + df["wind_output"].iloc[i]
    load = df["load_demand"].iloc[i]
    net  = gen - load  # Positive = surplus, Negative = deficit

    total_renewable[i] = gen
    charge    = 0.0
    discharge = 0.0
    grid_ex   = 0.0

    if net > 0:
        # SURPLUS: try to charge battery first, then sell to grid
        can_charge = min(net, BATTERY_MAX_CHARGE_RATE, SOC_MAX - current_soc)
        charge     = max(0, can_charge)
        current_soc += charge
        leftover   = net - charge
        grid_ex    = leftover  # Positive = selling to grid
    else:
        # DEFICIT: try to discharge battery first, then buy from grid
        needed     = abs(net)
        can_discharge = min(needed, BATTERY_MAX_DISCHARGE_RATE, current_soc - SOC_MIN)
        discharge  = max(0, can_discharge)
        current_soc -= discharge
        still_needed = needed - discharge
        grid_ex    = -still_needed  # Negative = buying from grid

    battery_soc[i]       = current_soc
    battery_charge[i]    = charge
    battery_discharge[i] = discharge
    grid_exchange[i]     = grid_ex

df["total_renewable"]   = total_renewable
df["grid_exchange"]     = grid_exchange
df["battery_soc"]       = battery_soc
df["battery_charge"]    = battery_charge
df["battery_discharge"] = battery_discharge
print("✅ Battery and grid simulation complete")

# ── TIME FEATURES ──────────────────────────────────────────────────────────────
df["hour"]        = hour_of_day
df["day_of_week"] = day_of_week
df["month"]       = month
df["day_of_year"] = day_of_year

# ── LAG & ROLLING FEATURES (Memory of recent history) ─────────────────────────
# These help the model know "what was happening 1 hour ago?"
df["load_demand_lag1"]         = df["load_demand"].shift(1)
df["solar_irradiance_lag1"]    = df["solar_irradiance"].shift(1)
df["wind_speed_lag1"]          = df["wind_speed"].shift(1)

# 3-hour rolling averages (smooth out spikes)
df["wind_speed_roll_3h"]          = df["wind_speed"].rolling(3).mean()
df["solar_irradiance_roll_3h"]    = df["solar_irradiance"].rolling(3).mean()
df["load_roll_3h"]                = df["load_demand"].rolling(3).mean()

# Drop the first few rows that have NaN due to lag/rolling
df = df.dropna()

print(f"\n Dataset shape: {df.shape}")
print(f" Date range:   {df.index[0]} → {df.index[-1]}")
print(f"\n📊 Key Statistics:")
print(f"  Solar output:   min={df['solar_output'].min():.1f}, max={df['solar_output'].max():.1f}, mean={df['solar_output'].mean():.1f} kW")
print(f"  Wind output:    min={df['wind_output'].min():.1f}, max={df['wind_output'].max():.1f}, mean={df['wind_output'].mean():.1f} kW")
print(f"  Load demand:    min={df['load_demand'].min():.1f}, max={df['load_demand'].max():.1f}, mean={df['load_demand'].mean():.1f} kW")
print(f"  Battery SOC:    min={df['battery_soc'].min():.1f}, max={df['battery_soc'].max():.1f} kWh")

# ── SAVE ───────────────────────────────────────────────────────────────────────
output_path = "/home/claude/processed_dataset.csv"
df.to_csv(output_path)
print(f"\n✅ Dataset saved to: {output_path}")
print("=" * 60)
print("  Step 1 COMPLETE! Ready for model training.")
print("=" * 60)
