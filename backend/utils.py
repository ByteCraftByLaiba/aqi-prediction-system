# utils.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --------- Paths (resolve relative to repo root when not provided) ----------
def _default_data_dir() -> Path:
    here = Path(__file__).resolve().parents[1]  # project root (…/aqi-prediction-system)
    return here / "data"

# --------- EPA helpers you already had (kept for imports in main.py) --------
def epa_cat_pm25(x: float) -> str:
    if x <= 12: return "Good"
    elif x <= 35.4: return "Moderate"
    elif x <= 55.4: return "Unhealthy for Sensitive Groups"
    elif x <= 150.4: return "Unhealthy"
    elif x <= 250.4: return "Very Unhealthy"
    else: return "Hazardous"

def epa_cat_pm10(x: float) -> str:
    if x <= 54: return "Good"
    elif x <= 154: return "Moderate"
    elif x <= 254: return "Unhealthy for Sensitive Groups"
    elif x <= 354: return "Unhealthy"
    elif x <= 424: return "Very Unhealthy"
    else: return "Hazardous"

def worst_category(*cats: str) -> str:
    order = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
    return max(cats, key=lambda c: order.index(c))

# --------- Core loaders -----------------------------------------------------
def load_raw(data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Load the latest merged hourly raw file produced by your data collection.
    Prefers data/raw_lahore_hourly.csv; else tries the newest CSV in data/.
    Returns a DataFrame with a parsed 'time' column.
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    raw_path = data_dir / "raw_lahore_hourly.csv"
    
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        # fallback: pick most recent CSV in data/
        csvs = list(data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        raw_path = max(csvs, key=os.path.getmtime)
        df = pd.read_csv(raw_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

# --------- Feature engineering mirrors training (minimal) -------------------
def _add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["hour"] = d["time"].dt.hour
    d["weekday"] = d["time"].dt.dayofweek
    d["month"] = d["time"].dt.month
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)

    d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    d["wday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["wday_cos"] = np.cos(2*np.pi*d["weekday"]/7)
    d["month_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["month"]/12)
    return d

def _add_rolls_and_lags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    # basic rolls used in training
    d["pm2_5_avg_3h"]       = d["pm2_5"].rolling(3, min_periods=1).mean()
    d["temperature_avg_3h"] = d["temperature_2m"].rolling(3, min_periods=1).mean()

    # extended stats used in dataset_per_target_export.py
    for col in ["pm2_5","pm10"]:
        for w in [6, 12]:
            d[f"{col}_mean_{w}h"] = d[col].rolling(w, min_periods=1).mean()
            d[f"{col}_std_{w}h"]  = d[col].rolling(w, min_periods=1).std()
            d[f"{col}_min_{w}h"]  = d[col].rolling(w, min_periods=1).min()
            d[f"{col}_max_{w}h"]  = d[col].rolling(w, min_periods=1).max()

    d["pm2_5_change_rate"] = d["pm2_5"].diff()
    d["pm10_change_rate"]  = d["pm10"].diff()
    d["ozone_change_rate"] = d["ozone"].diff()

    d["pm_ratio"]       = d["pm2_5"] / (d["pm10"] + 1e-6)
    d["temp_pm2_5"]     = d["temperature_2m"] * d["pm2_5"]
    d["humidity_pm2_5"] = d["relative_humidity_2m"] * d["pm2_5"]

    for col in ["pm2_5","pm10","temperature_2m","relative_humidity_2m","ozone","wind_speed_10m"]:
        for lag in [1, 3, 24, 168]:
            d[f"{col}_lag_{lag}h"] = d[col].shift(lag)

    return d

def _rebuild_features_from_raw(data_dir: Path | None, needed_features: list[str]) -> pd.DataFrame:
    """
    Build feature table from raw data, then return the **last row** with at least the
    requested columns present (fills remaining with zeros if still missing).
    """
    df_raw = load_raw(data_dir)
    d = _add_time_features(df_raw)
    d = _add_rolls_and_lags(d)

    # last row should have all lags/rolls (if you have >=168 hours of data)
    last = d.sort_values("time").tail(1).copy()

    # sanitize NaNs for inference
    last = last.replace([np.inf, -np.inf], np.nan)
    # keep 'time' if present, but we only return needed feature columns
    # fill missing required features
    for col in needed_features:
        if col not in last.columns:
            last[col] = 0.0
    last = last[needed_features].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return last

# --------- Public: build latest feature row -------------------------------
def build_latest_feature_row(features: list[str]) -> pd.DataFrame:
    """
    Return a **single-row DataFrame** with exactly the `features` columns, NaN-free,
    to be fed into your trained models.
    Strategy:
      1) Prefer the precomputed table `data/lahore_features_no_targets.csv` (fast).
      2) If missing, rebuild features from raw and return the last row.
    """
    data_dir = _default_data_dir()
    precomp = data_dir / "lahore_features_no_targets.csv"
    if precomp.exists():
        df = pd.read_csv(precomp)
        # Best effort: if 'time' exists, use last chronologically
        if "time" in df.columns:
            try:
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time")
            except Exception:
                pass
        last = df.tail(1).copy()
        # ensure all needed columns exist
        for c in features:
            if c not in last.columns:
                last[c] = 0.0
        # at the end of the function, right before return
        last = last.reindex(columns=features, fill_value=0.0)
        last = last.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return last

    # Fallback: rebuild from raw
    return _rebuild_features_from_raw(data_dir, features)
