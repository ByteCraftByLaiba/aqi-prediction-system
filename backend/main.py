# app/main.py
import os
import json
import math
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils import (
    load_raw,                 # merged hourly df with 'time'
    build_latest_feature_row, # returns 1-row DataFrame for FEATURES (no NaNs)
    epa_cat_pm25, epa_cat_pm10, worst_category
)

# ---------- paths ----------
BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

TARGETS: set[str] = {"pm2_5_t+3h","pm2_5_t+6h","pm10_t+3h","pm10_t+6h"}

app = FastAPI(title="Lahore AQI Forecasting API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Globals ----------------
MODELS: Dict[str, Any] = {}          # tgt -> loaded estimator or bundle
FEATURES: List[str] = []             # ordered feature list used by training
RAW_DF: Optional[pd.DataFrame] = None

# ---------------- JSON safety helpers ----------------
def _to_json_safe_scalar(x):
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if x is None:
        return None
    if isinstance(x, (str, bool, np.bool_)):
        return x
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_json_safe(v) for v in obj)
    if isinstance(obj, (pd.Series,)):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame,)):
        df = obj.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return to_json_safe(df.to_dict("records"))
    return _to_json_safe_scalar(obj)

# ---------------- helpers ----------------
def safe_name(target: str) -> str:
    return target.replace("+", "plus").replace(" ", "_")

def ensure_target(t: str) -> str:
    if t not in TARGETS:
        raise HTTPException(status_code=400, detail=f"Invalid target: {t}")
    return t

def _features_paths() -> list[Path]:
    # try models/features_used.json then project-root/features_used.json
    return [MODELS_DIR / "features_used.json", BASE_DIR / "features_used.json"]

def features_path() -> Path:
    for p in _features_paths():
        if p.exists():
            return p
    # last resort: raise; caller will handle and attempt fallback
    raise FileNotFoundError("features_used.json not found in models/ or project root")

def model_path_for_target(tgt: str) -> Path:
    return MODELS_DIR / f"{safe_name(tgt)}_best.joblib"

def _try_load_json(p: Path) -> Optional[dict]:
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None

# ---------------- Loaders ----------------
def load_features_list() -> List[str]:
    # Prefer explicit file(s)
    for p in _features_paths():
        d = _try_load_json(p)
        if d and "features" in d and isinstance(d["features"], list) and len(d["features"]) > 0:
            return d["features"]

    # Fallback: infer from any per-target dataset (intersection of features across targets)
    ds_dir = BASE_DIR / "datasets_per_target"
    if ds_dir.exists():
        feats_sets = []
        for tgt in TARGETS:
            fp = ds_dir / f"{tgt}.csv"
            if fp.exists():
                df = pd.read_csv(fp, nrows=1)
                cols = [c for c in df.columns if c != tgt and c != "time"]
                feats_sets.append(set(cols))
        if feats_sets:
            feats = sorted(list(set.intersection(*feats_sets))) if len(feats_sets) > 1 else sorted(list(feats_sets[0]))
            if feats:
                return feats

    raise FileNotFoundError("Unable to determine feature list. Provide models/features_used.json or datasets_per_target/*.csv")

def load_models_from_disk() -> Tuple[Dict[str, Any], List[str]]:
    loaded: Dict[str, Any] = {}
    missing: List[str] = []
    for tgt in TARGETS:
        mp = model_path_for_target(tgt)
        if mp.exists():
            try:
                loaded[tgt] = joblib.load(mp)
            except Exception as e:
                print(f"[API] Failed to load {mp}: {e}")
                missing.append(str(mp))
        else:
            missing.append(str(mp))
    return loaded, missing

def init_or_reload():
    global MODELS, FEATURES, RAW_DF
    # features
    try:
        FEATURES[:] = load_features_list()
    except Exception as e:
        print(f"[API] Warning: load_features_list failed: {e}")
        FEATURES.clear()
    # models
    MODELS, missing = load_models_from_disk()
    # raw
    try:
        RAW_DF = load_raw(DATA_DIR)
    except TypeError:
        RAW_DF = load_raw()
    except Exception as e:
        print(f"[API] Warning: load_raw() failed: {e}")
        RAW_DF = None
    return missing

# ---------------- Startup ----------------
@app.on_event("startup")
def _startup():
    missing = init_or_reload()
    if missing:
        print("[API] Warning: missing model files:\n  " + "\n  ".join(missing))

@app.get("/")
def root():
    return {"ok": True, "message": "AQI API ready"}

# ---------------- Admin ----------------
@app.get("/reload")
def reload_models():
    missing = init_or_reload()
    return JSONResponse(content=to_json_safe({"reloaded": True, "missing_models": missing, "features_len": len(FEATURES)}))

@app.get("/health")
def health():
    missing_targets = [t for t in TARGETS if t not in MODELS]
    return JSONResponse(content=to_json_safe({
        "status": "ok" if not missing_targets else "degraded",
        "loaded_models": list(MODELS.keys()),
        "missing_models": missing_targets,
        "models_dir": str(MODELS_DIR),
        "features_file": str(features_path()) if FEATURES else "N/A",
        "features_count": len(FEATURES),
        "raw_loaded": bool(RAW_DF is not None and not RAW_DF.empty),
    }))

# ---------------- Metadata ----------------
@app.get("/features")
def features():
    return JSONResponse(content=to_json_safe({"features": FEATURES}))

@app.get("/models")
def models_info():
    mp = MODELS_DIR / "metrics.json"
    if mp.exists():
        return JSONResponse(content=to_json_safe(_try_load_json(mp) or {}))
    return JSONResponse(content={})

# ---------------- Timeseries ----------------
@app.get("/timeseries")
def timeseries(hours: int = 168):
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        try:
            RAW_DF = load_raw(DATA_DIR)
        except TypeError:
            RAW_DF = load_raw()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Raw data not available: {e}")

    df = RAW_DF.copy()
    if "time" not in df.columns:
        raise HTTPException(status_code=500, detail="Raw data missing 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").tail(hours)

    # sanitize
    df = df.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in df.columns if c != "time"]
    if cols:
        df[cols] = df[cols].fillna(method="ffill").fillna(method="bfill")
        df[cols] = df[cols].fillna(df[cols].mean()).fillna(0.0)

    payload = {
        "time": df["time"].astype(str).tolist(),
        "pm2_5": df.get("pm2_5", pd.Series([None]*len(df))).tolist(),
        "pm10": df.get("pm10", pd.Series([None]*len(df))).tolist(),
        "ozone": df.get("ozone", pd.Series([None]*len(df))).tolist(),
        "nitrogen_dioxide": df.get("nitrogen_dioxide", pd.Series([None]*len(df))).tolist(),
        "carbon_monoxide": df.get("carbon_monoxide", pd.Series([None]*len(df))).tolist(),
        "sulphur_dioxide": df.get("sulphur_dioxide", pd.Series([None]*len(df))).tolist(),
    }
    return JSONResponse(content=to_json_safe(payload))

# ---------------- SHAP explanations ----------------
@app.get("/explanations")
def explanations(target: str = Query(..., description="e.g. pm2_5_t+3h"), top_k: int = 20):
    t = ensure_target(target)
    p1 = MODELS_DIR / "shap" / f"{t}_shap.csv"
    p2 = MODELS_DIR / "shap" / f"{safe_name(t)}_shap.csv"
    shap_csv = p1 if p1.exists() else (p2 if p2.exists() else None)
    if shap_csv is None:
        raise HTTPException(status_code=404, detail="SHAP summary not found. Re-train with SHAP enabled.")
    try:
        df = pd.read_csv(shap_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read SHAP file: {e}")
    df = df.head(max(1, top_k))
    return JSONResponse(content=to_json_safe({
        "target": t,
        "top_k": top_k,
        "features": df.to_dict("records")
    }))

# ---------------- Prediction ----------------
@app.get("/predict")
def predict(horizon: int = Query(3, enum=[3, 6])):
    # map horizon -> targets
    targets = ["pm2_5_t+3h","pm10_t+3h"] if horizon == 3 else ["pm2_5_t+6h","pm10_t+6h"]

    # Build the latest feature row (must match training FEATURES)
    if not FEATURES:
        raise HTTPException(status_code=503, detail="Feature list not loaded.")
    try:
        row = build_latest_feature_row(FEATURES)  # 1-row DataFrame, NaN-free
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build feature row: {e}")

    out = {}
    for t in targets:
        # 1) prefer preloaded model bundle; 2) fallback to disk
        bundle = MODELS.get(t)
        if bundle is None:
            mp = model_path_for_target(t)
            if not mp.exists():
                out[t] = {"error": "model not available"}
                continue
            try:
                bundle = joblib.load(mp)
            except Exception as e:
                out[t] = {"error": f"failed to load model: {e}"}
                continue

        try:
            # ---------- Case A: stacked ensemble bundle we built for pm10_t+6h ----------
            if isinstance(bundle, dict) and bundle.get("type") == "stacked":
                feats = bundle.get("features", FEATURES)
                X = row[feats] if feats else row
                p1 = bundle["lgbm"].predict(X)
                p2 = bundle["xgb"].predict(X)
                w0, w1 = bundle["weights"]
                pred_val = float(w0 * float(np.asarray(p1)[0]) + w1 * float(np.asarray(p2)[0]))

            # ---------- Case B: plain estimator saved as {"model": est, "features": [...]} ----------
            elif isinstance(bundle, dict) and "model" in bundle:
                est = bundle["model"]
                feats = bundle.get("features")
                X = row[feats] if feats else row
                pred_val = float(np.asarray(est.predict(X))[0])

            # ---------- Case C: AutoMLPipeline rich bundle saved with {"pipeline": pipe, ...} ----------
            elif isinstance(bundle, dict) and "pipeline" in bundle:
                pipe = bundle["pipeline"]
                # drop target col if present; pipeline handles its own preprocessing/selection
                X = row.drop(columns=[t], errors="ignore")
                pred_val = float(np.asarray(pipe.predict(X))[0])

            # ---------- Case D: already a fitted estimator/pipeline ----------
            else:
                # Best effort: treat as sklearn estimator/pipeline
                X = row.drop(columns=[t], errors="ignore")
                pred_val = float(np.asarray(bundle.predict(X))[0])

        except Exception as e:
            out[t] = {"error": f"inference failed: {e}"}
            continue

        # EPA category helper
        cat = epa_cat_pm25(pred_val) if t.startswith("pm2_5") else epa_cat_pm10(pred_val)
        out[t] = {"prediction": pred_val, "category": cat}

    return JSONResponse(content=to_json_safe(out))
