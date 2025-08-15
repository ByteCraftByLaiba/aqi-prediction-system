# streamlit_app/Home.py
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

# ---------- theme-aware styles ----------
st.markdown("""
<style>
:root { --card-bg:#ffffff; --card-border:rgba(0,0,0,.08); --text:#111; --muted:rgba(0,0,0,.6); }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#1e1e1e; --card-border:rgba(255,255,255,.12); --text:#eaeaea; --muted:rgba(255,255,255,.65); }
}
.card {border-radius:14px;padding:16px 18px;background:var(--card-bg);
       border:1px solid var(--card-border); box-shadow:0 8px 18px rgba(0,0,0,.05); color:var(--text);}
.k {font-weight:600;font-size:.95rem;opacity:.9}
.v {font-size:2rem;font-weight:700;margin-top:6px}
.b {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;color:#fff}
.band-outer {width:100%;height:10px;border-radius:6px;background:#3a3a3a22;overflow:hidden;margin-top:10px}
.band-inner {height:100%;border-radius:6px}
.small {font-size:.85rem;color:var(--muted)}
</style>
""", unsafe_allow_html=True)

st.title("🌍 Lahore AQI – What to expect next")
st.caption("Daily-updated forecasts powered by ML with SHAP explanations")

# ---------------- AQI helpers ----------------
def _aqi_linear(c, c_lo, c_hi, aqi_lo, aqi_hi):
    return (aqi_hi - aqi_lo) / (c_hi - c_lo) * (c - c_lo) + aqi_lo

def aqi_from_pm25(pm25):
    bps = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),
           (150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for c_lo, c_hi, a_lo, a_hi in bps:
        if pm25 <= c_hi: return _aqi_linear(pm25, c_lo, c_hi, a_lo, a_hi)
    return 500.0

def aqi_from_pm10(pm10):
    bps = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),
           (355,424,201,300),(425,504,301,400),(505,604,401,500)]
    for c_lo, c_hi, a_lo, a_hi in bps:
        if pm10 <= c_hi: return _aqi_linear(pm10, c_lo, c_hi, a_lo, a_hi)
    return 500.0

ORDER = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
COLORS = {"Good":"#00A65A","Moderate":"#FFCC00","Unhealthy for Sensitive Groups":"#FF7E00",
          "Unhealthy":"#FF0000","Very Unhealthy":"#8F3F97","Hazardous":"#7E0023"}

def worst_category(c1, c2):
    if not c1: return c2
    if not c2: return c1
    return max([c1,c2], key=lambda c: ORDER.index(c))

def aqi_band(aqi):
    width = int(np.clip(aqi/500*100, 0, 100))
    if   aqi <= 50:  cat="Good"
    elif aqi <= 100: cat="Moderate"
    elif aqi <= 150: cat="Unhealthy for Sensitive Groups"
    elif aqi <= 200: cat="Unhealthy"
    elif aqi <= 300: cat="Very Unhealthy"
    else:            cat="Hazardous"
    color = COLORS.get(cat, "#6c757d")
    return f"<div class='band-outer'><div class='band-inner' style='width:{width}%;background:{color}'></div></div><div class='small'>AQI {aqi:.0f} / 500</div>"

def card(title, value, unit, category, aqi_value=None):
    badge = COLORS.get(category or "", "#6c757d")
    band = aqi_band(aqi_value) if aqi_value is not None else ""
    st.markdown(f"""
    <div class="card">
      <div class="k">{title}</div>
      <div class="v">{value}{' ' + unit if unit else ''}</div>
      <div style="margin:8px 0 0 0"><span class="b" style="background:{badge}">{category or '—'}</span></div>
      <div style="margin-top:10px">{band}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- API helpers ----------------
def fetch_pred(h):
    r = requests.get(f"{API}/predict", params={"horizon": h}, timeout=60); r.raise_for_status(); return r.json()

def parse_pred_payload(payload, horizon):
    k25 = "pm2_5_t+3h" if horizon==3 else "pm2_5_t+6h"
    k10 = "pm10_t+3h"  if horizon==3 else "pm10_t+6h"
    if k25 not in payload or ("error" in payload.get(k25, {})): raise RuntimeError(payload.get(k25, {}).get("error", f"Missing {k25}"))
    if k10 not in payload or ("error" in payload.get(k10, {})): raise RuntimeError(payload.get(k10, {}).get("error", f"Missing {k10}"))
    pm25=float(payload[k25]["prediction"]); cat25=payload[k25].get("category","")
    pm10=float(payload[k10]["prediction"]); cat10=payload[k10].get("category","")
    aqi25=float(aqi_from_pm25(pm25)); aqi10=float(aqi_from_pm10(pm10))
    return {"pm2_5":pm25,"pm2_5_category":cat25,"pm2_5_aqi":aqi25,
            "pm10":pm10,"pm10_category":cat10,"pm10_aqi":aqi10,
            "overall_category":worst_category(cat25,cat10),"overall_aqi":max(aqi25,aqi10)}

def latest_timestamp():
    # pull a small window and read the max time
    try:
        r = requests.get(f"{API}/timeseries", params={"hours": 168}, timeout=60)
        r.raise_for_status()
        ts = pd.DataFrame(r.json())
        if "time" in ts.columns:
            t = pd.to_datetime(ts["time"], errors="coerce").dropna()
            if not t.empty: return t.max()
    except Exception:
        pass
    return None

# ---------------- UI ----------------
ref_time = latest_timestamp()
if ref_time is not None:
    st.caption(f"Forecasts are based on data up to **{ref_time}**.")
else:
    st.caption("Forecasts are based on the most recent data available.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Next few hours (about 3 hours from now)")
    try:
        p3 = parse_pred_payload(fetch_pred(3), 3)
        c1, c2, c3 = st.columns([1,1,1.1])
        with c1: card("PM2.5 forecast", f"{p3['pm2_5']:.1f}", "µg/m³", p3["pm2_5_category"], p3["pm2_5_aqi"])
        with c2: card("PM10 forecast",  f"{p3['pm10']:.1f}",  "µg/m³", p3["pm10_category"],  p3["pm10_aqi"])
        with c3: card("Overall air quality", p3["overall_category"], "", p3["overall_category"], p3["overall_aqi"])
    except Exception as e:
        st.error(f"Couldn’t fetch the “next few hours” forecast: {e}")

with col2:
    st.subheader("🕒 Later today (around 6 hours from now)")
    try:
        p6 = parse_pred_payload(fetch_pred(6), 6)
        c1, c2, c3 = st.columns([1,1,1.1])
        with c1: card("PM2.5 forecast", f"{p6['pm2_5']:.1f}", "µg/m³", p6["pm2_5_category"], p6["pm2_5_aqi"])
        with c2: card("PM10 forecast",  f"{p6['pm10']:.1f}",  "µg/m³", p6["pm10_category"],  p6["pm10_aqi"])
        with c3: card("Overall air quality", p6["overall_category"], "", p6["overall_category"], p6["overall_aqi"])
    except Exception as e:
        st.error(f"Couldn’t fetch the “later today” forecast: {e}")

st.markdown("---")
st.subheader("📈 Last 48 hours")
try:
    r = requests.get(f"{API}/timeseries", params={"hours": 48}, timeout=60); r.raise_for_status()
    ts = pd.DataFrame(r.json())
    if "time" in ts.columns:
        ts["time"] = pd.to_datetime(ts["time"], errors="coerce")
        ts = ts.dropna(subset=["time"]).sort_values("time")
    lc1, lc2 = st.columns(2)
    with lc1: st.line_chart(ts.set_index("time")[["pm2_5","pm10"]].dropna(how="all"))
    with lc2:
        cols = [c for c in ["ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide"] if c in ts.columns]
        if cols: st.line_chart(ts.set_index("time")[cols].dropna(how="all"))
        else: st.info("No pollutant breakdown available from the server.")
    if not ts.empty: st.caption(f"Window shown: **{ts['time'].min()} → {ts['time'].max()}**")
except Exception as e:
    st.error(f"Failed to load time series: {e}")
