# streamlit_app/pages/01_Forecast.py
import os, requests, pandas as pd, numpy as np, streamlit as st, altair as alt

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Detailed Forecast")
st.caption("Clear outlooks, AQI bands, and top feature drivers (SHAP)")

# ---------- theme-aware styles ----------
st.markdown("""
<style>
:root { --card-bg:#fff; --card-border:rgba(0,0,0,.08); --text:#111; --muted:rgba(0,0,0,.6); }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#1e1e1e; --card-border:rgba(255,255,255,.12); --text:#eaeaea; --muted:rgba(255,255,255,.65); }
}
.kpi-card {border-radius:16px;padding:16px 18px;background:var(--card-bg);
           border:1px solid var(--card-border); box-shadow:0 8px 16px rgba(0,0,0,.05); color:var(--text)}
.kpi-title {font-weight:600;font-size:.95rem;opacity:.9}
.kpi-value {font-size:2rem;font-weight:700;margin-top:6px}
.badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;color:#fff}
.band-outer {width:100%;height:10px;border-radius:6px;background:#3a3a3a22;overflow:hidden;margin-top:10px}
.band-inner {height:100%;border-radius:6px}
.small {font-size:.85rem;color:var(--muted)}
</style>
""", unsafe_allow_html=True)

ORDER = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
COLORS = {"Good":"#00A65A","Moderate":"#FFCC00","Unhealthy for Sensitive Groups":"#FF7E00",
          "Unhealthy":"#FF0000","Very Unhealthy":"#8F3F97","Hazardous":"#7E0023"}
def cat_color(c): return COLORS.get(c, "#6c757d")
def worst_cat(c1,c2):
    if not c1: return c2
    if not c2: return c1
    return max([c1,c2], key=lambda c: ORDER.index(c))

PM25_BPS = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
PM10_BPS = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,604,301,500)]
def aqi_from_conc(c, bps):
    c=float(c)
    for Cl,Ch,Il,Ih in bps:
        if Cl<=c<=Ch: return (Ih-Il)/(Ch-Cl)*(c-Cl)+Il
    return min(500.0, max(0.0, c))
def aqi25(c): return aqi_from_conc(c, PM25_BPS)
def aqi10(c): return aqi_from_conc(c, PM10_BPS)
def cat_for_aqi(a):
    return "Good" if a<=50 else ("Moderate" if a<=100 else ("Unhealthy for Sensitive Groups" if a<=150 else ("Unhealthy" if a<=200 else ("Very Unhealthy" if a<=300 else "Hazardous"))))
def band_html(aqi):
    width=int(np.clip(aqi/500*100,0,100)); color=cat_color(cat_for_aqi(aqi))
    return f"<div class='band-outer'><div class='band-inner' style='width:{width}%;background:{color}'></div></div><div class='small'>AQI {aqi:.0f} / 500</div>"

def kpi_card(title, value, unit, category, foot_html=""):
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-title'>{title}</div>
      <div class='kpi-value'>{value}{(' ' + unit) if unit else ''}</div>
      <div style="margin:8px 0 0 0"><span class='badge' style='background:{cat_color(category)}'>{category or '—'}</span></div>
      <div style="margin-top:10px">{foot_html}</div>
    </div>
    """, unsafe_allow_html=True)

def parse_pred_payload(payload, horizon):
    k25 = "pm2_5_t+3h" if horizon==3 else "pm2_5_t+6h"
    k10 = "pm10_t+3h"  if horizon==3 else "pm10_t+6h"
    if k25 not in payload or ("error" in payload.get(k25, {})): raise RuntimeError(payload.get(k25, {}).get("error", f"Missing {k25}"))
    if k10 not in payload or ("error" in payload.get(k10, {})): raise RuntimeError(payload.get(k10, {}).get("error", f"Missing {k10}"))
    pm25=float(payload[k25]["prediction"]); cat25=payload[k25].get("category","")
    pm10=float(payload[k10]["prediction"]); cat10=payload[k10].get("category","")
    return {"pm2_5":pm25,"pm2_5_category":cat25,"pm10":pm10,"pm10_category":cat10,"overall_category":worst_cat(cat25,cat10)}

def fetch_explanations(target, k=12):
    try:
        r = requests.get(f"{API}/explanations", params={"target":target,"top_k":k}, timeout=20); r.raise_for_status()
        feats = r.json().get("features", [])
        return pd.DataFrame(feats) if feats else None
    except Exception:
        return None

def shap_bar(df, title):
    if df is None or df.empty:
        st.info("No explanation data available."); return
    value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else (df.columns[1] if len(df.columns)>1 else None)
    if value_col is None: st.info("No SHAP values in the response."); return
    chart = (alt.Chart(df)
             .transform_window(rank="rank()", sort=[alt.SortField(value_col, order="descending")])
             .transform_filter(alt.datum.rank <= 12)
             .mark_bar()
             .encode(x=alt.X(f"{value_col}:Q", title="Mean |SHAP|"),
                     y=alt.Y("feature:N", sort="-x", title="Feature"),
                     tooltip=["feature", alt.Tooltip(f"{value_col}:Q", format=".4f")])
             .properties(height=280, title=title))
    st.altair_chart(chart, use_container_width=True)

def latest_timestamp():
    try:
        r = requests.get(f"{API}/timeseries", params={"hours": 168}, timeout=60); r.raise_for_status()
        ts = pd.DataFrame(r.json())
        if "time" in ts.columns:
            t = pd.to_datetime(ts["time"], errors="coerce").dropna()
            if not t.empty: return t.max()
    except Exception:
        pass
    return None

ref_time = latest_timestamp()
st.caption(f"Forecasts use data up to **{ref_time}**." if ref_time is not None else "Forecasts use the most recent data available.")

SECTIONS = [("✅ Next few hours (about 3 hours from now)", 3),
            ("🕒 Later today (around 6 hours from now)", 6)]
for label, h in SECTIONS:
    st.subheader(label)
    try:
        res = parse_pred_payload(requests.get(f"{API}/predict", params={"horizon": h}, timeout=60).json(), h)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}"); continue

    a25, a10 = aqi25(res["pm2_5"]), aqi10(res["pm10"])
    a_all = max(a25, a10)

    c1, c2, c3, c4 = st.columns([1,1,1,1.4])
    with c1: kpi_card("PM2.5 forecast", f"{res['pm2_5']:.1f}", "µg/m³", res["pm2_5_category"], band_html(a25))
    with c2: kpi_card("PM10 forecast",  f"{res['pm10']:.1f}",  "µg/m³", res["pm10_category"],  band_html(a10))
    with c3: kpi_card("Overall category", res["overall_category"], "", res["overall_category"], band_html(a_all))
    with c4:
        st.markdown("**Top drivers (SHAP)**")
        tgt25 = "pm2_5_t+3h" if h==3 else "pm2_5_t+6h"
        tgt10 = "pm10_t+3h"  if h==3 else "pm10_t+6h"
        t1, t2 = st.tabs(["PM2.5", "PM10"])
        with t1: shap_bar(fetch_explanations(tgt25, 12), f"{tgt25} feature impact")
        with t2: shap_bar(fetch_explanations(tgt10, 12), f"{tgt10} feature impact")
