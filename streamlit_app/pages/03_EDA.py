# streamlit_app/pages/03_EDA.py
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

API = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="EDA", page_icon="🔎", layout="wide")
st.title("🔎 Explore recent air quality")
st.caption("Distributions and correlations over the last 7 days (168 hours)")

try:
    r = requests.get(f"{API}/timeseries", params={"hours": 168}, timeout=60)
    r.raise_for_status()
    ts = pd.DataFrame(r.json())
except Exception as e:
    st.error(f"Failed to fetch time series: {e}")
    st.stop()

if "time" in ts.columns:
    ts["time"] = pd.to_datetime(ts["time"], errors="coerce")
    ts = ts.dropna(subset=["time"]).sort_values("time")

st.line_chart(ts.set_index("time")[["pm2_5","pm10"]].dropna(how="all"), height=220)

st.subheader("Distributions")
desired_cols = [
    "pm2_5","pm10","ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide",
    "temperature_2m","relative_humidity_2m","wind_speed_10m"
]
available = [c for c in desired_cols if c in ts.columns]
missing = [c for c in desired_cols if c not in ts.columns]
if missing:
    st.caption(f"ℹ️ Missing in server response: {', '.join(missing)}")

c1, c2, c3 = st.columns(3)
if not available:
    st.info("No numeric columns available to plot.")
else:
    for i, col in enumerate(available):
        with [c1, c2, c3][i % 3]:
            s = ts[[col]].dropna()
            if s.empty:
                st.write(f"({col}: no data)")
                continue
            st.altair_chart(
                alt.Chart(s).transform_density(
                    col, as_=[col, "density"]
                ).mark_area(opacity=0.5).encode(
                    x=alt.X(f"{col}:Q", title=col),
                    y=alt.Y("density:Q", title="Density"),
                    tooltip=[alt.Tooltip(f"{col}:Q", format=".2f")]
                ).properties(height=150, width=250),
                use_container_width=True
            )

st.subheader("Correlation (Pearson)")
num_cols = [c for c in available if np.issubdtype(ts[c].dropna().dtype, np.number)]
if len(num_cols) >= 2:
    corr = ts[num_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)
else:
    st.info("Not enough numeric columns for correlation.")
