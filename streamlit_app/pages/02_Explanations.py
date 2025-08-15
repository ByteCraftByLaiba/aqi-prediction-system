# streamlit_app/pages/02_Explanations.py
import os, requests, pandas as pd, streamlit as st, altair as alt
API = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Explanations", page_icon="🧠", layout="wide")
st.title("🧠 Why the model thinks so (SHAP)")
st.caption("Top features by average absolute SHAP value")

targets = ["pm2_5_t+3h","pm2_5_t+6h","pm10_t+3h","pm10_t+6h"]

def fetch_shap(target: str, k=20) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{API}/explanations", params={"target": target, "top_k": k}, timeout=30)
        r.raise_for_status()
        data = r.json().get("features", [])
        return pd.DataFrame(data) if data else None
    except Exception as e:
        st.warning(f"Failed to fetch SHAP for {target}: {e}")
        return None

def shap_chart(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        st.info("No data to display.")
        return
    value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    if value_col is None:
        st.info("No SHAP values found in the response.")
        return
    chart = (
        alt.Chart(df)
        .transform_window(rank="rank()", sort=[alt.SortField(value_col, order="descending")])
        .transform_filter(alt.datum.rank <= 20)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title="Mean |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip(f"{value_col}:Q", format=".4f")],
        ).properties(height=420, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

# Controls
c1, c2 = st.columns([2,1])
with c1:
    target = st.selectbox("Choose a forecast to explain", options=targets, index=0,
                          format_func=lambda t: {"pm2_5_t+3h":"PM2.5 – next few hours",
                                                 "pm2_5_t+6h":"PM2.5 – later today",
                                                 "pm10_t+3h":"PM10 – next few hours",
                                                 "pm10_t+6h":"PM10 – later today"}[t])
with c2:
    top_k = st.slider("How many features?", 5, 40, 20, step=1)

df = fetch_shap(target, k=top_k)
shap_chart(df, f"{target} – top {top_k} features")

# Optional table for detail + download
if df is not None and not df.empty:
    st.markdown("### Table")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{target}_shap_top{top_k}.csv", mime="text/csv")
