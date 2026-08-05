import pandas as pd
from pathlib import Path
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="IA-FINANCIERA Dashboard", layout="wide")

CSV = Path("models/prob_summary.csv")
if not CSV.exists():
    st.error("No existe models/prob_summary.csv. Corre primero el pipeline.")
    st.stop()

df = pd.read_csv(CSV)
df = df.drop_duplicates(subset=["ticker"], keep="last").copy()
df["confianza"] = (df["proba_ens"] - 0.5).abs()
df = df.sort_values("confianza", ascending=False).reset_index(drop=True)

st.title("📊 IA-FINANCIERA — Resumen de Probabilidades")

col1, col2 = st.columns([2,3])

with col1:
    st.subheader("Top‑N Alcistas (Ensemble)")
    topn = st.slider("Top‑N", 5, 30, 10, step=1)
    top_up = df.sort_values("proba_ens", ascending=False).head(topn)
    st.plotly_chart(px.bar(top_up, x="ticker", y="proba_ens", text="proba_ens",
                           title=f"Top‑{topn} Alcistas (ENS)"), use_container_width=True)

    st.subheader("Top‑N Bajistas (Ensemble)")
    top_down = df.sort_values("proba_ens", ascending=True).head(topn)
    st.plotly_chart(px.bar(top_down, x="ticker", y="proba_ens", text="proba_ens",
                           title=f"Top‑{topn} Bajistas (ENS)"), use_container_width=True)

with col2:
    st.subheader("Mapa de calor (LogReg / RF / Ensemble)")
    show_cols = ["proba_logreg","proba_rf","proba_ens"]
    df_heat = df[["ticker"]+show_cols].copy().set_index("ticker")
    st.plotly_chart(px.imshow(df_heat.T, aspect="auto", color_continuous_scale="Viridis",
                              labels=dict(x="Ticker", y="Modelo", color="Proba subir")),
                    use_container_width=True)

st.subheader("Tabla detallada")
st.dataframe(df[["ticker","proba_logreg","proba_rf","proba_ens","pred","confianza"]],
             use_container_width=True)
st.caption("Fuente: models/prob_summary.csv")
