from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SRC = Path("models/prob_summary.csv")
OUT = Path("images/heatmap_probs.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)
# Normalizamos formato y orden
df = df[["ticker","proba_logreg","proba_rf","proba_ens"]].copy()
df = df.sort_values("proba_ens", ascending=False).reset_index(drop=True)

mat = df[["proba_logreg","proba_rf","proba_ens"]].to_numpy()
tickers = df["ticker"].tolist()
cols = ["LogReg","RandomForest","Ensemble"]

plt.figure(figsize=(8, max(4, 0.35*len(tickers))))
plt.imshow(mat, aspect="auto", interpolation="nearest")
plt.yticks(range(len(tickers)), tickers)
plt.xticks(range(len(cols)), cols)
plt.colorbar(label="Probabilidad de subir")
plt.title("Mapa de calor de probabilidades (último run)")
plt.tight_layout()
plt.savefig(OUT, dpi=160)
print(f"✅ Heatmap guardado en: {OUT}")
