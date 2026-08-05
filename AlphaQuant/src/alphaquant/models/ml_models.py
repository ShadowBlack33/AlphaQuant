
"""Top ML regressors commonly effective for tabular time-series features."""
from __future__ import annotations
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def get_model_zoo(random_state: int=42) -> Dict[str, Pipeline]:
    models = {
        "linreg": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=random_state))]),
        "lasso":  Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.001, random_state=random_state))]),
        "elastic":Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state))]),
        "svr":    Pipeline([("scaler", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.01))]),
        "rf":     Pipeline([("model", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1))]),
        "gbr":    Pipeline([("model", GradientBoostingRegressor(random_state=random_state))]),
    }
    if HAS_XGB:
        models["xgb"] = Pipeline([("model", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=random_state, tree_method="hist"))])
    return models
