
"""Classical statistical models for univariate forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_arima(y: np.ndarray, order: Tuple[int,int,int]=(1,1,1)):
    model = ARIMA(y, order=order)
    res = model.fit()
    return res

def fit_sarimax(y: np.ndarray, exog: Optional[np.ndarray]=None, order=(1,1,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res
