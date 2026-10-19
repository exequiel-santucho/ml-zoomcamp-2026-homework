#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 6: Árboles de decisión y Ensemble Learning (Homework 6).

Ejecutar:
    uv run python hw6_questions.py
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)
DATA_PATH = Path(__file__).parent / "data" / "car_fuel_efficiency_2026.csv"
TARGET = "fuel_efficiency_mpg"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
df = load_data().fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_train = df_train.pop(TARGET).values
y_val = df_val.pop(TARGET).values
df_test.pop(TARGET)

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient="records"))
X_val = dv.transform(df_val.to_dict(orient="records"))
feature_names = list(dv.get_feature_names_out())

# ---------------------------------------------------------------------------
header("Q1. Árbol con max_depth=1: ¿qué feature usa para el split?")
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
print(export_text(dt, feature_names=feature_names))
split_feature = feature_names[dt.tree_.feature[0]]
print("Feature del split:", split_feature, "-> columna original:", split_feature.split("=")[0])

# ---------------------------------------------------------------------------
header("Q2. Random Forest (n_estimators=10)")
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
print("RMSE val:", round(root_mean_squared_error(y_val, rf.predict(X_val)), 4))

# ---------------------------------------------------------------------------
header("Q3. Variando n_estimators")
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"n_estimators={n:<4} RMSE={round(root_mean_squared_error(y_val, rf.predict(X_val)), 3)}")

# ---------------------------------------------------------------------------
header("Q4. Mejor max_depth (RMSE medio sobre n_estimators)")
for d in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        scores.append(root_mean_squared_error(y_val, rf.predict(X_val)))
    print(f"max_depth={d:<3} mean RMSE={np.mean(scores):.4f}")

# ---------------------------------------------------------------------------
header("Q5. Feature importance")
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(imp.round(4))

# ---------------------------------------------------------------------------
header("Q6. XGBoost: eta=0.3 vs eta=0.1")
features_xgb = [f.replace("=", "_").replace(" ", "_") for f in feature_names]
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features_xgb)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features_xgb)
watchlist = [(dtrain, "train"), (dval, "val")]

for eta in [0.3, 0.1]:
    xgb_params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                      evals=watchlist, verbose_eval=False)
    score = root_mean_squared_error(y_val, model.predict(dval))
    print(f"eta={eta} RMSE val={score:.4f}")
