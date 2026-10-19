#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 2: Machine Learning para Regresión (Homework 2).

Regresión lineal implementada desde cero con NumPy (ecuación normal),
igual que en las lecciones del módulo.

Ejecutar:
    uv run python hw2_questions.py
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)
DATA_PATH = Path(__file__).parent / "data" / "car_fuel_efficiency_2026.csv"

FEATURES = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]
TARGET = "fuel_efficiency_mpg"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Funciones del curso
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, seed: int):
    """Split 60/20/20 exactamente como en la lección."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return df_train, df_val, df_test


def train_linear_regression(X: np.ndarray, y: np.ndarray, r: float = 0.0):
    """w = (XᵀX + r·I)⁻¹ Xᵀy  — con r=0 es la ecuación normal sin regularizar."""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])  # columna de 1s para el sesgo (w0)
    XTX = X.T @ X
    XTX = XTX + r * np.eye(XTX.shape[0])
    w_full = np.linalg.inv(XTX) @ X.T @ y
    return w_full[0], w_full[1:]


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y - y_pred) ** 2).mean()))


def prepare_X(df: pd.DataFrame, fill_value: float) -> np.ndarray:
    return df[FEATURES].fillna(fill_value).values


# ---------------------------------------------------------------------------
df = load_data()[FEATURES + [TARGET]]

header("EDA. ¿fuel_efficiency_mpg tiene cola larga?")
print(df[TARGET].describe())
print("Skewness:", round(df[TARGET].skew(), 3), "(cerca de 0 → sin cola larga)")

header("Q1. Columna con valores faltantes")
print(df.isnull().sum())

header("Q2. Mediana de horsepower")
print("Mediana:", df["horsepower"].median())

# ---------------------------------------------------------------------------
header("Q3. Imputar con 0 vs con la media (del train)")
df_train, df_val, df_test = split(df, seed=42)
y_train, y_val = df_train[TARGET].values, df_val[TARGET].values

mean_hp = df_train["horsepower"].mean()  # ¡solo del train! (evita data leakage)
for name, fill in [("With 0", 0.0), ("With mean", mean_hp)]:
    w0, w = train_linear_regression(prepare_X(df_train, fill), y_train)
    y_pred = w0 + prepare_X(df_val, fill) @ w
    print(f"{name:10s} -> RMSE val = {round(rmse(y_val, y_pred), 3)}")

# ---------------------------------------------------------------------------
header("Q4. Regularización (NA=0)")
X_train, X_val = prepare_X(df_train, 0), prepare_X(df_val, 0)
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression(X_train, y_train, r=r)
    score = rmse(y_val, w0 + X_val @ w)
    print(f"r={r:<6} RMSE={round(score, 4)}")

# ---------------------------------------------------------------------------
header("Q5. Estabilidad frente a la semilla")
scores = []
for seed in range(10):
    tr, va, _ = split(df, seed)
    w0, w = train_linear_regression(prepare_X(tr, 0), tr[TARGET].values)
    s = rmse(va[TARGET].values, w0 + prepare_X(va, 0) @ w)
    scores.append(s)
    print(f"seed={seed} RMSE={s:.4f}")
print("std:", round(float(np.std(scores)), 3))

# ---------------------------------------------------------------------------
header("Q6. Train+val, r=0.001, seed=9 -> RMSE test")
tr, va, te = split(df, seed=9)
full = pd.concat([tr, va]).reset_index(drop=True)
w0, w = train_linear_regression(prepare_X(full, 0), full[TARGET].values, r=0.001)
print("RMSE test:", round(rmse(te[TARGET].values, w0 + prepare_X(te, 0) @ w), 3))
