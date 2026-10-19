#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 1: Introducción a Machine Learning (Homework 1).

Ejecutar:
    uv run python hw1_questions.py
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)
DATA_PATH = Path(__file__).parent / "data" / "car_fuel_efficiency_2026.csv"


sys.stdout.reconfigure(encoding="utf-8")  # acentos OK en la consola de Windows


def load_data() -> pd.DataFrame:
    """Descarga el CSV una sola vez y lo lee con pandas."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print(f"Descargando {DATA_URL} ...")
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


df = load_data()

# Q1. Versión de pandas
header("Q1. Pandas version")
print("pandas:", pd.__version__)

# Q2. Cantidad de registros
header("Q2. Records count")
print("Filas:", len(df), "| Columnas:", df.shape[1])

# Q3. Tipos de combustible
header("Q3. Fuel types")
print(df["fuel_type"].value_counts())
print("Cantidad de tipos:", df["fuel_type"].nunique())

# Q4. Columnas con valores faltantes
header("Q4. Missing values")
missing = df.isnull().sum()
print(missing)
print("Columnas con NA:", int((missing > 0).sum()))

# Q5. Máxima eficiencia de autos de Asia
header("Q5. Max fuel efficiency (Asia)")
print("Max MPG Asia:", df.loc[df["origin"] == "Asia", "fuel_efficiency_mpg"].max())

# Q6. Mediana de horsepower antes y después de imputar con la moda
header("Q6. Median horsepower")
median_before = df["horsepower"].median()
mode_hp = df["horsepower"].mode()[0]
median_after = df["horsepower"].fillna(mode_hp).median()
print(f"Mediana antes: {median_before}")
print(f"Moda: {mode_hp}")
print(f"Mediana después de fillna(moda): {median_after}")
if median_after > median_before:
    print("-> Yes, it increased")
elif median_after < median_before:
    print("-> Yes, it decreased")
else:
    print("-> No")

# Q7. Regresión lineal "a mano" con la ecuación normal
header("Q7. Sum of weights")
X = df.loc[df["origin"] == "Asia", ["vehicle_weight", "model_year"]].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print("X =\n", X)
print("w =", w)
print("Suma de w:", round(w.sum(), 4))
