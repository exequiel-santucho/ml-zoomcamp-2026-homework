#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 3: Machine Learning para Clasificación (Homework 3).

Dataset: lead scoring 2026 (target: `converted`).

Ejecutar:
    uv run python hw3_questions.py
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)
DATA_PATH = Path(__file__).parent / "data" / "course_lead_scoring_2026.csv"
TARGET = "converted"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------
df = load_data()
categorical = [c for c in df.columns if df[c].dtype == "object"]
numerical = [c for c in df.columns if c not in categorical and c != TARGET]
print("Categóricas:", categorical)
print("Numéricas:  ", numerical)

header("Valores faltantes (antes de imputar)")
print(df.isnull().sum())

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

# ---------------------------------------------------------------------------
header("Q1. Moda de `industry`")
print(df["industry"].value_counts().head())
print("Moda:", df["industry"].mode()[0])

# ---------------------------------------------------------------------------
header("Q2. Matriz de correlación (numéricas)")
corr = df[numerical].corr()
print(corr.round(3))
pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]
for a, b in pairs:
    print(f"{a} ~ {b}: {corr.loc[a, b]:.4f}")

# ---------------------------------------------------------------------------
# Split 60/20/20
# ---------------------------------------------------------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

y_train = df_train.pop(TARGET).values
y_val = df_val.pop(TARGET).values
y_test = df_test.pop(TARGET).values
print(f"\nSplit -> train={len(df_train)} val={len(df_val)} test={len(df_test)}")

# ---------------------------------------------------------------------------
header("Q3. Mutual information (categóricas vs converted, train)")
mi = {c: round(mutual_info_score(df_train[c], y_train), 2) for c in categorical}
for c, s in sorted(mi.items(), key=lambda kv: -kv[1]):
    print(f"{c:20s} {s}")


# ---------------------------------------------------------------------------
def train_and_eval(features, C=1.0) -> float:
    """One-hot (DictVectorizer) + regresión logística; devuelve accuracy en val."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
    X_val = dv.transform(df_val[features].to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))


all_features = categorical + numerical

header("Q4. Regresión logística - accuracy en validación")
base_acc = train_and_eval(all_features)
print("Accuracy:", round(base_acc, 2), f"(sin redondear: {base_acc:.4f})")

# ---------------------------------------------------------------------------
header("Q5. Eliminación de features")
for f in all_features:
    acc = train_and_eval([x for x in all_features if x != f])
    print(f"sin {f:26s} acc={acc:.4f}  diff={base_acc - acc:+.4f}")

# ---------------------------------------------------------------------------
header("Q6. Regularización (C)")
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    print(f"C={C:<8} acc={round(train_and_eval(all_features, C=C), 3)}")
