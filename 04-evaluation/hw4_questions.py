#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 4: Métricas de evaluación para clasificación (Homework 4).

Ejecutar:
    uv run python hw4_questions.py          # genera además plots/ con las curvas
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")  # sin ventana: guardamos los gráficos a archivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)
HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "course_lead_scoring_2026.csv"
PLOTS = HERE / "plots"
TARGET = "converted"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
df = load_data()
categorical = [c for c in df.columns if df[c].dtype == "object"]
numerical = [c for c in df.columns if c not in categorical and c != TARGET]
df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_train = df_train[TARGET].values
y_val = df_val[TARGET].values
features = categorical + numerical


def train(df_tr: pd.DataFrame, y: np.ndarray, C: float = 1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df_tr[features].to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X, y)
    return dv, model


def predict(df_: pd.DataFrame, dv, model) -> np.ndarray:
    X = dv.transform(df_[features].to_dict(orient="records"))
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
header("Q1. AUC de cada variable numérica como 'score' (train)")
for c in numerical:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:  # correlación negativa -> invertimos la variable
        auc = roc_auc_score(y_train, -df_train[c])
    print(f"{c:26s} AUC={auc:.4f}")

# ---------------------------------------------------------------------------
header("Q2. AUC del modelo en validación")
dv, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)
print("AUC val:", round(roc_auc_score(y_val, y_pred), 3))

# ---------------------------------------------------------------------------
header("Q3/Q4. Precision, Recall y F1 por threshold")
rows = []
for t in np.linspace(0, 1, 101):
    t = round(t, 2)
    pred_pos = y_pred >= t
    tp = int((pred_pos & (y_val == 1)).sum())
    fp = int((pred_pos & (y_val == 0)).sum())
    fn = int((~pred_pos & (y_val == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    rows.append((t, tp, fp, fn, p, r, f1))
scores = pd.DataFrame(rows, columns=["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])

valid = scores[(scores.precision > 0) | (scores.recall > 0)].copy()
valid["diff"] = (valid.precision - valid.recall).abs()
best_pr = valid.loc[valid["diff"].idxmin()]
print(f"Q3 -> intersección P/R en threshold={best_pr.threshold:.2f} "
      f"(P={best_pr.precision:.3f}, R={best_pr.recall:.3f})")

best_f1 = scores.loc[scores.f1.idxmax()]
print(f"Q4 -> F1 máximo en threshold={best_f1.threshold:.2f} (F1={best_f1.f1:.4f})")

PLOTS.mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(scores.threshold, scores.precision, label="precision")
plt.plot(scores.threshold, scores.recall, label="recall")
plt.plot(scores.threshold, scores.f1, label="F1", linestyle="--")
plt.axvline(best_pr.threshold, color="gray", alpha=0.4)
plt.xlabel("threshold"); plt.legend(); plt.title("Precision / Recall / F1")
plt.tight_layout(); plt.savefig(PLOTS / "precision_recall_f1.png", dpi=120)
print("Gráfico guardado en plots/precision_recall_f1.png")

# ---------------------------------------------------------------------------
def cv_auc(C: float):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    aucs = []
    for train_idx, val_idx in kfold.split(df_full_train):
        tr, va = df_full_train.iloc[train_idx], df_full_train.iloc[val_idx]
        dv_, m_ = train(tr, tr[TARGET].values, C=C)
        aucs.append(roc_auc_score(va[TARGET].values, predict(va, dv_, m_)))
    return np.array(aucs)


header("Q5. 5-Fold CV (C=1.0)")
aucs = cv_auc(1.0)
print("AUCs:", aucs.round(4))
print(f"mean={aucs.mean():.4f} std={aucs.std():.4f} -> std≈{round(aucs.std(), 3)}")

header("Q6. Tuning de C con 5-Fold CV")
for C in [0.000001, 0.001, 1]:
    a = cv_auc(C)
    print(f"C={C:<8} mean={round(a.mean(), 3)} std={round(a.std(), 3)}")
