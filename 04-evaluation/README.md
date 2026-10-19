# Módulo 4 - Evaluación (Homework 4)

ROC AUC, precision/recall, F1 por umbral y validación cruzada K-Fold.

## Archivos

- **hw4_questions.py**: Q1-Q6. Calcula TP/FP/FN a mano para cada umbral (para entender las fórmulas) y usa `roc_auc_score` + `KFold` de Scikit-Learn.
- **plots/precision_recall_f1.png**: curvas de precision, recall y F1 vs umbral (se genera al correr).
- **data/**: se descarga sola la primera vez.

## Ejecutar

```bash
uv sync
uv run python hw4_questions.py
```

## Salida esperada

```
Q1 lead_score AUC=0.7882 (la mayor)
Q2 AUC val: 0.732
Q3 intersección P/R en threshold=0.63
Q4 F1 máximo en threshold=0.41
Q5 std≈0.007
Q6 C=0.001 mean=0.749 std=0.01 (mejor)
```
