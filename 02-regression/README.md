# Módulo 2 - Regresión (Homework 2)

Regresión lineal **implementada desde cero con NumPy** (ecuación normal + regularización Ridge),
igual que en las lecciones. No usa Scikit-Learn para el modelo.

## Archivos

- **hw2_questions.py**: EDA + Q1-Q6. Contiene `split()`, `train_linear_regression()`, `rmse()` y `prepare_X()`.
- **data/**: se descarga sola la primera vez.

## Ejecutar

```bash
uv sync
uv run python hw2_questions.py
```

## Salida esperada

```
Q1 horsepower (877 NA)
Q2 Mediana: 254.0
Q3 With 0 -> 2.205 | With mean -> 2.202
Q4 r=0 -> 2.2053 (mejor)
Q5 std: 0.029
Q6 RMSE test: 2.236
```
