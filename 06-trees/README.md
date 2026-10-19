# Módulo 6 - Árboles y Ensembles (Homework 6)

Árbol de decisión, Random Forest y XGBoost para regresión de eficiencia de combustible.

## Archivos

- **hw6_questions.py**: Q1-Q6 (DecisionTreeRegressor, RandomForestRegressor, XGBoost con DMatrix + watchlist).
- **data/**: se descarga sola la primera vez.

## Ejecutar

```bash
uv sync
uv run python hw6_questions.py     # ~1-2 min (Q4 entrena 16 random forests)
```

Versiones de referencia: scikit-learn 1.7.2, xgboost 3.2.0.

## Salida esperada

```
Q1 model_year <= 1997.50
Q2 RMSE val: 1.8371
Q3 n_estimators=100 -> 1.768 (mínimo)
Q4 max_depth=10 -> 1.7554 (mínimo)
Q5 vehicle_weight 0.2039 (mayor entre las 4 opciones)
Q6 eta=0.3 -> 1.8265 | eta=0.1 -> 1.7248
```
