## Homework 6: Árboles de Decisión y Ensemble Learning

> Dataset: `car_fuel_efficiency_2026.csv`. Objetivo: regresión de `fuel_efficiency_mpg`.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/06-trees/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06 — **vence el 26/10/2026, 23:00 UTC**

### Preparación

* Rellenar faltantes con 0.
* Split:

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```

* `DictVectorizer(sparse=True)` para obtener las matrices.

## Pregunta 1

Árbol de regresión con `max_depth=1`. ¿Qué feature original usa para el split?

* `'vehicle_weight'`
* **`'model_year'`** ✓ (`model_year <= 1997.5` → 28.61 / 31.17 MPG)
* `'origin'`
* `'fuel_type'`

## Pregunta 2

Random Forest con `n_estimators=10`, `random_state=1`, `n_jobs=-1`. ¿RMSE en validación?

* 0.1837
* **1.837** ✓ (1.8371)
* 18.37
* 183.7

## Pregunta 3

Probar `n_estimators` en `[10, 50, 100, 150]` (`random_state=1`). ¿Cuál da el menor RMSE (3 decimales)?

| n_estimators | RMSE |
|---|---|
| 10 | 1.837 |
| 50 | 1.772 |
| **100** ✓ | **1.768** |
| 150 | 1.769 |

## Pregunta 4

Para `max_depth` en `[10, 15, 20, 25]`, probar `n_estimators` en `[10, 50, 100, 150]` y calcular el RMSE medio. ¿Mejor `max_depth`?

| max_depth | RMSE medio |
|---|---|
| **10** ✓ | **1.7554** |
| 15 | 1.7840 |
| 20 | 1.7882 |
| 25 | 1.7863 |

## Pregunta 5

Random Forest con `n_estimators=10`, `max_depth=20`, `random_state=1`. ¿Feature más importante (entre estas 4)?

* **`vehicle_weight`** ✓ (0.204)
* `horsepower` (0.078)
* `acceleration` (0.055)
* `engine_displacement` (0.061)

> La más importante en general es `model_year` (0.296), pero no está entre las opciones.

## Pregunta 6

XGBoost, 100 rondas, con:

```python
xgb_params = {'eta': 0.3, 'max_depth': 6, 'min_child_weight': 1,
              'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 1}
```

Cambiar `eta` a 0.1. ¿Cuál da mejor RMSE en validación?

* 0.3 → 1.8265
* **0.1** ✓ → **1.7248**
* Ambos iguales

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | model_year |
| Q2 | 1.837 |
| Q3 | 100 |
| Q4 | 10 |
| Q5 | vehicle_weight |
| Q6 | 0.1 |
