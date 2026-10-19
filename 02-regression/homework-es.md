## Homework 2: Machine Learning para Regresión

> Dataset: `car_fuel_efficiency_2026.csv` (release fijado 2026).
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/02-regression/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02 (vencía el 28/09/2026)

El objetivo es un modelo de regresión que prediga `fuel_efficiency_mpg`.

### Preparar el dataset

Usar solo: `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`, `fuel_efficiency_mpg`.

### EDA

¿`fuel_efficiency_mpg` tiene cola larga? **No**: media ≈ mediana ≈ 30, skewness = 0.08 (distribución simétrica). No hace falta `log1p`.

### Pregunta 1

Hay una columna con valores faltantes. ¿Cuál es?

* `'engine_displacement'`
* **`'horsepower'`** ✓ (877 faltantes)
* `'vehicle_weight'`
* `'model_year'`

### Pregunta 2

¿Cuál es la mediana (percentil 50) de `horsepower`?

- 204
- **254** ✓
- 304
- 354

### Preparar y dividir

Mezclar y dividir 60/20/20 exactamente como en la lección (seed 42). Para Q5 repetir con cada semilla; para Q6 usar seed 9.

### Pregunta 3

* Rellenar los faltantes de la columna de Q1 con 0 o con la **media del train**.
* Entrenar regresión lineal sin regularización para cada opción.
* Comparar el RMSE en validación, redondeado a 3 decimales.

- With 0 → 2.205
- **With mean → 2.202** ✓
- Both are equally good

### Pregunta 4

* Regresión lineal regularizada, NA = 0.
* Probar `r` en `[0, 0.01, 0.1, 1, 5, 10, 100]`, RMSE en validación redondeado a 4 decimales.
* Si hay empate, elegir el `r` más chico.

| r | RMSE |
|---|---|
| **0** ✓ | **2.2053** |
| 0.01 | 2.2058 |
| 0.1 | 2.2241 |
| 1 | 2.3492 |
| 5 | 2.4094 |
| 10 | 2.4195 |
| 100 | 2.4292 |

### Pregunta 5

* Probar semillas `[0..9]` para el split, NA = 0, sin regularización.
* Desvío estándar (`np.std`) de los RMSE de validación, redondeado a 3 decimales.

- 0.006
- 0.016
- **0.029** ✓
- 0.036

> Un std bajo indica que el modelo es **estable** frente a cómo se dividen los datos.

### Pregunta 6

* Split con seed 9, combinar train + validation.
* NA = 0, entrenar con `r=0.001`.
* ¿RMSE en test?

- 0.236
- **2.236** ✓
- 22.10
- 221.0

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | horsepower |
| Q2 | 254 |
| Q3 | With mean |
| Q4 | 0 |
| Q5 | 0.029 |
| Q6 | 2.236 |
