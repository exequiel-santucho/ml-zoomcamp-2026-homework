## Homework 1: Introducción a Machine Learning

> Usa el release fijado 2026 del dataset de eficiencia de combustible (`car_fuel_efficiency_2026.csv`).
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/01-intro/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01 (vencía el 21/09/2026)

## Preparar el entorno

Instalar Python, NumPy, Pandas, Matplotlib y Seaborn (acá usamos `uv`, ver `README.md`).

## Q1. Versión de Pandas

¿Qué versión de Pandas instalaste? (`pd.__version__`)

**Respuesta:** `2.3.2` (la versión de referencia del release 2026)

## Obtener los datos

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

## Q2. Cantidad de registros

¿Cuántos registros tiene el dataset?

- 5000
- 9000
- **10000** ✓
- 15000

## Q3. Tipos de combustible

¿Cuántos tipos de combustible hay en el dataset?

- 1
- 2
- **3** ✓ (Gasoline 6913, Diesel 2053, Hybrid 1034)
- 4

## Q4. Valores faltantes

¿Cuántas columnas tienen valores faltantes?

- 0
- 1
- **2** ✓ (`horsepower`: 877, `acceleration`: 264)
- 3
- 4

## Q5. Máxima eficiencia

¿Cuál es la máxima eficiencia de combustible de los autos de Asia?

- 21.2
- 31.2
- **41.2** ✓
- 51.2

## Q6. Mediana de horsepower

1. Calcular la mediana de `horsepower`.
2. Calcular el valor más frecuente (moda) de esa columna.
3. Rellenar los faltantes de `horsepower` con la moda usando `fillna`.
4. Volver a calcular la mediana.

¿Cambió?

- Sí, aumentó
- **Sí, disminuyó** ✓
- No

**Explicación:** mediana original = 254, moda = 252. Al agregar 877 valores iguales a 252 (por debajo de la mediana), la mediana nueva pasa a 252.

## Q7. Suma de los pesos

1. Seleccionar los autos de Asia.
2. Quedarse con `vehicle_weight` y `model_year`.
3. Tomar las primeras 7 filas.
4. Obtener el array de NumPy → `X`.
5. Calcular `XTX = X.T @ X`.
6. Invertir `XTX`.
7. Crear `y = [1100, 1300, 800, 900, 1000, 1100, 1200]`.
8. `w = inv(XTX) @ X.T @ y`.
9. ¿Cuál es la suma de los elementos de `w`?

> Nota: acabás de implementar regresión lineal (ecuación normal). Lo vemos en el Módulo 2.

- 0.0369
- **0.369** ✓ (w = [0.1364, 0.2327] → 0.3692)
- 3.69
- 36.9

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 2.3.2 |
| Q2 | 10000 |
| Q3 | 3 |
| Q4 | 2 |
| Q5 | 41.2 |
| Q6 | Sí, disminuyó |
| Q7 | 0.369 |
