# Módulo 1 - Introducción a ML (Homework 1)

Solución del Homework 1 de ML Zoomcamp 2026.

## Archivos

- **hw1_questions.py**: resuelve Q1-Q7 e imprime cada resultado.
- **data/**: se crea sola; el script descarga el CSV la primera vez.
- **pyproject.toml / uv.lock**: entorno reproducible (Python 3.11, pandas 2.3.2, numpy 2.3.3).

## Ejecutar

```bash
uv sync
uv run python hw1_questions.py
```

## Salida esperada

```
Q1 pandas: 2.3.2
Q2 Filas: 10000
Q3 Cantidad de tipos: 3
Q4 Columnas con NA: 2
Q5 Max MPG Asia: 41.2
Q6 Mediana 254.0 -> 252.0  (Yes, it decreased)
Q7 Suma de w: 0.3692
```

## Qué practica

Pandas (filtrado, `isnull`, `value_counts`, `fillna`, `median`/`mode`) y álgebra lineal con NumPy
(`X.T @ X`, `np.linalg.inv`): la **ecuación normal** de la regresión lineal.
