## Homework 5: Deploy de Modelos de Machine Learning

> Release congelado de lead scoring 2026: artefacto, checksum del dataset, script de training, API y Dockerfile.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/05-deployment/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05 — **vence el 19/10/2026 a las 23:00 UTC (20:00 hora Argentina)**

El modelo es `DictVectorizer` + `LogisticRegression(solver="liblinear", C=1.0)` entrenado con 8 features:
4 categóricas (faltante → `NA`) y 4 numéricas (faltante → mediana del train).

## Setup

```bash
uv sync --locked
uv run python smoke_test.py
sha256sum pipeline.bin
# 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin  ✓
```

## Pregunta 1 — chequeo del entorno

Correr `uv --version` y anotarlo (no se califica).

**Respuesta:** `uv 0.8.11` en la máquina local (la imagen Docker usa `uv 0.10.11`).

## Pregunta 2 — dependencia fijada

En `pyproject.toml` / `uv.lock`, ¿qué versión de Scikit-Learn usa el entorno de referencia?

- `1.6.1`
- **`1.7.2`** ✓
- `1.8.0`
- `2.0.0`

## Pregunta 3 — cargar el modelo

Script que cargue `pipeline.bin` con `pickle` y llame a `predict_proba` para:

```json
{"lead_source": "paid_ads", "industry": "technology", "employment_status": "employed",
 "location": "north_america", "number_of_courses_viewed": 2, "annual_income": 79276.0,
 "interaction_count": 4, "lead_score": 0.41}
```

Probabilidad de conversión (3 decimales, tolerancia ±0.005):

**Respuesta: `0.533`** ✓ (0.532994) — ver `hw5_q3_load_model.py`

## Pregunta 4 — servir el modelo

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Enviar a `POST /predict`:

```json
{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed",
 "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0,
 "interaction_count": 7, "lead_score": 0.74}
```

`conversion_probability` (3 decimales, ±0.005):

**Respuesta: `0.770`** ✓ (respuesta: `{"conversion_probability": 0.769799, "conversion": true}`)

## Contenedor

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

## Pregunta 5 — configuración del contenedor

¿Qué tag de imagen base declara el Dockerfile?

- `python:3.9-slim-bullseye`
- **`python:3.11.15-slim-bookworm`** ✓
- `python:3.12-slim-bookworm`
- `python:3.13.10-slim-bookworm`

## Pregunta 6 — correr el contenedor

Misma request de Q4 contra el contenedor (`python q6_test.py`). `conversion_probability`:

**Respuesta: `0.770`** ✓

Verificación de `/health` dentro del contenedor:

```json
{"status":"ok",
 "model_sha256":"1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
 "feature_defaults_sha256":"1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"}
```

Ambos checksums coinciden con `model_metadata.json`. Python dentro del contenedor: `3.11.15`. Tamaño de la imagen local: 686 MB (no se califica).

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | uv 0.8.11 (no calificada) |
| Q2 | 1.7.2 |
| Q3 | 0.533 |
| Q4 | 0.770 |
| Q5 | python:3.11.15-slim-bookworm |
| Q6 | 0.770 |
