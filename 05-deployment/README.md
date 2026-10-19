# Módulo 5 - Deployment (Homework 5)

Servir el modelo congelado de lead scoring 2026 con **FastAPI + uvicorn**, entorno **uv** bloqueado y **Docker**.

## Archivos

| Archivo | Origen | Para qué |
|---|---|---|
| `pipeline.bin` | release 2026 | Pipeline pickleado (DictVectorizer + LogisticRegression) |
| `model.py` | release 2026 | `normalize_record()`: política de faltantes de training |
| `predict.py` | release 2026 | API FastAPI: `GET /health`, `POST /predict` |
| `train.py` | release 2026 | Cómo se entrenó el artefacto (no hace falta correrlo) |
| `feature_defaults.json`, `model_metadata.json` | release 2026 | Medianas de imputación y checksums |
| `smoke_test.py`, `q6_test.py` | release 2026 | Pruebas del modelo y del API |
| `pyproject.toml`, `uv.lock`, `.python-version` | release 2026 | Entorno fijado (Python 3.11.15, sklearn 1.7.2) |
| `Dockerfile` | release 2026 | Imagen canónica |
| **`hw5_q3_load_model.py`** | **mío** | Q3: verifica SHA-256 y predice con pickle |
| **`.gitattributes`** | **mío** | Evita que Git en Windows cambie los checksums (CRLF) |

## Ejecutar

```bash
# Entorno (si tu uv no conoce Python 3.11.15, usá cualquier 3.11: --python 3.11)
uv sync --locked
uv run python smoke_test.py

# Q3
uv run python hw5_q3_load_model.py

# Q4: API local (en una terminal)...
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
# ...y en otra
uv run python q6_test.py
curl -s http://localhost:9696/health

# Q6: contenedor
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
uv run python q6_test.py
```

## Salida esperada

```
Q3 -> probabilidad de conversión = 0.532994 ≈ 0.533
Q4 -> {'conversion_probability': 0.769799, 'conversion': True}
Q6 -> {'conversion_probability': 0.769799, 'conversion': True}   (desde Docker)
```

## Nota (Windows)

Con `core.autocrlf=true`, Git convierte `feature_defaults.json` a CRLF y el checksum que reporta `/health`
cambia (`5cb9...` en lugar de `1694...`). Las predicciones no cambian, pero para que el artefacto coincida
byte a byte hay que dejar los JSON en LF. El `.gitattributes` de esta carpeta lo garantiza.

Esta misma imagen se reutiliza en el **Homework 10** (Kubernetes) con el tag `zoomcamp-model:2026-hw10`.
