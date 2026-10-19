# Módulo 5: Deploy de Modelos de Machine Learning - Resumen de Conceptos Clave

## Introducción

Un modelo que vive solo en un notebook **no le sirve a nadie**. Este módulo toma el modelo de churn y lo convierte en un **servicio web** que otros sistemas pueden consultar, empaquetado para que corra igual en cualquier máquina y desplegado en la nube.

```
Notebook ──► train.py ──► model.bin ──► predict.py (API) ──► entorno (uv/pipenv) ──► Docker ──► Nube
```

El curso tiene dos versiones del stack; conviene conocer ambas:

| Paso | Versión clásica (videos) | Versión moderna (workshop 2025-26 / tarea) |
|---|---|---|
| Web framework | Flask | **FastAPI** |
| Servidor | gunicorn / waitress | **uvicorn** |
| Entorno y dependencias | Pipenv (`Pipfile.lock`) | **uv** (`pyproject.toml` + `uv.lock`) |
| Modelo | `(dv, model)` por separado | **Pipeline de Scikit-Learn** (un solo objeto) |
| Validación de entrada | Manual | **Pydantic** |
| Nube | AWS Elastic Beanstalk | Fly.io (o cualquiera que corra contenedores) |

---

## 1. Guardar y cargar el modelo: pickle

```python
import pickle

# guardar (en train.py)
with open("model.bin", "wb") as f_out:     # "wb" = write binary
    pickle.dump((dv, model), f_out)

# cargar (en predict.py)
with open("model.bin", "rb") as f_in:      # "rb" = read binary
    dv, model = pickle.load(f_in)
```

- Hay que guardar **todo lo necesario para predecir**: el `DictVectorizer` *y* el modelo. Sin el `dv` no podemos transformar los datos nuevos igual que en training.
- Al cargar, **las librerías deben estar instaladas en la misma versión** (por eso fijamos versiones).

### Pipelines: un objeto en vez de dos

```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(DictVectorizer(), LogisticRegression(solver="liblinear"))
pipeline.fit(train_dicts, y_train)
pipeline.predict_proba([cliente])[0, 1]
```

Evita el error clásico de olvidar el preprocesamiento en producción (**training-serving skew**).

### ⚠️ Seguridad de pickle

`pickle.load` puede **ejecutar código arbitrario**. Nunca cargar un `.bin` de origen desconocido. En la tarea 2026 se verifica el **SHA-256** del artefacto antes de usarlo:

```bash
sha256sum pipeline.bin   # debe coincidir con model_metadata.json
```

---

## 2. Del notebook al script

Jupyter → *Download as .py* → limpiar → dos scripts:

- **`train.py`**: lee datos, entrena (con los hiperparámetros elegidos en los módulos 3-4), evalúa y guarda `model.bin`.
- **`predict.py`**: carga el modelo y expone la predicción.

Parámetros (C, n_splits, ruta de salida) arriba del script como constantes o por argumentos (`argparse`).

---

## 3. Servicios web y HTTP

Un **web service** es un programa al que se le habla por la red usando **HTTP**.

| Método | Uso | Ejemplo |
|---|---|---|
| `GET` | Pedir información | `GET /health` |
| `POST` | Enviar datos para procesar | `POST /predict` con un JSON |

Flujo: cliente (app de marketing) → `POST /predict` con los datos del cliente en JSON → el servicio responde `{"churn_probability": 0.76, "churn": true}`.

---

## 4. Flask (versión clásica)

```python
from flask import Flask, request, jsonify

app = Flask("churn")

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return jsonify({"churn_probability": float(y_pred), "churn": bool(y_pred >= 0.5)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
```

- `float()` y `bool()`: los tipos de NumPy (`np.float64`, `np.bool_`) **no son serializables a JSON**.
- `app.run(debug=True)` es el **servidor de desarrollo**: no usar en producción.
- **Producción**: `gunicorn --bind 0.0.0.0:9696 predict:app` (Linux/Mac) o `waitress-serve --listen=0.0.0.0:9696 predict:app` (Windows).

---

## 5. FastAPI + Pydantic (versión moderna, la de la tarea)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="lead scoring")

class Lead(BaseModel):              # esquema de entrada: valida tipos automáticamente
    lead_source: str | None = None
    number_of_courses_viewed: int | None = None
    annual_income: float | None = None
    # ...

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(lead: Lead):
    record = normalize_record(lead.model_dump())
    p = float(pipeline.predict_proba([record])[0, 1])
    return {"conversion_probability": round(p, 6), "conversion": p >= 0.5}
```

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
# docs interactivas gratis en http://localhost:9696/docs
```

**Ventajas sobre Flask:** validación automática (un `"abc"` en un campo `int` devuelve 422 en vez de romper el modelo), documentación OpenAPI/Swagger automática, tipado, async y muy buen rendimiento.

**Endpoint `/health`**: estándar en producción; lo usan load balancers y Kubernetes (readiness probe, Módulo 10) para saber si el servicio está vivo.

**`normalize_record`**: aplica en producción **exactamente la misma política de faltantes que en training** (categóricas → `"NA"`, numéricas → mediana del train guardada en `feature_defaults.json`).

---

## 6. Gestión de entornos y dependencias

**Problema:** el servicio A necesita `scikit-learn==1.7.2` y el B `1.3.0`. Instalarlo todo global rompe uno de los dos. Además, el modelo pickleado con una versión puede no cargar en otra.

**Solución:** un **entorno virtual por proyecto** + **lockfile** con versiones exactas.

| Herramienta | Archivos | Comandos |
|---|---|---|
| venv + pip | `requirements.txt` | `python -m venv .venv`, `pip install -r ...` |
| Pipenv | `Pipfile`, `Pipfile.lock` | `pipenv install scikit-learn==1.7.2`, `pipenv shell` |
| Conda | `environment.yml` | `conda create`, `conda activate` |
| Poetry | `pyproject.toml`, `poetry.lock` | `poetry add` |
| **uv** | `pyproject.toml`, **`uv.lock`** | `uv add`, `uv sync --locked`, `uv run` |

**Lockfile**: fija versión exacta **y hashes** de todas las dependencias (incluso las transitivas) → instalaciones idénticas en todas las máquinas.

```bash
uv sync --locked   # falla si el lock no coincide con pyproject → reproducibilidad garantizada
```

---

## 7. Docker: aislar todo el sistema

Un entorno virtual aísla librerías de Python; **Docker aísla todo**: sistema operativo, Python, librerías del sistema y código. "En mi máquina funciona" deja de ser un problema.

- **Imagen**: plantilla inmutable (como una clase).
- **Contenedor**: una imagen corriendo (como una instancia).
- **Dockerfile**: receta para construir la imagen. Cada instrucción = una **capa** cacheada.

Dockerfile de la tarea 2026 (anotado):

```dockerfile
# Imagen base fijada por tag Y digest (reproducibilidad total)
FROM python:3.11.15-slim-bookworm@sha256:d29f...

# Copiar el binario de uv desde su imagen oficial (multi-stage copy)
COPY --from=ghcr.io/astral-sh/uv:0.10.11@sha256:3472... /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /code

# 1º dependencias (cambian poco → capa cacheada)...
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# ...2º código y modelo (cambian seguido)
COPY model.py predict.py feature_defaults.json model_metadata.json pipeline.bin q6_test.py ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5     # -p HOST:CONTENEDOR
docker run -it --rm --entrypoint=bash zoomcamp-model:2026-hw5   # entrar a explorar
```

**Buenas prácticas que muestra este Dockerfile:**
- Imagen `slim` (más chica que la completa).
- **Orden de capas**: dependencias antes que el código → rebuilds rápidos.
- **Versiones fijadas** (tag + digest) → la imagen no cambia "sola" con el tiempo.
- `--host 0.0.0.0` → escuchar en todas las interfaces (con `127.0.0.1` no sería accesible desde fuera del contenedor).

---

## 8. Deploy en la nube

**AWS Elastic Beanstalk** (videos):
```bash
pipenv install awsebcli --dev
eb init -p docker -r eu-west-1 churn-serving
eb local run --port 9696        # probar local
eb create churn-serving-env     # crear el entorno en AWS
eb terminate churn-serving-env  # ¡borrar al terminar para no pagar!
```
Escala automáticamente (más instancias con más tráfico). Cuidado: deja el servicio **público**.

**Fly.io** (workshop): `fly launch` + `fly deploy` con el mismo Dockerfile.

---

## 9. Reproducibilidad: el tema central de la tarea 2026

La cohorte 2026 rediseñó la tarea para que el resultado **no dependa de la máquina**:

| Elemento | Cómo se fija |
|---|---|
| Datos | Checksum del CSV en `model_metadata.json` |
| Modelo | SHA-256 de `pipeline.bin` |
| Dependencias | `uv.lock` + `exclude-newer` en `pyproject.toml` |
| Python | `.python-version` = 3.11.15 |
| Imagen base | Tag + digest |
| Faltantes | Medianas guardadas en `feature_defaults.json` |

**Anécdota real de esta tarea:** en Windows, Git convirtió los saltos de línea de `feature_defaults.json` a CRLF (`core.autocrlf=true`) y el checksum del endpoint `/health` dejó de coincidir con el de `model_metadata.json`. Las predicciones eran idénticas, pero el artefacto "no era el mismo byte a byte". Solución: normalizar a LF (y un `.gitattributes` con `*.json text eol=lf` y `*.bin binary`). **Así de fino es el control de artefactos en MLOps.**

---

## 10. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Versión de uv (no se califica) | 0.8.11 local / 0.10.11 en la imagen |
| Q2 | Versión fijada en el lockfile | scikit-learn `1.7.2` |
| Q3 | Cargar con pickle y predecir | 0.533 |
| Q4 | Servir con FastAPI/uvicorn | 0.770 |
| Q5 | Imagen base del Dockerfile | `python:3.11.15-slim-bookworm` |
| Q6 | Misma request contra el contenedor | 0.770 |

---

## Preguntas típicas de entrevista

1. **¿Cómo llevarías un modelo de un notebook a producción?** Script de training reproducible → serializar el pipeline → API (FastAPI) con validación → dependencias fijadas → Docker → deploy (con health checks, logs y monitoreo).
2. **¿Qué es training-serving skew y cómo lo evitás?** Diferencias entre el preprocesamiento de training y el de producción. Se evita con pipelines serializados y código de features compartido.
3. **Riesgos de pickle.** Ejecución de código arbitrario y dependencia de versiones. Alternativas: verificar checksums, `skops`, ONNX, joblib con control de versiones.
4. **Diferencia entre imagen y contenedor.** La imagen es la plantilla inmutable; el contenedor es una instancia en ejecución.
5. **¿Por qué copiar `pyproject.toml`/`uv.lock` antes que el código en el Dockerfile?** Para aprovechar la caché de capas: cambiar código no reinstala dependencias.
6. **¿Flask vs FastAPI?** FastAPI trae validación con Pydantic, docs automáticas, tipado y async; Flask es más minimalista.
7. **¿Por qué `0.0.0.0` y no `localhost` dentro de Docker?** `localhost` dentro del contenedor solo acepta conexiones del propio contenedor.
8. **¿Qué es un lockfile?** Archivo con versiones exactas (y hashes) de todas las dependencias, incluidas las transitivas.

## Recursos

- Lecciones: `machine-learning-zoomcamp/05-deployment/` (y `workshop/` para FastAPI + uv)
- Código de la tarea: esta carpeta
