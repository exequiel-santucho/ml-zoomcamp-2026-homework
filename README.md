# ML Zoomcamp 2026 - Homework

Mis soluciones de las tareas de [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) de [DataTalksClub](https://datatalks.club/), junto con un resumen de conceptos por módulo.

## Contenido

| Módulo | Tema | Herramientas |
|---|---|---|
| [01-intro](01-intro/) | Introducción a ML, CRISP-DM, NumPy y Pandas | pandas, numpy |
| [02-regression](02-regression/) | Regresión lineal desde cero (ecuación normal, Ridge) | numpy |
| [03-classification](03-classification/) | Regresión logística, mutual information, one-hot encoding | scikit-learn |
| [04-evaluation](04-evaluation/) | Precision/recall, F1, ROC AUC, K-Fold CV | scikit-learn |
| [05-deployment](05-deployment/) | Servir un modelo con FastAPI, uv y Docker | FastAPI, uvicorn, uv, Docker |
| [06-trees](06-trees/) | Árboles de decisión, Random Forest, XGBoost | scikit-learn, xgboost |
| [08-deep-learning](08-deep-learning/) | CNN en PyTorch y data augmentation | PyTorch |
| [09-serverless](09-serverless/) | Inferencia con ONNX Runtime en AWS Lambda | onnxruntime, Docker |
| [10-kubernetes](10-kubernetes/) | Deploy en Kubernetes local (Deployment, Service, HPA) | kind, kubectl |

Cada carpeta tiene:

- `README.md`: cómo correr la solución y la salida esperada.
- `homework-es.md`: enunciado en español con las respuestas.
- `RESUMEN_CONCEPTOS.md`: resumen del módulo para repasar.
- Scripts `hwN_questions.py` y archivos de entorno (`pyproject.toml` + `uv.lock`, o `requirements*.txt`).

## Cómo correr

Módulos 01-06 (Python 3.11 + [uv](https://docs.astral.sh/uv/)):

```bash
cd 02-regression
uv sync
uv run python hw2_questions.py
```

Los datasets se descargan solos la primera vez. Los módulos 05, 08, 09 y 10 tienen pasos extra (Docker, PyTorch CPU, assets ONNX, kind): ver el README de cada carpeta.

## Reproducibilidad

Versiones de referencia de la cohorte 2026: Python 3.11, NumPy 2.3.3, Pandas 2.3.2, Scikit-Learn 1.7.2, XGBoost 3.2.0, torch 2.9.0+cpu, onnxruntime 1.23.2.

#mlzoomcamp
