## Homework 9: Serverless Deep Learning

> Clasificador de pelo detrás de un handler compatible con AWS Lambda, corriendo en un contenedor Lambda local. Modelo e imagen fijados por SHA-256.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/09-serverless/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw09 — **vence el 30/11/2026, 23:00 UTC**

## Descargar los assets

```bash
PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
curl -fL -o hair_classifier_v1.onnx.data "${PREFIX}/hair_classifier_v1.onnx.data"
curl -fL -o hair_classifier_v1.onnx "${PREFIX}/hair_classifier_v1.onnx"
curl -fL -o sample.jpeg "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
sha256sum hair_classifier_v1.onnx hair_classifier_v1.onnx.data sample.jpeg   # los 3 coinciden con asset_manifest.json ✓
```

## Pregunta 1 — grafo ONNX

Con `session.get_inputs()` / `session.get_outputs()`, ¿cuál es el nombre del nodo de salida?

- **`output`** ✓ (forma `[batch, 1]`, float: probabilidad de `straight`)
- `sigmoid`
- `softmax`
- `prediction`

## Preprocesamiento

`resize((200, 200), BILINEAR)` → `/255` → normalización ImageNet → transponer a CHW → agregar batch.

## Pregunta 2 — tamaño objetivo

- `64x64`
- `128x128`
- **`200x200`** ✓
- `256x256`

## Pregunta 3 — entrada normalizada

Primer valor del canal R (`tensor[0, 0, 0, 0]`), 3 decimales (±0.01):

**Respuesta: `-1.056`** ✓ — píxel R = 0.2431 → (0.2431 − 0.485) / 0.229 = −1.056

## Pregunta 4 — inferencia ONNX local

```python
session = ort.InferenceSession("hair_classifier_v1.onnx", providers=["CPUExecutionProvider"])
output = session.run(["output"], {"input": prepare_image(download_image(SAMPLE_URL))})[0]
```

Probabilidad (3 decimales, ±0.02):

**Respuesta: `0.728`** ✓ (0.727697)

## Contenedor

```bash
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless
curl -s -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
```

## Pregunta 5 — configuración Lambda

¿Qué imagen base declara el Dockerfile?

- `public.ecr.aws/lambda/python:3.9`
- `public.ecr.aws/lambda/python:3.11`
- **`public.ecr.aws/lambda/python:3.13`** ✓
- `python:3.13-slim-bookworm`

## Pregunta 6 — invocar el contenedor

`straight_probability` en el body de la respuesta (3 decimales, ±0.02):

```json
{"statusCode": 200, "headers": {"Content-Type": "application/json"},
 "body": "{\"straight_probability\": 0.727697, \"straight\": true}"}
```

**Respuesta: `0.728`** ✓ (igual que Q4)

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | output |
| Q2 | 200x200 |
| Q3 | -1.056 |
| Q4 | 0.728 |
| Q5 | public.ecr.aws/lambda/python:3.13 |
| Q6 | 0.728 |
