# Módulo 9 - Serverless (Homework 9)

Inferencia con **ONNX Runtime** y un handler de **AWS Lambda** probado localmente en Docker.

## Archivos

- **hw9_questions.py**: verifica checksums y resuelve Q1-Q4 (grafo ONNX, preprocesamiento, inferencia).
- **lambda_function.py**, **smoke_test.py**, **Dockerfile**, **requirements*.txt**, **asset_manifest.json**: release oficial 2026.
- `hair_classifier_v1.onnx`, `hair_classifier_v1.onnx.data`, `sample.jpeg`: se descargan (no se suben a Git).

## Preparación

```bash
PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
curl -fL -o hair_classifier_v1.onnx.data "${PREFIX}/hair_classifier_v1.onnx.data"
curl -fL -o hair_classifier_v1.onnx "${PREFIX}/hair_classifier_v1.onnx"
curl -fL -o sample.jpeg "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

## Ejecutar

```bash
# Q1-Q4
.venv/Scripts/python hw9_questions.py        # Windows  (.venv/bin/python en Linux/Mac)
.venv/Scripts/python smoke_test.py

# Q6: contenedor Lambda
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless
curl -s -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
```

## Salida esperada

```
Q1 output: output ['s77', 1] tensor(float)
Q3 tensor[0,0,0,0]=-1.056
Q4 probabilidad 'straight' = 0.728
Q6 {"statusCode": 200, ..., "body": "{\"straight_probability\": 0.727697, \"straight\": true}"}
```
