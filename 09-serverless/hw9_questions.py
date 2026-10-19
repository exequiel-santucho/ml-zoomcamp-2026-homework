#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 9: Serverless Deep Learning (Homework 9).

Q1-Q4 corren localmente con ONNX Runtime. Q5-Q6 usan el contenedor Lambda
(ver README.md).

Ejecutar:
    .venv/Scripts/python hw9_questions.py      # Windows
    .venv/bin/python hw9_questions.py          # Linux/Mac
"""
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "asset_manifest.json").read_text(encoding="utf-8"))
FILES = {
    "model": HERE / "hair_classifier_v1.onnx",
    "external_data": HERE / "hair_classifier_v1.onnx.data",
    "sample_image": HERE / "sample.jpeg",
}

# 0) Verificar checksums de los assets
for key, path in FILES.items():
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "OK" if sha == MANIFEST[key]["sha256"] else "MISMATCH!"
    print(f"{path.name:32s} {status}")


def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize BILINEAR 200x200 -> [0,1] -> normalización ImageNet -> CHW -> batch."""
    image = image.resize((200, 200), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))[None, ...]  # HWC -> CHW y agrega batch


session = ort.InferenceSession(str(FILES["model"]), providers=["CPUExecutionProvider"])

print("\nQ1. Nodos del grafo ONNX")
for i in session.get_inputs():
    print("  input :", i.name, i.shape, i.type)
for o in session.get_outputs():
    print("  output:", o.name, o.shape, o.type)

print("\nQ2. Tamaño objetivo de prepare_image: 200x200")

# Usamos el archivo local (mismo checksum que la URL) para no depender de la red
image = Image.open(BytesIO(FILES["sample_image"].read_bytes())).convert("RGB")
tensor = prepare_image(image)
print(f"\nQ3. tensor.shape={tensor.shape}  tensor[0,0,0,0]={tensor[0, 0, 0, 0]:.3f}")

output = session.run(["output"], {"input": tensor})[0]
print(f"\nQ4. probabilidad 'straight' = {float(output.reshape(-1)[0]):.3f}")
