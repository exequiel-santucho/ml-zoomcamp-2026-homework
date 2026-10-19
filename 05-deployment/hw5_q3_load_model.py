#!/usr/bin/env python3
"""HW5 - Q3: cargar pipeline.bin con pickle y predecir la probabilidad de conversión.

Ejecutar:
    uv run python hw5_q3_load_model.py
"""
import hashlib
import pickle
from pathlib import Path

from model import normalize_record

MODEL_PATH = Path(__file__).with_name("pipeline.bin")
EXPECTED_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# 1) Verificar integridad del artefacto ANTES de deserializarlo
#    (pickle puede ejecutar código arbitrario: nunca cargues un .bin no confiable).
sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
assert sha == EXPECTED_SHA256, f"Checksum inesperado: {sha}"
print("sha256 OK:", sha)

# 2) Cargar el pipeline (DictVectorizer + LogisticRegression en un solo objeto)
with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)
print("Pipeline:", pipeline)

# 3) Aplicar la misma política de faltantes que en training y predecir
record = normalize_record(lead)
proba = float(pipeline.predict_proba([record])[0, 1])
print(f"Q3 -> probabilidad de conversión = {proba:.6f} ≈ {proba:.3f}")
