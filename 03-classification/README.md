# Módulo 3 - Clasificación (Homework 3)

Lead scoring con `DictVectorizer` + `LogisticRegression` de Scikit-Learn.

## Archivos

- **hw3_questions.py**: imputación, correlación, mutual information, regresión logística, eliminación de features y tuning de `C`.
- **data/**: se descarga sola la primera vez.

## Ejecutar

```bash
uv sync
uv run python hw3_questions.py
```

## Salida esperada

```
Q1 Moda: technology
Q2 interaction_count ~ lead_score: 0.9157
Q3 lead_source 0.03
Q4 Accuracy: 0.65
Q5 number_of_courses_viewed (diff +0.0020)
Q6 C=0.001 -> 0.645
```
