## Homework 3: Machine Learning para Clasificación

> Dataset: `course_lead_scoring_2026.csv` (5.000 leads). Target: `converted` (¿el lead se registró en la plataforma?).
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/03-classification/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03 (vencía el 05/10/2026)

### Preparación

* Revisar faltantes. Categóricas → `'NA'`; numéricas → `0.0`.

Faltantes encontrados: `lead_source` 148, `industry` 240, `employment_status` 194, `location` 208, `annual_income` 369, `lead_score` 35.

### Pregunta 1

¿Cuál es la moda de `industry`?

- `NA`
- **`technology`** ✓ (1173)
- `healthcare`
- `retail`

### Pregunta 2

Matriz de correlación de las numéricas. ¿Cuál par tiene la mayor correlación (entre los listados)?

- **`interaction_count` y `lead_score`** ✓ (0.916)
- `number_of_courses_viewed` y `lead_score` (0.757)
- `number_of_courses_viewed` y `interaction_count` (0.722)
- `annual_income` y `interaction_count` (0.123)

### Split

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
```

### Pregunta 3

Mutual information entre `converted` y las categóricas (solo train), redondeado a 2 decimales. ¿Cuál es la mayor?

- `industry` (0.00)
- `location` (0.00)
- **`lead_source`** ✓ (0.03)
- `employment_status` (0.02)

### Pregunta 4

Regresión logística con one-hot encoding:
`LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`.
Accuracy en validación redondeada a 2 decimales:

- 0.55
- **0.65** ✓ (0.645)
- 0.75
- 0.85

### Pregunta 5

Eliminación de features: entrenar sin cada feature y calcular `accuracy_original − accuracy_sin_feature`. ¿Cuál tiene la menor diferencia?

| Feature quitada | Accuracy | Diferencia |
|---|---|---|
| `lead_source` | 0.642 | 0.003 |
| **`number_of_courses_viewed`** ✓ | 0.643 | **0.002** |
| `interaction_count` | 0.601 | 0.044 |

> Dato extra: sin `annual_income` la accuracy sube a 0.724 (diff −0.079) por su escala enorme sin estandarizar.

### Pregunta 6

Regresión logística regularizada con `C` en `[0.000001, 0.00001, 0.0001, 0.001]`, accuracy redondeada a 3 decimales.

| C | Accuracy |
|---|---|
| 0.000001 | 0.598 |
| 0.00001 | 0.598 |
| 0.0001 | 0.613 |
| **0.001** ✓ | **0.645** |

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | technology |
| Q2 | interaction_count y lead_score |
| Q3 | lead_source |
| Q4 | 0.65 |
| Q5 | number_of_courses_viewed |
| Q6 | 0.001 |
