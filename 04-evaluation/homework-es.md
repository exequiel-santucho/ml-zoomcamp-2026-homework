## Homework 4: Métricas de Evaluación para Clasificación

> Dataset: `course_lead_scoring_2026.csv`. Target: `converted`.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/04-evaluation/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04 (vencía el 12/10/2026)

### Preparación

Categóricas faltantes → `'NA'`; numéricas → `0.0`.

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```

### Pregunta 1: importancia de features con ROC AUC

Para cada numérica, usarla como score y calcular el AUC contra `y` (train). Si AUC < 0.5, invertirla (`-df_train[col]`). ¿Cuál tiene el mayor AUC?

- **`lead_score`** ✓ (0.788)
- `number_of_courses_viewed` (0.723)
- `interaction_count` (0.765)
- `annual_income` (0.608)

### Pregunta 2: entrenar el modelo

One-hot con `DictVectorizer` + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`. AUC en validación (3 decimales):

- 0.532
- 0.632
- **0.732** ✓
- 0.832

### Pregunta 3: precision y recall

Evaluar umbrales de 0.0 a 1.0 (paso 0.01). ¿En qué umbral se cruzan las curvas? (ignorar umbrales donde ambas son 0; elegir la menor |P − R|)

* 0.43
* **0.63** ✓ (P = 0.733, R = 0.725)
* 0.73
* 0.83

### Pregunta 4: F1

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

¿En qué umbral el F1 es máximo?

- 0.21
- **0.41** ✓ (F1 = 0.758)
- 0.61
- 0.81

### Pregunta 5: 5-Fold CV

`KFold(n_splits=5, shuffle=True, random_state=1)` sobre `df_full_train`, mismo modelo, AUC en cada fold. ¿Desvío estándar?

- 0.001
- **0.007** ✓ (AUCs: 0.731, 0.730, 0.736, 0.741, 0.721 → 0.732 ± 0.0069)
- 0.013
- 0.060

### Pregunta 6: tuning de hiperparámetros

Probar `C` en `[0.000001, 0.001, 1]` con 5-Fold. Media y std redondeados a 3 decimales. ¿Qué `C` da la mejor media?

| C | Media AUC | Std |
|---|---|---|
| 0.000001 | 0.617 | 0.019 |
| **0.001** ✓ | **0.749** | 0.010 |
| 1 | 0.732 | 0.007 |

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | lead_score |
| Q2 | 0.732 |
| Q3 | 0.63 |
| Q4 | 0.41 |
| Q5 | 0.007 |
| Q6 | 0.001 |
