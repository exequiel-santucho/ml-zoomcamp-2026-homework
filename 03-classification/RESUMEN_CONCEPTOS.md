# Módulo 3: Machine Learning para Clasificación - Resumen de Conceptos Clave

## Introducción

Proyecto: **predicción de churn** (abandono de clientes) en una telco (~7.000 clientes, ~27% se va). El objetivo de negocio: identificar a quién le conviene ofrecer un descuento para retenerlo. Pasamos de predecir un número (regresión) a predecir una **probabilidad de pertenecer a una clase** (clasificación binaria).

Herramientas nuevas: **Scikit-Learn** (`train_test_split`, `DictVectorizer`, `LogisticRegression`), medidas de **importancia de features** (risk ratio, mutual information, correlación) y **one-hot encoding**.

---

## 1. Clasificación binaria

$$ g(x_i) \approx y_i, \qquad y_i \in \{0, 1\} $$

- **1 = clase positiva** (churn, spam, convierte, default).
- **0 = clase negativa**.
- El modelo devuelve **P(y = 1 | x)**, un número entre 0 y 1. Después decidimos con un **umbral** (por defecto 0.5).

---

## 2. Preparación de datos (problemas reales que aparecieron)

```python
df.columns = df.columns.str.lower().str.replace(" ", "_")
# totalcharges venía como texto porque los faltantes eran " " (espacio)
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce").fillna(0)
df.churn = (df.churn == "yes").astype(int)   # target a 0/1
```

> Lección: **siempre revisar los `dtypes`**. Una columna numérica leída como `object` es señal de datos sucios.

---

## 3. Validación con Scikit-Learn

```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
# 0.25 de 80% = 20% → 60/20/20

y_train = df_train.pop("churn").values   # pop: extrae y elimina el target
```

---

## 4. EDA y tasa global (global churn rate)

```python
global_churn = df_full_train.churn.mean()   # ≈ 0.27
```

Como el target es 0/1, **la media es la proporción de positivos**. Es la referencia contra la que comparamos cada grupo.

---

## 5. Importancia de features categóricas

### Diferencia y risk ratio

Para cada grupo de una categórica (ej. `contract = month-to-month`):

| Medida | Fórmula | Interpretación |
|---|---|---|
| **Diferencia** | `global − grupo` | > 0: el grupo churnea **menos**; < 0: **más** |
| **Risk ratio** | `grupo / global` | > 1: más riesgo; < 1: menos riesgo |

```python
df_group = df_full_train.groupby("contract").churn.agg(["mean", "count"])
df_group["diff"] = df_group["mean"] - global_churn
df_group["risk"] = df_group["mean"] / global_churn
```

Ejemplo: contrato mes-a-mes → risk ≈ 1.6 (60% más churn); contrato de 2 años → risk ≈ 0.1.

### Mutual Information (información mutua)

Mide **cuánto aprendemos del target sabiendo el valor de la categórica**. Concepto de teoría de la información.

```python
from sklearn.metrics import mutual_info_score
mutual_info_score(df_full_train.churn, df_full_train.contract)  # 0.098, la más alta
```

- 0 → la variable no dice nada del target.
- Más alto → más informativa.
- Sirve para **rankear** categóricas por importancia.

---

## 6. Importancia de features numéricas: correlación

**Coeficiente de correlación de Pearson** ∈ [−1, 1]:

| Valor | Interpretación |
|---|---|
| ~0 | Sin relación lineal |
| 0.2–0.5 | Relación moderada |
| > 0.6 | Relación fuerte |
| Signo + | Si una sube, la otra sube |
| Signo − | Si una sube, la otra baja |

```python
df_full_train[numerical].corrwith(df_full_train.churn)
# tenure: -0.35 → cuanto más tiempo es cliente, menos churn
```

> Correlación ≠ causalidad. Y solo detecta relaciones **lineales**.

---

## 7. One-hot encoding con `DictVectorizer`

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)   # aprende el vocabulario (fit) y transforma

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)           # ¡solo transform! nunca fit en validación

dv.get_feature_names_out()                # ['contract=month-to-month', ..., 'tenure']
```

- Las **strings** se convierten en columnas binarias `col=valor`.
- Los **números** pasan tal cual.
- Categorías nuevas en validación/test (no vistas en train) se ignoran (todas sus columnas = 0).

---

## 8. Regresión logística

Es una regresión lineal "aplastada" por la **función sigmoide**:

$$ z = w_0 + x^T w \qquad g(x) = \sigma(z) = \frac{1}{1 + e^{-z}} $$

| z (score lineal) | σ(z) |
|---|---|
| −∞ | 0 |
| 0 | 0.5 |
| +∞ | 1 |

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)

model.coef_[0]        # pesos
model.intercept_[0]   # bias
y_pred = model.predict_proba(X_val)[:, 1]   # columna 1 = P(clase positiva)
churn_decision = y_pred >= 0.5
accuracy = (churn_decision == y_val).mean()
```

- Se entrena minimizando la **log-loss / binary cross-entropy** (no hay fórmula cerrada como la ecuación normal: se usa optimización iterativa).
- `predict_proba` devuelve dos columnas: `[P(0), P(1)]`.

### Regularización: parámetro `C`

- **`C` es la INVERSA de la fuerza de regularización** (al revés que `r` en el Módulo 2).
- **C chico** → regularización fuerte → pesos chicos → modelo simple.
- **C grande** → poca regularización.

---

## 9. Interpretación del modelo

```python
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))
```

- **Peso positivo** → aumenta la probabilidad de churn.
- **Peso negativo** → la disminuye.
- Como en regresión lineal: el score es `bias + Σ pesoᵢ · featureᵢ`, después se aplica sigmoide.

**Ojo con las escalas:** el peso de una feature numérica depende de sus unidades. Comparar pesos solo tiene sentido si las features están en escalas parecidas (o estandarizadas).

---

## 10. Usar el modelo

1. Re-entrenar con **full_train** (train + val).
2. Validar en **test** (accuracy ≈ 81.5% en el curso, parecido a validación → generaliza).
3. Para un cliente nuevo: `dict` → `dv.transform([cliente])` → `model.predict_proba(X)[0, 1]`.

---

## 11. Lo que resolvimos en la tarea (lead scoring 2026, target `converted`)

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Moda | `technology` |
| Q2 | Matriz de correlación | `interaction_count` ~ `lead_score` (0.916) |
| Q3 | Mutual information | `lead_source` (0.03) |
| Q4 | Regresión logística + OHE | accuracy 0.65 |
| Q5 | Eliminación de features | `number_of_courses_viewed` (diff 0.002) |
| Q6 | Regularización C | C = 0.001 |

### 🔎 Hallazgo importante de Q5: el problema de la escala

Al quitar `annual_income` la accuracy **subió** de 0.645 a **0.724**. ¿Por qué una feature empeora el modelo?

- `annual_income` está en decenas de miles (~55.000); las demás están entre 0 y 10.
- La regularización (C=1) penaliza todos los pesos por igual, así que la feature con escala gigante **domina** la optimización y el modelo no aprovecha bien el resto.
- **Solución profesional:** escalar las numéricas (`StandardScaler`) o transformarlas (log). El repo tiene `notebook-scaling-ohe.ipynb` justamente sobre esto.

Esto también explica Q6: con C muy chico (regularización fuerte) los pesos se achican y el modelo pierde accuracy (0.598).

---

## Preguntas típicas de entrevista

1. **¿Por qué no usar regresión lineal para clasificar?** Sus salidas no están acotadas en [0,1] y no se interpretan como probabilidad; la logística aplica sigmoide y optimiza log-loss.
2. **¿Qué hace la sigmoide?** Convierte cualquier número real en una probabilidad (0, 1).
3. **¿Qué es one-hot encoding? ¿Problemas?** Columnas binarias por categoría. Problema: alta cardinalidad → muchas columnas (soluciones: top-N, hashing, target encoding).
4. **¿Qué significa `C` en `LogisticRegression`?** La inversa de la fuerza de regularización.
5. **¿Mutual information vs correlación?** MI sirve para categóricas y captura relaciones no lineales; Pearson es para numéricas y solo lineal.
6. **¿Por qué `fit_transform` en train y solo `transform` en val/test?** Para no filtrar información de val/test al preprocesamiento (data leakage) y mantener las mismas columnas.
7. **¿Por qué escalar features en modelos lineales regularizados?** La penalización depende del tamaño de los pesos, que depende de la escala de cada feature.

## Recursos

- Lecciones: `machine-learning-zoomcamp/03-classification/`
- Notebook de escalado: `machine-learning-zoomcamp/03-classification/notebook-scaling-ohe.ipynb`
- Código de la tarea: `hw3_questions.py`
