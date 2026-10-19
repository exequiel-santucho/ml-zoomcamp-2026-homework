# Módulo 6: Árboles de Decisión y Ensemble Learning - Resumen de Conceptos Clave

## Introducción

Proyecto: **credit risk scoring** — predecir si un cliente va a incumplir (default) un préstamo. Dejamos los modelos lineales y pasamos a los modelos más usados en datos tabulares en la industria y en Kaggle: **árboles de decisión**, **Random Forest** y **Gradient Boosting (XGBoost)**.

| Modelo | AUC validación (curso) |
|---|---|
| Árbol de decisión (tuneado) | ~0.785 |
| Random Forest | ~0.82 |
| **XGBoost** | **~0.836** (0.832 en test) |

---

## 1. Preparación de datos (credit scoring)

- Las categóricas venían codificadas como números (`status = 1, 2, 3`) → se **mapean a strings** (`{1: "ok", 2: "default"}`) para que `DictVectorizer` haga one-hot.
- Valores centinela: `99999999` significaba "faltante" → reemplazar por `NaN` y luego por 0.
- Target: `default = (status == "default").astype(int)`.

> Lección: los datos reales traen códigos mágicos. Leer el diccionario de datos es parte del trabajo.

---

## 2. Árboles de decisión

Un árbol aprende reglas **if-then-else** a partir de los datos:

```
¿records == "yes"?
├── sí → ¿job == "parttime"? → sí: DEFAULT / no: OK
└── no → ¿assets > 6000? → sí: OK / no: DEFAULT
```

```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
dt.fit(X_train, y_train)
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))
```

**Propiedades:**
- Capturan **relaciones no lineales** e **interacciones** entre features automáticamente.
- **No necesitan escalar** features (solo comparan con umbrales). ¡Gran diferencia con la regresión logística del Módulo 3!
- Fáciles de interpretar si son poco profundos.

### Overfitting

Sin límites, el árbol crece hasta que cada hoja tiene un solo ejemplo → **memoriza** el train (AUC train = 1.0, validación ≈ 0.65).

---

## 3. Cómo aprende un árbol: impureza y mejor split

Para cada feature y cada umbral posible, se divide el dataset en izquierda/derecha y se mide la **impureza** de cada lado:

| Criterio | Para | Fórmula |
|---|---|---|
| **Misclassification rate** | Clasificación (didáctico) | Fracción de errores si predigo la clase mayoritaria |
| **Gini** | Clasificación (default de sklearn) | 1 − Σ pₖ² |
| **Entropía** | Clasificación | −Σ pₖ log pₖ |
| **MSE** | **Regresión** | Varianza del target en el grupo |

**Impureza ponderada** del split = promedio de la impureza de cada lado pesado por su tamaño. Se elige el split con **menor impureza** (mayor *ganancia* de información).

**Algoritmo (greedy, recursivo):**
1. Buscar el mejor split (feature + umbral).
2. Dividir.
3. Repetir en cada hijo hasta un **criterio de parada**:
   - el grupo ya es puro,
   - se alcanzó `max_depth`,
   - el grupo es más chico que `min_samples_leaf`.

En la tarea, el árbol de regresión con `max_depth=1` eligió `model_year <= 1997.5` como primer split: es la variable que más reduce el MSE del MPG.

---

## 4. Tuning del árbol

Hiperparámetros principales:
- **`max_depth`**: profundidad máxima (controla complejidad).
- **`min_samples_leaf`**: mínimo de ejemplos por hoja (evita hojas "anecdóticas").

Estrategia del curso: primero barrer `max_depth` (ej. 4, 5, 6), después para esos valores barrer `min_samples_leaf`, y visualizar con un **heatmap** (`pivot` + `seaborn.heatmap`).

---

## 5. Ensembles y Random Forest

**Ensemble**: combinar muchos modelos. Analogía: preguntar a varios expertos y promediar → los errores individuales se compensan **si los expertos son diversos**.

**Random Forest:**
1. Entrenar N árboles, cada uno con:
   - un **bootstrap** del dataset (muestreo con reemplazo) → *bagging*,
   - un **subconjunto aleatorio de features** en cada split (`max_features`).
2. Predecir = **promedio** (regresión) o **votación/promedio de probabilidades** (clasificación).

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3,
                            random_state=1, n_jobs=-1)
```

| Hiperparámetro | Efecto |
|---|---|
| `n_estimators` | Más árboles → más estable, mejora hasta que se estanca (no overfittea por más árboles) |
| `max_depth` | Profundidad de cada árbol |
| `min_samples_leaf` | Tamaño mínimo de hoja |
| `max_features` | Cuántas features considera por split (más chico = más diversidad) |
| `random_state` | Reproducibilidad |
| `n_jobs=-1` | Entrenar en paralelo (los árboles son independientes) |

**Feature importance** (`rf.feature_importances_`): cuánto reduce la impureza cada feature, sumado sobre todos los splits y árboles.

---

## 6. Gradient Boosting y XGBoost

**Boosting**: entrenar modelos **en secuencia**, donde cada uno corrige los **errores (residuos)** de los anteriores.

```
modelo₁ → errores₁ → modelo₂ (aprende errores₁) → errores₂ → modelo₃ → ...
predicción = modelo₁ + η·modelo₂ + η·modelo₃ + ...
```

| | Random Forest (bagging) | Gradient Boosting |
|---|---|---|
| Entrenamiento | Paralelo, independiente | Secuencial |
| Árboles | Profundos | Poco profundos (débiles) |
| Reduce principalmente | Varianza | Sesgo |
| Riesgo de overfitting | Bajo | Mayor → requiere tuning |

### XGBoost

```python
import xgboost as xgb

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    "eta": 0.3,                    # learning rate
    "max_depth": 6,
    "min_child_weight": 1,         # ~ min_samples_leaf
    "objective": "binary:logistic",  # "reg:squarederror" para regresión
    "eval_metric": "auc",
    "nthread": 8, "seed": 1, "verbosity": 1,
}
watchlist = [(dtrain, "train"), (dval, "val")]
model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  evals=watchlist, verbose_eval=5)
```

- **`DMatrix`**: estructura de datos optimizada de XGBoost.
- **`watchlist`**: monitorea la métrica en train y val en cada ronda → se ve cuándo empieza el overfitting.
- Truco del curso: capturar la salida con `%%capture output` y parsearla para graficar AUC vs. número de árboles.

### Tuning de XGBoost (orden recomendado)

1. **`eta`** (learning rate): tamaño de cada paso. Más chico → aprende más lento pero mejor (necesita más rondas).
2. **`max_depth`**: complejidad de cada árbol (3-10).
3. **`min_child_weight`**: mínimo "peso" por hoja (1, 10, 30).
4. Opcionales: `subsample`, `colsample_bytree` (aleatoriedad estilo RF), `lambda`/`alpha` (regularización L2/L1).
5. **Número de rondas**: elegir donde la métrica de validación es máxima (o usar `early_stopping_rounds`).

En la tarea: `eta=0.1` dio RMSE 1.725 vs 1.827 con `eta=0.3` → con 100 rondas, pasos más chicos generalizan mejor.

---

## 7. Modelo final

1. Comparar los mejores de cada familia en validación → XGBoost ganó.
2. Re-entrenar en **full_train** con los hiperparámetros elegidos.
3. Evaluar en **test**: 0.832 vs 0.836 en validación → no hubo overfitting al validation set.

---

## 8. Lo que resolvimos en la tarea (regresión de MPG con árboles)

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Primer split de un árbol `max_depth=1` | `model_year` |
| Q2 | Random Forest base | RMSE 1.837 |
| Q3 | `n_estimators` | 100 (1.768; con 150 sube a 1.769) |
| Q4 | `max_depth` (RMSE medio) | 10 |
| Q5 | Feature importance | `vehicle_weight` (entre las 4 opciones) |
| Q6 | `eta` de XGBoost | 0.1 |

**Observaciones:**
- Comparado con la regresión lineal del Módulo 2 (RMSE ≈ 2.2), los árboles bajan a ≈ 1.73-1.77 usando **todas** las features (incluidas categóricas como `fuel_type` y `origin`) y capturando no linealidades.
- **Q4:** `max_depth=10` ganó a 15/20/25 → árboles menos profundos generalizan mejor en este dataset (regularización por profundidad).
- **Q5:** la feature más importante en general es `model_year` (0.296), pero entre las cuatro opciones de la pregunta la mayor es `vehicle_weight` (0.204).

---

## Preguntas típicas de entrevista

1. **¿Cómo elige un árbol el split?** Prueba todos los umbrales de todas las features y elige el que minimiza la impureza ponderada (Gini/entropía/MSE).
2. **¿Por qué los árboles no necesitan escalado?** Solo comparan una feature contra un umbral; la escala no cambia el orden.
3. **Bagging vs boosting.** Bagging: modelos independientes en paralelo sobre bootstraps, reduce varianza (RF). Boosting: secuencial, cada modelo corrige al anterior, reduce sesgo (XGBoost, LightGBM, CatBoost).
4. **¿Por qué Random Forest usa un subconjunto de features por split?** Para decorrelacionar los árboles y que el promedio reduzca más la varianza.
5. **¿Qué hace `eta` en XGBoost?** Escala la contribución de cada árbol nuevo; valores chicos necesitan más rondas pero suelen generalizar mejor.
6. **¿Cómo evitás overfitting en gradient boosting?** Early stopping con validación, `eta` chico, `max_depth` bajo, `min_child_weight`, subsampling y regularización L1/L2.
7. **Limitaciones de feature importance basada en impureza.** Sesgada hacia features con muchos valores únicos; alternativa: permutation importance o SHAP.
8. **¿Cuándo preferirías un modelo lineal sobre XGBoost?** Necesidad de interpretabilidad simple, pocos datos, relaciones lineales, o restricciones de latencia/tamaño.

## Recursos

- Lecciones: `machine-learning-zoomcamp/06-trees/`
- Código de la tarea: `hw6_questions.py`
