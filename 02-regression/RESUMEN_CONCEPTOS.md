# Módulo 2: Machine Learning para Regresión - Resumen de Conceptos Clave

## Introducción

Todo el módulo es **un proyecto completo**: predecir el precio (MSRP) de autos. La gracia es que **implementamos la regresión lineal desde cero con NumPy**, sin Scikit-Learn, para entender qué pasa adentro. En el camino aparecen ideas que vamos a usar todo el curso: EDA, transformación logarítmica, framework de validación, RMSE, feature engineering, one-hot encoding y regularización.

---

## 1. Preparación de datos

Primero, **normalizar** nombres de columnas y valores de texto para que todo sea uniforme:

```python
df.columns = df.columns.str.lower().str.replace(" ", "_")
strings = list(df.dtypes[df.dtypes == "object"].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ", "_")
```

Así evitamos bugs del tipo `"Engine HP"` vs `"engine_hp"` o `"BMW"` vs `"bmw"`.

---

## 2. Análisis exploratorio (EDA)

Preguntas que siempre hay que hacerse:
- ¿Qué columnas hay y de qué tipo? (`df.dtypes`)
- ¿Cuántos valores únicos tiene cada una? (`nunique`)
- ¿Hay faltantes? (`isnull().sum()`)
- ¿Cómo se distribuye el target? (histograma)

### Distribución del target y cola larga (long tail)

Los precios tienen **cola larga**: muchos autos baratos y pocos carísimos. Los modelos lineales sufren con esto porque los valores extremos dominan el error.

**Solución: transformación logarítmica.**

```python
y = np.log1p(df.msrp)      # log(1 + x): evita log(0)
y_pred_real = np.expm1(y_pred)  # para volver a la escala original
```

Después del log, la distribución se parece a una campana (normal) → el modelo aprende mejor.

> En la tarea 2026, `fuel_efficiency_mpg` **no** tiene cola larga (skewness ≈ 0.08), así que no hizo falta el log. Saber **cuándo no** aplicar una transformación también es parte del EDA.

---

## 3. Framework de validación

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)          # ¡mezclar! los datos pueden venir ordenados

df_train = df.iloc[idx[:n_train]]
df_val   = df.iloc[idx[n_train:n_train + n_val]]
df_test  = df.iloc[idx[n_train + n_val:]]

y_train = np.log1p(df_train.msrp.values)
del df_train["msrp"]            # ¡sacar el target de las features!
```

**Errores clásicos que esto previene:**
- No mezclar → train y test con distribuciones distintas (ej. ordenados por marca).
- Dejar el target dentro de X → **data leakage** (el modelo "ve" la respuesta).

---

## 4. Regresión lineal

### Para una observación

$$ g(x_i) = w_0 + w_1 x_{i1} + w_2 x_{i2} + \dots + w_n x_{in} = w_0 + \sum_{j} w_j x_{ij} $$

- **w₀ (bias / intercepto)**: la predicción "base" si no sabemos nada del auto.
- **wⱼ (pesos)**: cuánto cambia la predicción por cada unidad de la feature j.

### Forma vectorial

Agregamos una columna de **1s** a X para que el bias sea un peso más:

$$ g(x_i) = x_i^T w \qquad \Rightarrow \qquad g(X) = Xw $$

```python
def linear_regression(X, w):
    return X.dot(w)
```

### Entrenamiento: la ecuación normal

Queremos `Xw ≈ y`. Como X no es cuadrada, no se puede invertir; multiplicamos por Xᵀ:

$$ X^T X w = X^T y \quad\Rightarrow\quad w = (X^T X)^{-1} X^T y $$

- **XᵀX** es la **matriz de Gram**: cuadrada (n_features × n_features) y, en general, invertible.
- Es la solución **exacta** de mínimos cuadrados (minimiza el error cuadrático).

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]   # bias, pesos
```

> "Machine learning no es magia: es una fórmula."

---

## 5. Baseline y RMSE

**Baseline**: el primer modelo simple (solo features numéricas, faltantes en 0). Sirve como **punto de comparación**: cualquier mejora tiene que superarlo.

**RMSE (Root Mean Squared Error):**

$$ RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (g(x_i) - y_i)^2} $$

```python
def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())
```

- Está en las **mismas unidades** que el target (MPG, o log-dólares).
- Penaliza mucho los errores grandes (por el cuadrado).
- Menor es mejor.

**Función `prepare_X`**: centraliza cómo construimos X para train, val y test. Garantiza que **todas las particiones se procesan igual** y permite iterar rápido.

---

## 6. Valores faltantes (imputación)

| Estrategia | Pro | Contra |
|---|---|---|
| **0** | Simple | Para muchas variables 0 no tiene sentido físico (un auto con 0 HP) |
| **Media / mediana del train** | Valor "plausible" | Hay que calcularla **solo con train** |
| **Moda** | Sirve para categóricas | Puede sesgar la distribución |

**Regla de oro:** las estadísticas de imputación (media, mediana) se calculan **solo con train** y se aplican igual a val/test. Si usás todo el dataset, filtrás información del futuro → **data leakage**.

En la tarea: rellenar `horsepower` con la media dio RMSE 2.202 vs 2.205 con 0 → **la media es (apenas) mejor**.

---

## 7. Feature engineering

Crear features nuevas a partir de las existentes. Ejemplo del curso:

```python
df["age"] = 2017 - df.year   # la edad del auto explica mejor el precio que el año
```

Mejoró mucho el modelo: la distribución de las predicciones empezó a parecerse a la real.

---

## 8. Variables categóricas: one-hot encoding

Los modelos lineales solo entienden números. Una categórica (ej. `make`) se convierte en varias columnas binarias:

| make | make_bmw | make_toyota | make_ford |
|---|---|---|---|
| bmw | 1 | 0 | 0 |
| ford | 0 | 0 | 1 |

```python
for v in ["chevrolet", "ford", "volkswagen", "toyota", "dodge"]:
    df[f"make_{v}"] = (df.make == v).astype(int)
```

Se suele quedarse con las **top-N** categorías más frecuentes para no generar miles de columnas.

---

## 9. Regularización

Al agregar muchas categóricas, el RMSE **explotó**. Causa: **columnas casi duplicadas** (colinealidad) → XᵀX es casi singular → la inversa tiene números enormes → pesos gigantes e inestables.

**Solución (Ridge / regularización L2):** sumar un número pequeño `r` a la diagonal:

$$ w = (X^T X + r I)^{-1} X^T y $$

```python
XTX = X.T.dot(X)
XTX = XTX + r * np.eye(XTX.shape[0])
```

- Hace que la matriz sea invertible y "controla" el tamaño de los pesos.
- **r grande** → pesos más chicos → modelo más simple (puede **underfit**).
- **r = 0** → sin regularización (puede **overfit** o ser inestable).

---

## 10. Tuning del modelo

Probar varios valores de `r` y quedarse con el mejor en **validación**:

```python
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    score = rmse(y_val, w0 + X_val.dot(w))
```

En la tarea, con solo 4 features numéricas (sin colinealidad), **r = 0** fue el mejor: la regularización no siempre ayuda; ayuda cuando hay inestabilidad.

---

## 11. Usar el modelo (modelo final)

1. Juntar **train + validation** (más datos = mejor modelo).
2. Entrenar con el `r` elegido.
3. Evaluar **una vez** en test (RMSE parecido al de validación → el modelo generaliza).
4. Para predecir un auto nuevo: armar su X con `prepare_X`, predecir y aplicar `np.expm1` si usamos log.

---

## 12. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Faltantes | `horsepower` |
| Q2 | Mediana | 254 |
| Q3 | Imputación 0 vs media | Media (2.202 < 2.205) |
| Q4 | Regularización | r = 0 |
| Q5 | Estabilidad vs semilla (std del RMSE) | 0.029 |
| Q6 | Modelo final (train+val, r=0.001, seed 9) | RMSE test 2.236 |

**Lección de Q5:** cambiar la semilla del split cambia el RMSE entre 2.16 y 2.27. Con std = 0.029 el modelo es **estable**: diferencias menores a eso entre modelos son ruido, no mejoras reales.

---

## Preguntas típicas de entrevista

1. **Derivá la ecuación normal.** Minimizar ‖Xw − y‖² → gradiente = 0 → XᵀXw = Xᵀy → w = (XᵀX)⁻¹Xᵀy.
2. **¿Por qué aplicar log al target?** Para reducir la cola larga, estabilizar la varianza y que los errores relativos pesen parecido.
3. **¿Qué es la multicolinealidad y cómo afecta?** Features muy correlacionadas → XᵀX casi singular → pesos inestables. Se mitiga con regularización o eliminando features.
4. **¿Ridge vs Lasso?** Ridge (L2) achica todos los pesos; Lasso (L1) puede llevarlos a cero (selección de features).
5. **¿RMSE vs MAE?** RMSE castiga más los errores grandes; MAE es más robusto a outliers.
6. **¿Qué es data leakage? Dame un ejemplo.** Usar información que no estaría disponible al predecir: imputar con la media de todo el dataset, o dejar el target en X.
7. **¿Por qué la ecuación normal no escala?** Invertir XᵀX es O(n³) en features; para muchos features/datos se usa descenso por gradiente.

## Recursos

- Lecciones: `machine-learning-zoomcamp/02-regression/`
- Código de la tarea: `hw2_questions.py`
