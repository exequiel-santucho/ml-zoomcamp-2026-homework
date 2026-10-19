# Módulo 1: Introducción a Machine Learning - Resumen de Conceptos Clave

## Introducción

Este módulo arma el "mapa mental" de todo el curso: qué es Machine Learning, cuándo conviene usarlo en lugar de reglas escritas a mano, cómo se organiza un proyecto real (CRISP-DM) y cómo elegir un modelo sin engañarse a uno mismo (train/validation/test). Además repasa las tres herramientas que vamos a usar todo el tiempo: **NumPy**, **álgebra lineal** y **Pandas**.

> Idea central: un modelo de ML es una función `g` que aprende patrones a partir de ejemplos. `g(X) ≈ y`.

---

## 1. ¿Qué es Machine Learning?

Ejemplo del curso: **predecir el precio de un auto**.

- **Features (X)**: todo lo que sabemos del auto (año, marca, kilometraje, potencia...).
- **Target (y)**: lo que queremos predecir (el precio).
- **Algoritmo de ML**: recibe muchos pares (X, y) y produce un **modelo**.
- **Modelo**: se aplica a autos nuevos (de los que no sabemos el precio) y devuelve una predicción.

```
Datos (features + target) ──► Algoritmo de entrenamiento ──► Modelo
Features de un auto nuevo ──► Modelo ──► Predicción ($23.000)
```

**Vocabulario que hay que dominar:**

| Término | Significado |
|---|---|
| Feature / atributo / variable | Columna de entrada (lo que conocemos) |
| Target / label / variable objetivo | Columna a predecir |
| Observación / registro / fila | Un ejemplo (un auto) |
| Entrenar (fit) | Ajustar el modelo a los datos |
| Predecir (predict / inference) | Usar el modelo en datos nuevos |

---

## 2. ML vs. sistemas basados en reglas

Ejemplo: **detector de spam**.

**Enfoque con reglas (programación tradicional):**
```
si remitente == "promotions@online.com"          -> spam
si título contiene "tax review" y dominio ...      -> spam
si cuerpo contiene "deposit" y ...                 -> spam
si no                                              -> ok
```
Problemas: los spammers cambian de estrategia, las reglas se multiplican, se contradicen y el código se vuelve **inmantenible**.

**Enfoque con ML:**
1. Juntar datos etiquetados (emails marcados como spam / no spam).
2. Convertir cada email en un **vector de features** (`[1, 0, 0, 1, 1, 0]`: ¿tiene "deposit"?, ¿el título es largo?, ¿el remitente es conocido?...).
3. Entrenar un modelo que **descubre los patrones por sí mismo** con estadística.
4. El modelo devuelve una **probabilidad** (ej. 0.8) y decidimos con un umbral (≥ 0.5 → spam).

> Las reglas siguen siendo útiles: suelen convertirse en **features** del modelo.

| | Reglas | ML |
|---|---|---|
| Quién encuentra el patrón | El humano | El algoritmo |
| Entrada | Datos + reglas → salida | Datos + salidas → reglas (modelo) |
| Mantenimiento | Crece sin control | Re-entrenar con datos nuevos |
| Ideal cuando | Lógica simple, estable y explicable | Patrones complejos, cambiantes, muchos datos |

---

## 3. Aprendizaje supervisado

Le "enseñamos" al modelo con ejemplos de los que conocemos la respuesta.

$$ g(X) \approx y $$

- **X**: matriz de features (filas = observaciones, columnas = features).
- **y**: vector target.
- **g**: el modelo.

**Tipos de problemas supervisados:**

| Tipo | Target | Ejemplo | Módulo |
|---|---|---|---|
| **Regresión** | Número continuo | Precio de un auto, MPG | 2 |
| **Clasificación binaria** | 2 clases | Spam / no spam, churn | 3-4 |
| **Clasificación multiclase** | >2 clases | Tipo de prenda (remera, pantalón...) | 8 |
| **Ranking** | Orden de relevancia | Recomendaciones, buscadores | — |

> El **no supervisado** (clustering, reducción de dimensionalidad) no tiene target; no es el foco de este curso.

---

## 4. CRISP-DM: la metodología de un proyecto de ML

**CR**oss **I**ndustry **S**tandard **P**rocess for **D**ata **M**ining (IBM, años 90). Sigue vigente porque describe **todo** lo que rodea al modelo.

```
1. Business understanding  ─►  2. Data understanding
        ▲                              │
        │                              ▼
6. Deployment  ◄─ 5. Evaluation ◄─ 4. Modeling ◄─ 3. Data preparation
```

| Paso | Pregunta clave | Ejemplo (spam) |
|---|---|---|
| 1. **Business understanding** | ¿Cuál es el problema? ¿ML es necesario? ¿Cómo medimos el éxito? | "Reducir 50% los reportes de spam" (meta medible) |
| 2. **Data understanding** | ¿Qué datos hay? ¿Son confiables? ¿Alcanzan? | ¿El botón "reportar spam" se usa bien? |
| 3. **Data preparation** | Limpiar, construir features, armar X e y | Extraer features del email |
| 4. **Modeling** | Probar modelos y elegir el mejor | Regresión logística vs. árboles vs. redes |
| 5. **Evaluation** | ¿Cumplimos la meta de negocio? | ¿Bajó el spam reportado? |
| 6. **Deployment** | Llevarlo a producción, monitorear | Servicio web + A/B test |

**Puntos importantes:**
- Es **iterativo**: se empieza simple, se lanza y se mejora.
- A veces la conclusión del paso 1 es "**no hace falta ML**". Saber decirlo es señal de seniority.
- Hoy los pasos 5 y 6 se mezclan: se evalúa **en producción** con experimentos online (A/B tests).

---

## 5. Selección de modelos (model selection)

Problema: si elegimos el modelo que mejor anda en los mismos datos con que lo entrenamos, elegimos al que **memorizó** (overfitting).

**Solución: dividir los datos.**

```
Dataset completo
├── Train (60%)      → entrenar los modelos
├── Validation (20%) → comparar modelos y elegir hiperparámetros
└── Test (20%)       → verificación final, UNA vez, del modelo elegido
```

**Proceso:**
1. Split en train / validation / test.
2. Entrenar varios modelos en **train**.
3. Evaluarlos en **validation**.
4. Elegir el mejor.
5. Confirmar en **test** que no fue suerte (que el resultado se mantiene).
6. (Opcional) Re-entrenar con train+validation y volver a chequear en test.

**¿Por qué el test aparte?** Al probar muchos modelos sobre validation, alguno puede ganar **por casualidad** (problema de comparaciones múltiples). El test es un dataset "virgen" que detecta eso.

---

## 6. Entorno de trabajo

El curso usa Python + Jupyter + NumPy + Pandas + Scikit-Learn (+ Matplotlib/Seaborn para gráficos). Opciones: Anaconda, GitHub Codespaces, una VM en la nube o **uv** (lo que usamos acá, más moderno y reproducible).

```bash
uv init --python 3.11
uv add pandas numpy matplotlib seaborn jupyter
uv run jupyter notebook
```

---

## 7. NumPy: cálculo numérico con arrays

NumPy es la base de todo el ecosistema (Pandas, Scikit-Learn y PyTorch están construidos sobre arrays).

```python
import numpy as np

np.zeros(5); np.ones((2, 3)); np.full(4, 2.5)  # creación
np.arange(3, 10); np.linspace(0, 1, 11)        # rangos
np.array([[1, 2], [3, 4]])                     # 2D (matriz)
a.shape, a[0, 1], a[:, 1]                      # forma e indexado

np.random.seed(2)                              # ¡reproducibilidad!
np.random.rand(5, 2)                           # uniformes [0,1)
np.random.randn(5, 2)                          # normales estándar

a + 1; a * 2; a ** 2; a + b                    # operaciones elemento a elemento
a >= 2; a[a > b]                               # comparaciones y máscaras booleanas
a.min(), a.max(), a.sum(), a.mean(), a.std()   # agregaciones
```

**Conceptos clave:**
- **Vectorización**: operar sobre todo el array sin `for` → mucho más rápido (código en C).
- **Broadcasting**: NumPy "estira" automáticamente las dimensiones compatibles (`matriz + vector`).
- **Semilla (seed)**: fija los números aleatorios para que los resultados sean reproducibles.

---

## 8. Álgebra lineal (lo justo y necesario)

**Producto vector-vector (producto punto):**

$$ u \cdot v = \sum_{i} u_i v_i $$

**Producto matriz-vector:** cada fila de la matriz hace producto punto con el vector → un vector.

**Producto matriz-matriz:** cada columna de V multiplicada por U → una matriz. Requisito: `U.shape[1] == V.shape[0]`.

```python
u.dot(v)       # o u @ v
U.dot(v)       # matriz-vector
U.dot(V)       # matriz-matriz  (o U @ V)
```

**Matriz identidad (I):** 1s en la diagonal; `U @ I = U`.

**Matriz inversa:** solo para matrices cuadradas no singulares; `A⁻¹ @ A = I`.

```python
np.eye(3)                 # identidad
np.linalg.inv(A)          # inversa
```

**Dato importante:** no hace falta memorizar fórmulas: cualquier producto matricial se puede escribir como muchos productos punto, y en código se vuelve sencillo.

---

## 9. Pandas: datos tabulares

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head(); df.shape; df.dtypes; df.describe()
df["col"]; df[["col1", "col2"]]               # seleccionar columnas
df[df.year >= 2015]                            # filtrar filas
df[(df.make == "Nissan") & (df.year >= 2015)]  # varias condiciones
df.loc[idx]; df.iloc[0]                        # por etiqueta / por posición
df["col"].str.lower().str.replace(" ", "_")   # operaciones de strings
df.isnull().sum()                              # faltantes por columna
df["col"].nunique(); df["col"].value_counts()  # categorías
df["col"].fillna(valor)                        # imputar
df.groupby("cat").price.mean()                 # agrupar (como SQL)
df.values; df.to_dict(orient="records")        # a NumPy / a lista de dicts
```

**DataFrame** = tabla (columnas con tipos distintos). **Series** = una columna.

---

## 10. Lo que resolvimos en la tarea (dataset de eficiencia de combustible 2026)

| Pregunta | Concepto practicado | Resultado |
|---|---|---|
| Q1 | Versión de librería | pandas `2.3.2` |
| Q2 | `len(df)` | 10.000 registros |
| Q3 | `nunique()` | 3 tipos de combustible |
| Q4 | `isnull().sum()` | 2 columnas con NA (`horsepower`, `acceleration`) |
| Q5 | Filtrado + `max()` | 41.2 MPG (Asia) |
| Q6 | Mediana, moda, `fillna` | La mediana **bajó** (254 → 252) |
| Q7 | Ecuación normal a mano | Σw ≈ 0.369 |

**Lección de Q6:** imputar con la **moda** (252) desplaza la distribución hacia ese valor; como la moda es menor que la mediana, la mediana baja. La imputación **no es neutral**: cambia las estadísticas de los datos.

**Lección de Q7:** `w = (XᵀX)⁻¹ Xᵀ y` es exactamente la **regresión lineal** que construimos en el Módulo 2.

---

## Preguntas típicas de entrevista

1. **¿Cuál es la diferencia entre ML y programación tradicional?** En programación tradicional escribimos reglas; en ML el algoritmo aprende las reglas a partir de datos y respuestas.
2. **¿Para qué sirven los conjuntos de validación y test?** Validation para elegir modelo/hiperparámetros; test para estimar sin sesgo el desempeño final.
3. **¿Qué es overfitting?** Cuando el modelo memoriza el train y generaliza mal: alta performance en train, baja en validación.
4. **Explicá CRISP-DM.** Seis fases iterativas: negocio → datos → preparación → modelado → evaluación → deployment.
5. **¿Cuándo NO usarías ML?** Cuando reglas simples resuelven el problema, no hay datos suficientes o se necesita explicabilidad total y la lógica es estable.
6. **¿Por qué fijar una semilla aleatoria?** Para que los experimentos sean reproducibles y comparables.

## Recursos

- Lecciones: `machine-learning-zoomcamp/01-intro/`
- Código de la tarea: `hw1_questions.py`
