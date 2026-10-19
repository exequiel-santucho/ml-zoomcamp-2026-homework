# Módulo 4: Métricas de Evaluación para Clasificación - Resumen de Conceptos Clave

## Introducción

En el Módulo 3 el modelo de churn dio **80% de accuracy**. ¿Es bueno? Este módulo enseña a responder esa pregunta con rigor. Vemos por qué la accuracy engaña, cómo usar la **matriz de confusión**, **precision/recall**, **F1**, **curva ROC**, **AUC** y **validación cruzada (K-Fold)** para estimar el rendimiento y ajustar hiperparámetros.

> **Métrica** = un único número que describe el rendimiento de un modelo.

---

## 1. Accuracy y el modelo "dummy"

$$ Accuracy = \frac{\text{predicciones correctas}}{\text{total}} $$

```python
thresholds = np.linspace(0, 1, 21)
for t in thresholds:
    acc = ((y_pred >= t) == y_val).mean()
```

- El umbral 0.5 suele ser bueno, pero **no siempre el óptimo**.
- **Modelo dummy**: predecir "nadie churnea" (umbral = 1.0) ya da **73%** de accuracy, porque el 73% de los clientes no se va.
- Nuestro modelo (80%) mejora apenas 7 puntos sobre "no hacer nada".

**Conclusión:** con **clases desbalanceadas**, la accuracy es engañosa. Necesitamos métricas que miren cada tipo de error.

---

## 2. Matriz de confusión (confusion table)

|  | **Predicho: No churn (0)** | **Predicho: Churn (1)** |
|---|---|---|
| **Real: No churn (0)** | TN (true negative) | FP (false positive) |
| **Real: Churn (1)** | FN (false negative) | TP (true positive) |

```python
actual_positive = (y_val == 1)
predict_positive = (y_pred >= 0.5)
tp = (predict_positive & actual_positive).sum()
tn = (~predict_positive & ~actual_positive).sum()
fp = (predict_positive & ~actual_positive).sum()
fn = (~predict_positive & actual_positive).sum()
```

**Costo de cada error en el negocio (churn):**
- **FP**: le damos descuento a alguien que no se iba → perdemos plata.
- **FN**: no detectamos a alguien que se va → perdemos el cliente.

> Elegir el umbral es una **decisión de negocio**: ¿qué error es más caro?

---

## 3. Precision y Recall

$$ Precision = \frac{TP}{TP + FP} \qquad Recall = \frac{TP}{TP + FN} $$

| Métrica | Pregunta que responde | Importa cuando... |
|---|---|---|
| **Precision** | De los que predije positivos, ¿cuántos lo eran? | Los FP son caros (spam que manda a la papelera emails buenos) |
| **Recall (sensibilidad, TPR)** | De los positivos reales, ¿cuántos encontré? | Los FN son caros (diagnóstico médico, fraude) |

En el curso: precision ≈ 67%, recall ≈ 54% → se nos escapa casi la mitad de los que churnean, algo que la accuracy de 80% escondía.

### Trade-off precision/recall

- **Subir el umbral** → menos positivos predichos → ↑ precision, ↓ recall.
- **Bajar el umbral** → más positivos → ↓ precision, ↑ recall.

### F1 score

Media **armónica** de ambas (castiga si una de las dos es baja):

$$ F_1 = 2 \cdot \frac{P \cdot R}{P + R} $$

---

## 4. Curva ROC (Receiver Operating Characteristic)

Viene del análisis de radares en la Segunda Guerra Mundial. Evalúa el modelo en **todos los umbrales**.

$$ TPR = \frac{TP}{TP + FN} \;(\text{= recall}) \qquad FPR = \frac{FP}{FP + TN} $$

- **TPR** (true positive rate): queremos que sea **alto**.
- **FPR** (false positive rate): queremos que sea **bajo**.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")   # modelo aleatorio
```

| Modelo | Curva ROC |
|---|---|
| **Aleatorio** | Diagonal (TPR = FPR) |
| **Ideal** | Sube recto a (0, 1): TPR=1 con FPR=0 |
| **Nuestro** | Entre ambos; cuanto más pegado a la esquina superior izquierda, mejor |

Ventaja: **funciona bien con clases desbalanceadas**.

---

## 5. ROC AUC (Area Under the Curve)

El área bajo la curva ROC resume la curva en un número.

| AUC | Significado |
|---|---|
| 0.5 | Aleatorio |
| 0.7–0.8 | Aceptable / bueno |
| 0.8–0.9 | Muy bueno |
| 1.0 | Perfecto (sospechar data leakage) |
| < 0.5 | Peor que el azar → **invertir** las predicciones |

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred)
```

### Interpretación probabilística (¡pregunta clásica de entrevista!)

> **AUC = probabilidad de que un positivo elegido al azar tenga un score mayor que un negativo elegido al azar.**

```python
pos = y_pred[y_val == 1]; neg = y_pred[y_val == 0]
n = 100_000
(np.random.choice(pos, n) > np.random.choice(neg, n)).mean()   # ≈ AUC
```

Por eso AUC mide la **capacidad de ranking**, independiente del umbral.

**Uso extra (tarea Q1):** se puede usar una feature numérica directamente como "score" y calcular su AUC → mide qué tan bien esa feature sola separa las clases (feature importance). Si da < 0.5, la relación es negativa: se invierte con `-df[col]`.

---

## 6. Validación cruzada (K-Fold Cross-Validation)

Un único split de validación puede tener **suerte o mala suerte**. K-Fold usa todos los datos de `full_train` para validar:

```
Fold 1: [VAL] [ T ] [ T ] [ T ] [ T ]
Fold 2: [ T ] [VAL] [ T ] [ T ] [ T ]
...
Fold 5: [ T ] [ T ] [ T ] [ T ] [VAL]
→ 5 scores → media ± desvío estándar
```

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_tr, df_va = df_full_train.iloc[train_idx], df_full_train.iloc[val_idx]
    dv, model = train(df_tr, df_tr.churn.values, C=C)
    scores.append(roc_auc_score(df_va.churn.values, predict(df_va, dv, model)))
print(f"{np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

- **Media**: estimación del rendimiento.
- **Std**: estabilidad. Std alto → el modelo depende mucho de qué datos le tocan.
- **Tuning**: repetir para cada valor de `C` y elegir la mejor media (con std bajo).
- **Cuándo usarlo**: datasets chicos/medianos. Con datasets enormes, un único split suele alcanzar (y K-Fold es caro).

Tip del curso: `tqdm` para mostrar barra de progreso en los loops.

---

## 7. Resumen de métricas

| Métrica | Fórmula | Ventaja | Cuidado |
|---|---|---|---|
| Accuracy | (TP+TN)/N | Simple | Engaña con desbalance |
| Precision | TP/(TP+FP) | Controla FP | Depende del umbral |
| Recall/TPR | TP/(TP+FN) | Controla FN | Depende del umbral |
| F1 | 2PR/(P+R) | Balancea P y R | Ignora TN |
| FPR | FP/(FP+TN) | Eje X de ROC | — |
| ROC AUC | Área bajo ROC | Independiente del umbral, robusto a desbalance | No dice qué umbral usar |

---

## 8. Lo que resolvimos en la tarea (lead scoring 2026)

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | AUC de features individuales | `lead_score` (0.788) |
| Q2 | AUC del modelo | 0.732 |
| Q3 | Cruce precision/recall | umbral 0.63 |
| Q4 | F1 máximo | umbral 0.41 |
| Q5 | Std 5-Fold | 0.007 |
| Q6 | Tuning de C con CV | C = 0.001 (0.749 ± 0.010) |

**Observaciones:**
- **Q3 vs Q4:** el F1 máximo (0.41) no coincide con el cruce de P y R (0.63). El F1 favorece umbrales bajos cuando la clase positiva es mayoritaria (en este dataset convierte el 57.7%).
- **Q6:** C=0.001 (más regularización) **supera** a C=1 → refuerza el hallazgo del Módulo 3: la feature `annual_income` sin escalar hace daño, y una regularización más fuerte lo atenúa.
- Ver gráfico en `plots/precision_recall_f1.png`.

---

## Preguntas típicas de entrevista

1. **¿Por qué la accuracy no sirve con clases desbalanceadas?** Un modelo trivial que predice siempre la clase mayoritaria tiene accuracy alta sin aprender nada.
2. **Explicá precision vs recall con un ejemplo.** Detector de fraude: recall alto para no dejar pasar fraudes; filtro de spam: precision alta para no perder emails legítimos.
3. **¿Qué mide el AUC? ¿Interpretación?** Capacidad de ranking: P(score(positivo) > score(negativo)).
4. **¿Cómo elegís el umbral de decisión?** Según el costo de FP vs FN del negocio; se puede maximizar F1, fijar un recall mínimo, o optimizar una función de costo.
5. **¿Para qué sirve K-Fold? ¿Cuándo no usarlo?** Estimación más robusta y tuning con datos limitados; no con datasets enormes o series temporales (ahí se usa validación temporal).
6. **AUC de 0.99 en validación: ¿qué sospechás?** Data leakage (una feature que contiene el target o información futura).
7. **¿ROC AUC vs PR AUC?** Con desbalance extremo (ej. 1% positivos), PR AUC es más informativa porque ROC puede verse optimista.

## Recursos

- Lecciones: `machine-learning-zoomcamp/04-evaluation/`
- Código de la tarea: `hw4_questions.py`
