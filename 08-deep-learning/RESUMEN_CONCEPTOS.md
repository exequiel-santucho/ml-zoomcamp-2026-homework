# Módulo 8: Redes Neuronales y Deep Learning - Resumen de Conceptos Clave

## Introducción

Proyecto del módulo: **clasificar fotos de ropa** en 10 clases (remera, pantalón, zapatos, vestido...). Pasamos de datos tabulares a **imágenes**, donde los modelos clásicos no alcanzan y entran las **redes neuronales convolucionales (CNN)** y el **transfer learning**.

- Videos: TensorFlow/Keras con **Xception** pre-entrenado en ImageNet.
- Workshop 2025-26 y tarea: **PyTorch** con **MobileNetV2** (y exportación a ONNX, que se usa en el Módulo 9).

La tarea 2026 entrena una CNN chica desde cero para **pelo rizado vs lacio** (clasificación binaria) con reglas estrictas de reproducibilidad.

---

## 1. Imágenes como datos

- Una imagen a color es un **tensor** de alto × ancho × 3 canales (RGB), valores 0-255.
- Keras usa formato **HWC** (`(150, 150, 3)`); PyTorch usa **CHW** (`(3, 200, 200)`) y agrega la dimensión de **batch** al frente: `(N, 3, 200, 200)`.
- **Preprocesamiento**: redimensionar a un tamaño fijo, pasar a float en [0, 1] y **normalizar** con la media y desvío de ImageNet:

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

> El preprocesamiento es parte del **contrato** del modelo: si en producción redimensionás con otra interpolación (NEAREST vs BILINEAR), las predicciones cambian. La cohorte 2026 lo fija explícitamente.

---

## 2. Modelos pre-entrenados

Redes entrenadas en **ImageNet** (14M imágenes, 1000 clases): Xception, ResNet, MobileNet, EfficientNet...

```python
# Keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
model = Xception(weights="imagenet", input_shape=(299, 299, 3))

# PyTorch
from torchvision import models
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
```

Sirven "tal cual" para clases genéricas, pero ImageNet no tiene "remera" o "pelo rizado" como clases → necesitamos adaptarlas.

---

## 3. Redes neuronales convolucionales (CNN)

```
Imagen ──► [Capas convolucionales] ──► vector de features ──► [Capas densas] ──► predicción
           (extraen patrones)                                 (deciden)
```

### Capa convolucional

- Aplica **filtros** (kernels, ej. 3×3) que se deslizan sobre la imagen.
- Cada filtro detecta un patrón (bordes, texturas; en capas profundas, formas complejas como ojos o mangas).
- Salida: **feature maps** (uno por filtro).
- Parámetros de un `Conv2d(3, 32, kernel_size=3)`: 32 filtros × (3×3×3) + 32 bias = **896**.

### Pooling

- **MaxPool 2×2**: se queda con el máximo de cada bloque → reduce a la mitad alto y ancho.
- Menos cómputo y cierta invariancia a pequeños desplazamientos.

### Capas densas (fully connected)

- Se "aplana" (`flatten`) el último feature map a un vector y se aplican capas `Linear`/`Dense`.
- Concentran la mayoría de los parámetros.

**Ejemplo de la tarea (calcular formas es pregunta de entrevista):**

```
Entrada        3 × 200 × 200
Conv 3×3       32 × 198 × 198      (200 − 3 + 1 = 198)   → 896 parámetros
MaxPool 2      32 × 99 × 99
Flatten        313.632
Linear → 64    64                                          → 313.632×64 + 64 = 20.072.512
Linear → 1     1 (logit)                                   → 64 + 1 = 65
Total                                                        20.073.473
```

> El 99.99% de los parámetros está en la primera capa densa. Por eso las arquitecturas modernas usan *global average pooling* en lugar de aplanar mapas grandes.

### Funciones de activación

| Activación | Fórmula | Uso |
|---|---|---|
| **ReLU** | max(0, x) | Capas ocultas (no linealidad barata) |
| **Sigmoid** | 1/(1+e⁻ˣ) | Salida binaria → probabilidad |
| **Softmax** | eˣⁱ/Σeˣʲ | Salida multiclase → probabilidades que suman 1 |

---

## 4. Funciones de pérdida y logits

| Problema | Salida del modelo | Pérdida (PyTorch) | Pérdida (Keras) |
|---|---|---|---|
| Binario | 1 logit | `BCEWithLogitsLoss` | `BinaryCrossentropy(from_logits=True)` |
| Multiclase | K logits | `CrossEntropyLoss` | `CategoricalCrossentropy(from_logits=True)` |
| Regresión | 1 valor | `MSELoss` | `MeanSquaredError` |

**Logit** = score crudo antes de sigmoid/softmax. Es mejor que la red devuelva logits y que la pérdida aplique sigmoid internamente: es **numéricamente más estable**. Aplicar sigmoid dos veces (en el modelo y en la pérdida) fue un bug real de la tarea 2025.

---

## 5. Transfer learning

Reusar la parte convolucional de un modelo pre-entrenado (el **base model**, que ya sabe extraer features visuales) y entrenar solo una **cabeza** nueva para nuestras clases.

```python
# Keras
base_model = Xception(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False                       # congelar
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)      # mapa → vector
outputs = keras.layers.Dense(10)(x)               # 10 clases (logits)
model = keras.Model(inputs, outputs)
```

```python
# PyTorch
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad = False                       # congelar
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 10))
```

**Ventajas:** muchos menos datos y tiempo; resultados mucho mejores que entrenar desde cero.

---

## 6. Loop de entrenamiento en PyTorch

```python
for epoch in range(num_epochs):
    model.train()                                  # activa dropout / batchnorm en modo train
    for images, labels in train_loader:
        optimizer.zero_grad()                      # 1. limpiar gradientes
        logits = model(images)                     # 2. forward
        loss = criterion(logits, labels)           # 3. pérdida
        loss.backward()                            # 4. backprop (gradientes)
        optimizer.step()                           # 5. actualizar pesos

    model.eval()                                   # desactiva dropout
    with torch.no_grad():                          # sin gradientes → más rápido
        ...evaluar en validación...
```

Vocabulario:
- **Epoch**: una pasada completa por el dataset de entrenamiento.
- **Batch**: subconjunto de ejemplos procesados juntos (ej. 20 o 32).
- **Optimizer**: SGD (con *momentum*) o Adam; actualiza pesos usando los gradientes.

---

## 7. Learning rate

El tamaño del paso del optimizador. Analogía del curso: leer un libro rápido o despacio.

- **Muy alto** → la pérdida oscila o diverge.
- **Muy bajo** → aprende muy lento o se estanca.
- Se tunea probando valores en escala logarítmica: `[0.0001, 0.001, 0.01, 0.1]` y graficando accuracy de validación por época.

> Tip del curso: experimentar con **imágenes chicas** (150×150) para iterar rápido y recién al final entrenar el modelo grande (299×299).

---

## 8. Checkpointing

Guardar el modelo **cada vez que mejora** la métrica de validación (no el de la última época, que puede estar sobreajustado).

```python
# Keras
checkpoint = keras.callbacks.ModelCheckpoint(
    "xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
    save_best_only=True, monitor="val_accuracy", mode="max")
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])

# PyTorch
if val_acc > best_acc:
    best_acc = val_acc
    torch.save(model.state_dict(), "best_model.pth")
```

---

## 9. Más capas, dropout y regularización

- **Capa densa interna extra** (ej. 100 neuronas + ReLU) entre el vector de features y la salida → más capacidad.
- **Dropout**: durante el entrenamiento "apaga" aleatoriamente una fracción de neuronas (ej. 20-50%). Obliga a la red a no depender de features puntuales → **reduce overfitting**. En evaluación se desactiva (`model.eval()`).
- Se tunea `droprate` como cualquier hiperparámetro; con dropout conviene entrenar más épocas.

---

## 10. Data augmentation

Generar variaciones de las imágenes de **entrenamiento** en cada época: rotaciones, recortes, flips, zoom, cambios de brillo.

```python
train_transform_augmented = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- El modelo ve "más datos" sin recolectarlos → generaliza mejor.
- **Nunca** aumentar validación/test: tienen que representar los datos reales.
- Elegir augmentations que tengan sentido para el dominio (un flip vertical de una remera no es realista).
- Efecto típico: la **accuracy de train baja** (la tarea es más difícil) y la de **validación se estabiliza o sube**.

---

## 11. Reproducibilidad en deep learning (foco de la tarea 2026)

```python
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DataLoader(..., num_workers=0, generator=torch.Generator().manual_seed(42))
```

Fuentes de no determinismo: inicialización de pesos, orden de los batches, augmentations aleatorias, workers en paralelo, algoritmos de cuDNN en GPU. Por eso la tarea usa CPU, una semilla por loader y `num_workers=0`.

---

## 12. Usar el modelo

```python
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
x = preprocess(Image.open("pantalon.jpg")).unsqueeze(0)   # agrega dimensión batch
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)
dict(zip(classes, probs[0].tolist()))
```

Exportar a **ONNX** para servir sin PyTorch (más liviano, Módulo 9):

```python
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})
```

---

## 13. Lo que resolvimos en la tarea (pelo rizado vs lacio, CPU)

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Pérdida correcta para 1 logit | `nn.BCEWithLogitsLoss()` |
| Q2 | Conteo de parámetros | 20.073.473 |
| Q3 | Mediana train accuracy (baseline) | 0.81 |
| Q4 | Std train loss (baseline) | 0.139 |
| Q5 | Media eval loss (augmentation) | 0.576 |
| Q6 | Media últimas 5 eval accuracy (augm.) | 0.72 |

**Qué se observa en los logs (`hw8_run.log`):**
- Baseline: la train accuracy sube hasta 0.94 mientras la eval loss **sube** (0.60 en la época 1 → 1.02 en la época 10) y la eval accuracy cae a 0.64 → **overfitting** clásico.
- Verificado: `reference_train.py` (script oficial) da exactamente los mismos valores (`reference_run.log`).
- Con augmentation: train accuracy baja a ~0.70 (la tarea es más difícil), pero la eval loss baja a ~0.55 y la eval accuracy se estabiliza en ~0.72-0.73 → la augmentation actúa como **regularización**.

---

## Preguntas típicas de entrevista

1. **¿Qué es una convolución y por qué sirve para imágenes?** Un filtro que se desliza y comparte pesos en toda la imagen: detecta el mismo patrón en cualquier posición con pocos parámetros.
2. **Calculá la forma de salida de una conv.** `(W − K + 2P)/S + 1`. Con W=200, K=3, P=0, S=1 → 198.
3. **¿Qué es transfer learning y cuándo usarlo?** Reusar un modelo entrenado en un dataset grande y adaptarlo; ideal con pocos datos y dominios parecidos.
4. **¿Por qué devolver logits y usar `BCEWithLogitsLoss`?** Estabilidad numérica (log-sum-exp) y evita aplicar sigmoid dos veces.
5. **¿Cómo combatís overfitting en redes?** Más datos/augmentation, dropout, weight decay, early stopping, modelos más chicos, transfer learning.
6. **¿Qué hacen `model.eval()` y `torch.no_grad()`?** `eval` cambia el comportamiento de dropout/batchnorm; `no_grad` desactiva el cálculo de gradientes (menos memoria, más rápido).
7. **¿Learning rate muy alto vs muy bajo?** Alto: diverge u oscila; bajo: converge muy lento o queda atascado.
8. **¿Por qué normalizar con la media/std de ImageNet?** Porque el modelo pre-entrenado aprendió con esas estadísticas de entrada.

## Recursos

- Lecciones: `machine-learning-zoomcamp/08-deep-learning/` (y `pytorch/` para el workshop)
- Código de la tarea: `hw8_questions.py`
