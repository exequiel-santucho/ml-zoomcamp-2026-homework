## Homework 8: Redes Neuronales y Deep Learning

> CNN chica en PyTorch para clasificar pelo **rizado (curly=0)** vs **lacio (straight=1)**. Dataset, versiones, arquitectura, split y semilla están fijados.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/08-deep-learning/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw08 — **vence el 23/11/2026, 23:00 UTC**

## Dataset

```bash
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
sha256sum data.zip   # 9e53453e...7d67 ✓
unzip -q data.zip
```

| Split | Curly | Straight | Total |
|---|---:|---:|---:|
| Train | 410 | 390 | 800 |
| Evaluación | 103 | 98 | 201 |

## Entorno

Python 3.11, `torch==2.9.0+cpu`, `torchvision==0.24.0+cpu`, `numpy==2.3.3`, `Pillow==11.3.0` (ver `requirements-cpu.txt`). Las respuestas numéricas se calculan en **CPU**.

## Reglas de reproducibilidad

Semilla 42 en `random`, `numpy` y `torch`; `torch.use_deterministic_algorithms(True)`; cuDNN determinístico; `num_workers=0`; un `torch.Generator` con semilla propia para el loader de train.

## Modelo

`Conv2d(3, 32, 3)` → ReLU → `MaxPool2d(2)` → flatten → `Linear(32*99*99, 64)` → ReLU → `Linear(64, 1)` (devuelve **logits**).
Optimizador: `SGD(lr=0.002, momentum=0.8)`.

## Pregunta 1 — pérdida

¿Qué pérdida corresponde al modelo?

- `nn.MSELoss()`
- **`nn.BCEWithLogitsLoss()`** ✓ (1 logit; combina sigmoid + binary cross-entropy de forma estable)
- `nn.CrossEntropyLoss()`
- `nn.CosineEmbeddingLoss()`

## Pregunta 2 — cantidad de parámetros

- `896`
- `11214912`
- `15896912`
- **`20073473`** ✓ (896 conv + 20.072.512 densa oculta + 65 salida)

## Métricas de entrenamiento

10 épocas con el transform base. En cada época, evaluar el split de evaluación con `model.eval()` y `torch.no_grad()`. Guardar `history_baseline.json` con `train_loss`, `train_accuracy`, `evaluation_loss`, `evaluation_accuracy`.

## Pregunta 3 — accuracy baseline

Mediana de `train_accuracy` (2 decimales, ±0.05):

**Respuesta: `0.81`** ✓ (0.808125)

## Pregunta 4 — loss baseline

Desvío estándar poblacional (`np.std(..., ddof=0)`) de `train_loss` (3 decimales, ±0.02):

**Respuesta: `0.139`** ✓ (0.13934)

## Data augmentation

Mismo modelo y optimizador (¡sin recrearlos!). Solo cambia el transform de train: `RandomRotation(50)`, `RandomResizedCrop(200, scale=(0.9,1.0), ratio=(0.9,1.1))`, `RandomHorizontalFlip()`. 10 épocas más → `history_augmented.json`.

## Pregunta 5 — eval loss con augmentation

Media de `evaluation_loss` en `history_augmented.json` (3 decimales, ±0.02):

**Respuesta: `0.576`** ✓ (0.57583)

## Pregunta 6 — eval accuracy con augmentation

Media de las últimas 5 `evaluation_accuracy` (épocas 6-10 de la fase aumentada; 2 decimales, ±0.05):

**Respuesta: `0.72`** ✓ (0.72040)

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | nn.BCEWithLogitsLoss() |
| Q2 | 20073473 |
| Q3 | 0.81 |
| Q4 | 0.139 |
| Q5 | 0.576 |
| Q6 | 0.72 |

> Validado: mi `hw8_questions.py` y el `reference_train.py` oficial dan exactamente los mismos números.
