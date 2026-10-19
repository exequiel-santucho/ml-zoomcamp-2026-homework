# Módulo 9: Serverless Deep Learning - Resumen de Conceptos Clave

## Introducción

Tomamos el clasificador de imágenes del Módulo 8 y lo desplegamos **sin administrar servidores** con **AWS Lambda**. El desafío: los frameworks de deep learning (TensorFlow ~1.7 GB, PyTorch ~800 MB) son demasiado pesados para Lambda. La solución es convertir el modelo a un **runtime de inferencia liviano**:

- Videos: **TensorFlow Lite** (`tflite_runtime`).
- Workshop 2025-26 y tarea 2026: **ONNX Runtime** (funciona igual para modelos de PyTorch, Keras o Scikit-Learn).

```
Modelo entrenado (PyTorch/Keras) ─► export ONNX / TFLite ─► lambda_function.py ─► imagen Docker Lambda ─► ECR ─► Lambda ─► API Gateway
```

---

## 1. ¿Qué es serverless y AWS Lambda?

- Escribís una **función** (`lambda_handler(event, context)`); AWS se encarga de servidores, escalado y disponibilidad.
- **Pagás por invocación y por tiempo de cómputo** (ms × memoria), no por tiempo ocioso. Free tier generoso.
- Escala automáticamente de 0 a miles de invocaciones concurrentes.

| Ventaja | Limitación |
|---|---|
| Sin servidores que mantener | **Cold start**: la primera invocación tarda (carga del contenedor y del modelo) |
| Costo cero sin tráfico | Límite de tiempo (15 min) y memoria (10 GB) |
| Escalado automático | Imagen ≤ 10 GB, paquete zip ≤ 250 MB |
| Ideal para tráfico bajo/irregular | Sin GPU |

```python
def lambda_handler(event, context):
    url = event["url"]
    return {"prediction": predict(url)}
```

- **`event`**: el input (JSON). Si viene de API Gateway, el JSON real está en `event["body"]` como string.
- **`context`**: metadata de la invocación (tiempo restante, request id).

---

## 2. Runtimes de inferencia livianos

### TensorFlow Lite (videos)

```python
# Conversión (una vez, con TF completo)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
open("clothing-model.tflite", "wb").write(converter.convert())

# Inferencia (solo tflite_runtime, liviano)
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path="clothing-model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
```

### ONNX (Open Neural Network Exchange) — tarea 2026

Formato **abierto e independiente del framework**. Se exporta desde PyTorch (`torch.onnx.export`), Keras (`tf2onnx`) o Scikit-Learn (`skl2onnx`) y se ejecuta con **ONNX Runtime**.

```python
import onnxruntime as ort

session = ort.InferenceSession("hair_classifier_v1.onnx", providers=["CPUExecutionProvider"])
session.get_inputs()[0].name, session.get_inputs()[0].shape    # 'input', ['batch', 3, 200, 200]
session.get_outputs()[0].name                                  # 'output'
output = session.run(["output"], {"input": X})[0]
```

- Modelos grandes guardan los pesos en un archivo **external data** (`.onnx.data`), que tiene que estar **al lado** del `.onnx` con el nombre exacto.
- `providers`: CPU, CUDA, TensorRT... En Lambda, CPU.

---

## 3. Preprocesamiento sin el framework

En producción no tenemos `torchvision.transforms` ni `keras.preprocessing`: hay que reproducirlo **con PIL + NumPy** exactamente igual que en training.

```python
from PIL import Image
import numpy as np

def prepare_image(image):
    image = image.resize((200, 200), Image.Resampling.BILINEAR)   # misma interpolación
    array = np.asarray(image, dtype=np.float32) / 255.0             # [0, 1]
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
            np.array([0.229, 0.224, 0.225], dtype=np.float32)       # normalización ImageNet
    return np.transpose(array, (2, 0, 1))[None, ...]                # HWC → CHW + batch
```

Checklist del "contrato" de entrada: **RGB** (`.convert("RGB")` por si llega RGBA/escala de grises), tamaño, **interpolación**, escala, normalización, **orden de canales** (CHW vs HWC) y `float32`.

> En 2025 una tarea usó interpolación implícita y la siguiente `NEAREST`: las predicciones no coincidían. Un detalle de preprocesamiento rompe un modelo perfecto.

La librería `keras-image-helper` del autor del curso encapsula esto para Keras (`create_preprocessor("xception", target_size=(299, 299))`).

---

## 4. Docker para Lambda

AWS provee **imágenes base** con el runtime de Lambda incluido (y el *Runtime Interface Emulator* para probar local):

```dockerfile
FROM public.ecr.aws/lambda/python:3.13

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt \
    --target ${LAMBDA_TASK_ROOT}

COPY lambda_function.py hair_classifier_v1.onnx hair_classifier_v1.onnx.data ${LAMBDA_TASK_ROOT}/
CMD ["lambda_function.lambda_handler"]      # módulo.función
```

```bash
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
     -d '{"image_url": "https://..."}'
```

- Probar **local** antes de subir evita sorpresas en AWS.
- Cargar el modelo **fuera** del handler (a nivel de módulo): se carga una vez por contenedor y se reutiliza entre invocaciones "calientes".
- Las wheels deben ser compatibles con **Amazon Linux** (por eso se instalan dentro de la imagen base).

---

## 5. Deploy en AWS

```bash
# 1. Crear repositorio en ECR (Elastic Container Registry)
aws ecr create-repository --repository-name clothing-tflite-images
# 2. Login, tag y push
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker tag mlzoomcamp-2026-serverless:latest $REMOTE_URI
docker push $REMOTE_URI
```

3. Crear la función Lambda desde la imagen (**container image**). Usar el **digest** inmutable de la imagen.
4. Ajustar **memoria** (ej. 1024 MB; más memoria = más CPU) y **timeout** (ej. 30 s; el default de 3 s no alcanza para el cold start).
5. Probar con un evento de test.

---

## 6. API Gateway

Expone la Lambda como un **endpoint HTTP REST**:

1. Crear API REST → recurso `/predict` → método `POST` → integración con la Lambda.
2. **Deploy** a un *stage* (ej. `test`) → URL pública `https://xxx.execute-api.region.amazonaws.com/test/predict`.
3. El body llega a la Lambda como **string** en `event["body"]` → por eso el handler hace `json.loads(event["body"])` y devuelve `{"statusCode": 200, "body": json.dumps(result)}`.

⚠️ Sin autenticación la API queda **abierta a cualquiera** (y al costo de cualquiera). En producción: API keys, IAM, Cognito, throttling.

---

## 7. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Nodo de salida del grafo ONNX | `output` |
| Q2 | Tamaño de entrada | 200×200 |
| Q3 | Primer valor del canal R normalizado | −1.056 |
| Q4 | Inferencia local con ONNX Runtime | 0.728 |
| Q5 | Imagen base del Dockerfile | `public.ecr.aws/lambda/python:3.13` |
| Q6 | Invocación del contenedor Lambda | 0.728 |

**Verificación de Q3 a mano:** el primer píxel tiene R ≈ 0.243 tras dividir por 255 → (0.243 − 0.485) / 0.229 ≈ −1.056.

---

## Preguntas típicas de entrevista

1. **¿Cuándo usarías serverless y cuándo no?** Sí: tráfico bajo o irregular, sin GPU, latencia tolerante a cold starts. No: tráfico alto constante (sale más caro), modelos enormes, GPU o latencia estricta.
2. **¿Qué es un cold start y cómo lo mitigás?** La latencia de inicializar el contenedor y cargar el modelo. Mitigación: imágenes livianas, cargar el modelo fuera del handler, provisioned concurrency.
3. **¿Por qué ONNX?** Formato portable entre frameworks, runtime liviano y optimizado, permite entrenar en PyTorch y servir en cualquier lenguaje/plataforma.
4. **¿Qué es training-serving skew en visión?** Diferencias de preprocesamiento (interpolación, normalización, orden de canales) entre entrenamiento y producción.
5. **Diferencia entre Lambda con zip y con imagen de contenedor.** Zip: hasta 250 MB descomprimido; contenedor: hasta 10 GB y control total del entorno.
6. **¿Cómo exponés una Lambda a internet?** API Gateway (o Lambda Function URLs), idealmente con autenticación y rate limiting.

## Recursos

- Lecciones: `machine-learning-zoomcamp/09-serverless/` (y `workshop/lambda-onnx`)
- Código de la tarea: esta carpeta
