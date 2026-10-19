## Homework 10: Kubernetes y Model Serving

> Desplegar la misma API de lead scoring del Homework 5 en un cluster local creado con `kind`.
> Enunciado original: `machine-learning-zoomcamp/cohorts/2026/homework/10-kubernetes/homework.md`
> Entrega: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw10 — **vence el 07/12/2026, 23:00 UTC**

## Construir la imagen 2026

```bash
cd ../05-deployment
docker build -t zoomcamp-model:2026-hw10 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw10
python q6_test.py
```

## Pregunta 1 — esquema del contenedor local

`conversion_probability` (3 decimales, ±0.005):

**Respuesta: `0.770`** ✓ (`{'conversion_probability': 0.769799, 'conversion': True}`)

## Pregunta 2 — chequeo del entorno

```bash
kind --version              # kind version 0.33.0
kubectl version --client    # Client Version: v1.29.2
```

No se califica (anotar las versiones).

## Crear el cluster

```bash
kind create cluster --name mlzoomcamp-2026
kubectl cluster-info --context kind-mlzoomcamp-2026
```

## Pregunta 3 — primitivas de Kubernetes

¿Cuál es la unidad de cómputo desplegable más chica que Kubernetes crea y administra?

- Node
- **Pod** ✓
- Deployment
- Service

## Pregunta 4 — tipo de servicio por defecto

`kubectl get services --context kind-mlzoomcamp-2026` → ¿`TYPE` del servicio `kubernetes`?

```
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   39s
```

- `NodePort`
- **`ClusterIP`** ✓
- `ExternalName`
- `LoadBalancer`

## Pregunta 5 — cargar la imagen local

¿Qué comando carga la imagen en el cluster?

- `kind create cluster`
- `kind build node-image`
- **`kind load docker-image`** ✓
- `kubectl apply`

## Pregunta 6 — desplegar la API

```bash
kubectl apply -f deployment.yaml --context kind-mlzoomcamp-2026
kubectl rollout status deployment/subscription --context kind-mlzoomcamp-2026
# deployment "subscription" successfully rolled out
```

¿Qué puerto usa el contenedor?

- `80`
- `8080`
- `9000`
- **`9696`** ✓

## Pregunta 7 — exponer el deployment

```bash
kubectl apply -f service.yaml --context kind-mlzoomcamp-2026
kubectl port-forward service/subscription 9696:80 --context kind-mlzoomcamp-2026
python ../05-deployment/q6_test.py     # {'conversion_probability': 0.769799, 'conversion': True}
```

¿Qué selector rutea el tráfico al deployment?

- `app: api`
- **`app: subscription`** ✓
- `app: lead-scoring`
- `app: zoomcamp-model`

## Autoescalado

```bash
kubectl apply -f hpa.yaml --context kind-mlzoomcamp-2026
kubectl get hpa subscription-hpa --context kind-mlzoomcamp-2026
# subscription-hpa   Deployment/subscription   cpu: <unknown>/20%   1   3   1
```

`<unknown>` = falta metrics-server (estado de infraestructura, no es respuesta).

## Pregunta 8 — configuración del HPA

¿Qué `maxReplicas` declara `hpa.yaml`?

- `1`
- `2`
- **`3`** ✓
- `4`

## Limpieza

```bash
kind delete cluster --name mlzoomcamp-2026
```

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 0.770 |
| Q2 | kind 0.33.0 / kubectl v1.29.2 (no calificada) |
| Q3 | Pod |
| Q4 | ClusterIP |
| Q5 | kind load docker-image |
| Q6 | 9696 |
| Q7 | app: subscription |
| Q8 | 3 |
