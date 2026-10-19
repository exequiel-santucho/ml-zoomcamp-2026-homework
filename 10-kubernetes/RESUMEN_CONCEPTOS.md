# Módulo 10: Kubernetes y Model Serving - Resumen de Conceptos Clave

## Introducción

Último paso del recorrido de deployment: correr modelos en **Kubernetes**, el estándar de la industria para orquestar contenedores. En los videos se despliega el clasificador de ropa con **TensorFlow Serving** + un **gateway** en Flask; en la tarea 2026 se despliega el **mismo servicio FastAPI del Módulo 5** en un cluster local con **kind**.

```
Módulo 5: un contenedor en tu máquina
Módulo 9: una función serverless
Módulo 10: N réplicas orquestadas, con balanceo, health checks y autoescalado
```

---

## 1. Arquitectura de los videos: TF Serving + Gateway

```
Cliente ──HTTP/JSON──► Gateway (Flask) ──gRPC/protobuf──► TF Serving (C++)
                       - descarga la imagen              - solo inferencia
                       - preprocesa                       - muy rápido
                       - convierte a protobuf             - puede usar GPU
```

- **TensorFlow Serving**: servidor en C++ optimizado para modelos TF (`saved_model`), habla **gRPC** (protocolo binario, más eficiente que JSON).
- **Gateway**: hace el preprocesamiento y la conversión a protobuf.
- **Ventaja de separarlos:** escalan de forma **independiente** (ej. 2 máquinas con GPU para TF Serving y 5 CPU baratas para gateways).

```bash
docker run -it --rm -p 8500:8500 \
  -v "$(pwd)/clothing-model:/models/clothing-model/1" \
  -e MODEL_NAME="clothing-model" tensorflow/serving:2.7.0
saved_model_cli show --dir clothing-model --all   # ver firma: nombres de input/output
```

---

## 2. Docker Compose

Correr **varios contenedores juntos** con una red compartida, definido en `docker-compose.yaml`:

```yaml
services:
  clothing-model:
    image: zoomcamp-10-model:xception-v4-001
  gateway:
    image: zoomcamp-10-gateway:002
    environment:
      - TF_SERVING_HOST=clothing-model:8500    # el nombre del servicio es el hostname
    ports:
      - "9696:9696"
```

```bash
docker compose up -d
docker compose down
```

Ideal para desarrollo local; para producción a escala se usa Kubernetes.

---

## 3. Conceptos de Kubernetes

| Concepto | Qué es | Analogía |
|---|---|---|
| **Cluster** | Conjunto de máquinas administradas por Kubernetes | El datacenter |
| **Node** | Una máquina (física o VM) del cluster | Un servidor |
| **Pod** | **La unidad mínima desplegable**: uno o más contenedores que comparten red y almacenamiento | Un contenedor "con envoltorio" |
| **Deployment** | Declara cuántas réplicas de un pod queremos y cómo actualizarlas | El "molde" + la cantidad |
| **Service** | Punto de acceso estable (IP/DNS) que balancea tráfico entre los pods de un deployment | El balanceador |
| **HPA** | Horizontal Pod Autoscaler: agrega/quita réplicas según CPU u otra métrica | El que contrata/despide |
| **Ingress** | Expone servicios HTTP al exterior con reglas de ruteo | La puerta de entrada |

**Tipos de Service:**

| Tipo | Accesible desde |
|---|---|
| **ClusterIP** (default) | Solo dentro del cluster |
| **NodePort** | Un puerto en cada nodo |
| **LoadBalancer** | IP externa (requiere un proveedor cloud: EKS, GKE, AKS) |
| **ExternalName** | Alias DNS a un servicio externo |

**Modelo declarativo:** describís el **estado deseado** en YAML (`kubectl apply -f`) y Kubernetes trabaja constantemente para que el estado real coincida (si un pod muere, crea otro).

---

## 4. kind: Kubernetes local

**kind** (Kubernetes IN Docker) crea un cluster de un nodo usando un contenedor de Docker como "máquina".

```bash
kind create cluster --name mlzoomcamp-2026
kubectl cluster-info --context kind-mlzoomcamp-2026
kubectl get services --context kind-mlzoomcamp-2026   # 'kubernetes' es ClusterIP
kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026
kind delete cluster --name mlzoomcamp-2026
```

> Los nodos de kind **no ven las imágenes del Docker local**: hay que cargarlas con `kind load docker-image` (o subirlas a un registry). Por eso el deployment usa `imagePullPolicy: IfNotPresent`.

Alternativas locales: minikube, k3d, Docker Desktop Kubernetes.

---

## 5. Deployment (YAML de la tarea, anotado)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: subscription
spec:
  replicas: 1
  selector:
    matchLabels:
      app: subscription          # qué pods administra este deployment
  template:                      # plantilla del pod
    metadata:
      labels:
        app: subscription        # etiqueta que matchea el selector (¡y el Service!)
    spec:
      containers:
        - name: subscription
          image: zoomcamp-model:2026-hw10
          imagePullPolicy: IfNotPresent
          resources:
            requests: {memory: "64Mi", cpu: "100m"}   # lo que se reserva (scheduling)
            limits:   {memory: "256Mi", cpu: "500m"}  # el máximo permitido
          ports:
            - containerPort: 9696
          readinessProbe:                  # ¿está listo para recibir tráfico?
            httpGet: {path: /health, port: 9696}
            initialDelaySeconds: 2
            periodSeconds: 5
```

- **`100m` CPU** = 0.1 núcleo ("milicores").
- **Requests vs limits:** requests = garantía para ubicar el pod en un nodo; limits = techo (si pasa el límite de memoria, el pod se reinicia: *OOMKilled*).
- **Readiness probe** usa el `/health` que creamos en el Módulo 5. (También existe **liveness probe**: si falla, Kubernetes reinicia el contenedor.)

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deployment/subscription
kubectl get pods
kubectl describe pod <nombre>     # eventos: ¿por qué no arranca?
kubectl logs <nombre>
```

---

## 6. Service y port-forward

```yaml
apiVersion: v1
kind: Service
metadata:
  name: subscription
spec:
  type: ClusterIP
  selector:
    app: subscription       # rutea a los pods con esta etiqueta
  ports:
    - port: 80              # puerto del Service
      targetPort: 9696      # puerto del contenedor
```

```bash
kubectl apply -f service.yaml
kubectl port-forward service/subscription 9696:80    # túnel local → Service
python q6_test.py
```

---

## 7. Autoescalado (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: subscription-hpa
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: subscription}
  minReplicas: 1
  maxReplicas: 3
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: {type: Utilization, averageUtilization: 20}   # % de los requests de CPU
```

- Necesita **metrics-server** instalado; sin él, el target aparece como `<unknown>` (lo que vimos en kind).
- Para probarlo: un loop de requests (`load_test.py` del workshop) y `kubectl get hpa --watch`.

---

## 8. EKS (Kubernetes en AWS)

```bash
eksctl create cluster -f eks-config.yaml     # crea el cluster (tarda ~15-20 min)
# subir imágenes a ECR, cambiar 'image:' en los YAML, Service tipo LoadBalancer
eksctl delete cluster --name mlzoomcamp-eks  # ¡EKS cobra por hora!
```

---

## 9. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Contenedor local (imagen del HW5) | 0.770 |
| Q2 | Versiones (no calificada) | kind 0.33.0, kubectl client v1.29.2 (cluster k8s v1.37.0) |
| Q3 | Unidad mínima desplegable | **Pod** |
| Q4 | Tipo del servicio `kubernetes` | **ClusterIP** |
| Q5 | Cargar imagen local en kind | `kind load docker-image` |
| Q6 | Puerto del contenedor | **9696** |
| Q7 | Selector del Service | `app: subscription` |
| Q8 | `maxReplicas` del HPA | **3** |

Vía port-forward, el servicio en Kubernetes devolvió la misma probabilidad (0.769799) y los mismos checksums en `/health` que el contenedor local.

---

## Preguntas típicas de entrevista

1. **Pod vs Deployment vs Service.** Pod: unidad que corre contenedores. Deployment: mantiene N réplicas de un pod y maneja updates. Service: IP/DNS estable que balancea entre esos pods.
2. **¿Readiness vs liveness probe?** Readiness: si falla, el pod deja de recibir tráfico. Liveness: si falla, el contenedor se reinicia.
3. **Requests vs limits.** Requests se usan para el scheduling y como base del HPA; limits son el techo (CPU se estrangula, memoria se mata).
4. **¿Cómo escala Kubernetes un modelo?** HPA según CPU/memoria o métricas custom (ej. requests por segundo); Cluster Autoscaler agrega nodos.
5. **¿Por qué separar preprocesamiento (gateway) e inferencia (TF Serving)?** Para escalar cada parte con el hardware adecuado (CPU vs GPU) y de forma independiente.
6. **¿gRPC vs REST/JSON?** gRPC es binario (protobuf), más rápido y compacto, con contratos tipados; REST/JSON es más simple e interoperable.
7. **¿Cómo hacés un rollout sin downtime?** Rolling update del Deployment + readiness probes (los pods nuevos reciben tráfico solo cuando están listos); rollback con `kubectl rollout undo`.

## Recursos

- Lecciones: `machine-learning-zoomcamp/10-kubernetes/` (y `workshop/`)
- Código de la tarea: esta carpeta (`run_hw10.sh` automatiza todo)
