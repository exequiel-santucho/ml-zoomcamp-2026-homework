# Módulo 10 - Kubernetes (Homework 10)

Despliegue de la API de lead scoring (imagen del HW5) en un cluster local **kind**.

## Archivos

- **deployment.yaml**, **service.yaml**, **hpa.yaml**: manifiestos oficiales del release 2026.
- **run_hw10.sh**: automatiza todo (build → cluster → deploy → service → prueba → HPA → limpieza).
- **tools/kind.exe**: binario de kind para Windows (no se sube a Git; descargar de https://kind.sigs.k8s.io).

## Requisitos

- Docker corriendo.
- `kubectl` (viene con Docker Desktop).
- `kind` (en el PATH o en `tools/`).

## Ejecutar

```bash
bash run_hw10.sh                  # todo de punta a punta y borra el cluster al final
KEEP_CLUSTER=1 bash run_hw10.sh   # deja el cluster vivo para explorar con kubectl
```

## Salida esperada (resumida)

```
Q1  {'conversion_probability': 0.769799, 'conversion': True}
Q4  kubernetes   ClusterIP   10.96.0.1   <none>   443/TCP
Q6  deployment "subscription" successfully rolled out
Q7  {'conversion_probability': 0.769799, 'conversion': True}   (vía port-forward)
Q8  subscription-hpa   Deployment/subscription   cpu: <unknown>/20%   1   3   1
```

## Problemas que encontré

- `kind create cluster` falló una vez con `connection refused` en el API server porque la máquina estaba
  sin CPU libre (entrenando el HW8). Con la CPU libre funcionó al primer intento.
- Si Docker Desktop devuelve `Internal Server Error`, reiniciarlo.
- Si `kind create cluster` falla con `failed while waiting for the control plane to start`, el control plane está
  tardando más de los 4 minutos por defecto (pasa con Docker Desktop en Windows). Solución: un config de kind con
  `timeouts.controlPlaneComponentHealthCheck: 10m0s` en la `InitConfiguration` y `--image kindest/node:v1.33.1`.
- Si `kind load docker-image` parece colgado, es lentitud de Docker al exportar la imagen. Alternativa:
  `docker save <imagen> -o image.tar && kind load image-archive image.tar --name <cluster>`.
