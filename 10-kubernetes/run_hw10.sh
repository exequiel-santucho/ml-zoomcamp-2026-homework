#!/usr/bin/env bash
# HW10 de punta a punta: imagen -> cluster kind -> deployment -> service -> HPA -> prueba -> limpieza.
# Requiere Docker corriendo. Usa tools/kind.exe si `kind` no está en el PATH.
#
#   bash run_hw10.sh            # corre todo y borra el cluster al final
#   KEEP_CLUSTER=1 bash run_hw10.sh   # deja el cluster vivo para explorar
set -euo pipefail
cd "$(dirname "$0")"

CLUSTER=mlzoomcamp-2026
CTX=kind-$CLUSTER
IMAGE=zoomcamp-model:2026-hw10
# funciona tanto con la carpeta hw5-deployment/ como con 05-deployment/ al mismo nivel
if [[ -d ../../05-deployment/hw5-deployment ]]; then HW5=../../05-deployment/hw5-deployment; else HW5=../05-deployment; fi
KIND=$(command -v kind || echo tools/kind.exe)

echo "== Q2: versiones"
"$KIND" --version
kubectl version --client

echo "== Construir la imagen del HW5 con el tag de HW10"
docker build -t "$IMAGE" "$HW5"

echo "== Q1: probar el contenedor local"
docker run -d --rm --name hw10-local -p 9696:9696 "$IMAGE" >/dev/null
sleep 4
uv run --project "$HW5" python "$HW5/q6_test.py"
docker stop hw10-local >/dev/null

echo "== Crear cluster"
"$KIND" create cluster --name "$CLUSTER"
kubectl cluster-info --context "$CTX"

echo "== Q4: servicios por defecto"
kubectl get services --context "$CTX"

echo "== Q5: cargar la imagen en el cluster"
"$KIND" load docker-image "$IMAGE" --name "$CLUSTER"

echo "== Q6: deployment"
kubectl apply -f deployment.yaml --context "$CTX"
kubectl rollout status deployment/subscription --context "$CTX"
kubectl get pods --context "$CTX"

echo "== Q7: service + port-forward"
kubectl apply -f service.yaml --context "$CTX"
kubectl get service subscription --context "$CTX"
kubectl port-forward service/subscription 9696:80 --context "$CTX" >/dev/null 2>&1 &
PF_PID=$!
sleep 5
uv run --project "$HW5" python "$HW5/q6_test.py"
kill $PF_PID || true

echo "== Q8: HPA"
kubectl apply -f hpa.yaml --context "$CTX"
kubectl get hpa subscription-hpa --context "$CTX"

if [[ "${KEEP_CLUSTER:-0}" != "1" ]]; then
  echo "== Limpieza"
  "$KIND" delete cluster --name "$CLUSTER"
fi
