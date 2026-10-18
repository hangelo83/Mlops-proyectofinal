#!/bin/bash
# deploy.sh — Flujo manual de despliegue pre-producción (Unidad 8)
#
# Uso en Codespace:
#   bash deploy.sh              # despliega con tag 'latest'
#   bash deploy.sh v1.2.0       # despliega con tag específico
#
# Flujo: build → levantar → smoke tests → tag → verify → (rollback si falla)

set -e  # salir si cualquier comando falla

VERSION=${1:-latest}
IMAGE_NAME='infartos-api'
COMPOSE_FILE='docker-compose.preprod.yml'

echo "============================================="
echo " DEPLOY PRE-PRODUCCIÓN — versión: $VERSION"
echo "============================================="

# ── PASO 1: Guardar versión actual para rollback ──────────────────────────────
echo "[1/6] Guardando versión actual para rollback..."
CURRENT=$(docker images --format "{{.Tag}}" $IMAGE_NAME 2>/dev/null | grep -v latest | head -1)
CURRENT=${CURRENT:-none}  # si no hay imagen previa, no se intenta rollback
echo "  Versión actual: $CURRENT"

# ── PASO 2: Build de todas las imágenes ──────────────────────────────────────
echo "[2/6] Construyendo imágenes Docker..."
docker compose -f $COMPOSE_FILE build
echo "  Build OK"

# ── PASO 3: Levantar el entorno pre-prod (espera a que la API esté sana) ──────
echo "[3/6] Levantando entorno completo (mlflow → trainer → api)..."
docker compose -f $COMPOSE_FILE up -d --wait
echo "  Servicios arriba"

# ── PASO 4: Smoke tests ───────────────────────────────────────────────────────
echo "[4/6] Ejecutando smoke tests..."
if pytest tests/smoke/ -v --tb=short -q; then
    echo "  Smoke tests OK"
else
    echo "  SMOKE TESTS FALLARON — ejecutando rollback..."
    docker compose -f $COMPOSE_FILE down
    if [ "$CURRENT" != "none" ]; then
        docker tag "$IMAGE_NAME:$CURRENT" "$IMAGE_NAME:latest"
        docker compose -f $COMPOSE_FILE up -d
        echo "  Rollback a versión $CURRENT completado"
    fi
    exit 1
fi

# ── PASO 5: Tag de la versión ─────────────────────────────────────────────────
echo "[5/6] Tagging imagen como $VERSION..."
docker tag "$IMAGE_NAME:latest" "$IMAGE_NAME:$VERSION"
echo "  Imagen taggeada: $IMAGE_NAME:$VERSION"

# ── PASO 6: Verificación final ────────────────────────────────────────────────
echo "[6/6] Verificación final..."
curl -sf http://localhost:8000/health | python3 -m json.tool
echo ""
echo "============================================="
echo " DEPLOY EXITOSO — $IMAGE_NAME:$VERSION"
echo " API   : http://localhost:8000"
echo " Docs  : http://localhost:8000/docs"
echo " MLflow: http://localhost:5000"
echo "============================================="
