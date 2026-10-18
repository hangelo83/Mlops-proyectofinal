# Makefile — Pipeline MLOps End-to-End — Prevención de Infartos
# Uso: make <target>      Ayuda: make help
.PHONY: install all lint test pipeline etapa1 etapa2 etapa3 etapa4 etapa5 etapa6 etapa7 \
        demo drift-demo comparar versions serve docker-build docker-run docker-stop smoke \
        preprod-up preprod-down preprod-logs preprod-ps deploy rollback dvc-repro mlflow-ui clean help

COMPOSE_FILE = docker-compose.preprod.yml
VERSION      ?= latest

install:
	pip install -r requirements.txt

# ── CI/CD local: lo mismo que GitHub Actions antes de hacer push (Unidad 7) ──
all: lint test pipeline docker-build
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

lint:
	flake8 .

test:
	pytest tests/ --ignore=tests/smoke --cov=src --cov=api --cov-report=term-missing

# ── Pipeline completo (Unidad 10) ─────────────────────────────────────────────
pipeline:
	python main.py

# Demo de punta a punta: 7 etapas + drift simulado + Docker + MLflow UI y API arriba
demo:
	python main.py --demo

# Etapas individuales
etapa1: ; python data/run.py
etapa2: ; python preprocess/run.py
etapa3: ; python segregate/run.py
etapa4: ; pytest check_data/test_data.py -v --tb=short
etapa5: ; python random_forest/run.py
etapa6: ; python evaluate/run.py
etapa7: ; python drift/run.py

# Demo de monitoreo: lote con población más envejecida (Unidad 9)
drift-demo:
	python drift/run.py --simular_drift

# ── MLflow: comparar runs y versiones del Model Registry (Unidades 3, 6, 8) ───
comparar:
	python src/comparar_runs.py

versions:
	python src/manage_versions.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
	  --allowed-hosts "localhost:*,127.0.0.1:*,*.app.github.dev"

# ── DVC: mismo pipeline declarado en dvc.yaml (Unidad 3) ──────────────────────
dvc-repro:
	dvc repro

# ── Servidor API (Unidad 5) ───────────────────────────────────────────────────
serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t infartos-api:local .

docker-run:
	docker run -d -p 8000:8000 --name infartos-api infartos-api:local

docker-stop:
	docker stop infartos-api && docker rm infartos-api

# Smoke tests (requiere la API levantada: make serve, make docker-run o make preprod-up)
smoke:
	pytest tests/smoke/ -v --tb=short

# ── Pre-producción (Unidad 8) ─────────────────────────────────────────────────
preprod-up:
	docker compose -f $(COMPOSE_FILE) up --build -d --wait
	@echo "  API   : http://localhost:8000/docs   (o URL pública en la pestaña PORTS)"
	@echo "  MLflow: http://localhost:5000"

preprod-down:
	docker compose -f $(COMPOSE_FILE) down -v

preprod-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

preprod-ps:
	docker compose -f $(COMPOSE_FILE) ps -a

deploy:
	bash deploy.sh $(VERSION)

## Rollback a una versión anterior de la imagen: make rollback VERSION=v1.0.0
rollback:
	docker compose -f $(COMPOSE_FILE) down
	docker tag infartos-api:$(VERSION) infartos-api:latest
	docker compose -f $(COMPOSE_FILE) up -d --no-build

clean:
	rm -rf data/infartos_*.csv artifacts/ reportes/ mlruns/ mlflow.db pipeline_run.log .coverage
	find . -name "__pycache__" -type d -not -path "./.venv/*" -exec rm -rf {} +
	@echo "Limpieza completada"

help:
	@echo ""
	@echo "=== Pipeline MLOps — Prevención de Infartos ==="
	@echo "  make install       — instalar dependencias"
	@echo "  make pipeline      — ejecutar las 7 etapas en orden"
	@echo "  make demo          — 7 etapas + drift simulado + Docker + MLflow UI y API arriba"
	@echo "  make etapa1..7     — ejecutar una etapa específica"
	@echo "  make test / lint   — tests unitarios + API / flake8"
	@echo "  make all           — lint + test + pipeline + docker (igual que CI)"
	@echo "  make serve         — levantar API FastAPI en puerto 8000"
	@echo "  make mlflow-ui     — UI de MLflow en puerto 5000"
	@echo "  make comparar      — tabla comparativa de runs"
	@echo "  make versions      — versiones del Model Registry"
	@echo "  make drift-demo    — monitoreo con drift simulado"
	@echo "  make docker-build / docker-run / smoke"
	@echo "  make preprod-up    — stack mlflow + trainer + api"
	@echo "  make deploy VERSION=v1.0.0 — despliegue con smoke tests y rollback"
	@echo "  make rollback VERSION=v1.0.0 — volver a una imagen anterior"
	@echo "  make dvc-repro     — reproducir el pipeline con DVC"
	@echo "  make clean         — limpiar artefactos"
	@echo ""
