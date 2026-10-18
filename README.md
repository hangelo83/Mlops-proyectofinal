# Prevención de infartos en asegurados — Pipeline MLOps

Proyecto final del curso **MLOps: Del Modelo al Entorno Productivo** (Smart Data, 4ta edición).

Una aseguradora quiere identificar a los asegurados con alto riesgo de infarto para incluirlos
en un programa de prevención, y así evitar costos de cuidados intensivos y pagos de cobertura.
Este repositorio lleva el modelo del notebook del caso a un pipeline reproducible que se ejecuta
con un comando, valida los datos y la calidad del modelo antes de desplegarlo, versiona cada
modelo en MLflow, lo sirve como API en Docker y monitorea el drift.

El código reutiliza la estructura, las funciones y las librerías del repositorio del curso
(Unidades 1 a 10). La base es el pipeline de 7 etapas de la Unidad 10.

## Inicio rápido en GitHub Codespaces

1. En GitHub: **Code → Codespaces → Create codespace on main**. El contenedor
   (`.devcontainer/`) instala Python 3.11, las dependencias, DVC y Docker.
2. En la terminal del Codespace (Ctrl+`):

```bash
python main.py                                   # las 7 etapas (~1-2 min)
uvicorn api.app:app --host 0.0.0.0 --port 8000   # API → pestaña PORTS → 8000 → /docs
```

| Comando | Para qué sirve |
|---|---|
| `python main.py` | Ejecuta las 7 etapas en secuencia |
| `python main.py --steps random_forest evaluate` | Ejecuta solo algunas etapas |
| `make test` | Tests unitarios y de la API con cobertura |
| `make mlflow-ui` | UI de MLflow en el puerto 5000 (runs, Model Registry) |
| `make comparar` | Tabla comparativa de runs (GridSearch vs RandomSearch) |
| `make drift-demo` | Monitoreo con un lote de drift simulado |
| `make docker-build && make docker-run && make smoke` | API en Docker + smoke tests |
| `make preprod-up` | Stack pre-producción: MLflow + trainer + API |
| `make deploy VERSION=v1.0.0` | Despliegue con smoke tests y rollback automático |
| `make dvc-repro` | El mismo pipeline, reproducido con DVC |
| `make help` | Todos los comandos |

## Pipeline

```
data/raw/dataset_infartos.csv
   │
   ▼  1. download       data/run.py            valida columnas (cargar_datos)
   ▼  2. preprocess     preprocess/run.py      imputa IMC, crea variables *_Encoded
   ▼  3. segregate      segregate/run.py       split 80/20 estratificado
   ▼  4. check_data     check_data/            11 tests de schema, rangos y distribución
   ▼  5. random_forest  random_forest/run.py   GridSearchCV + RandomizedSearchCV → MLflow
   │                                           el mejor se registra en el Model Registry (Staging)
   ▼  6. evaluate       evaluate/run.py        quality gate AUC / Recall / F1 en test
   │                                           si aprueba: Staging → Production
   ▼  7. drift          drift/run.py           Evidently + PSI, train vs test
   │
   ▼  api/ (FastAPI)  →  Docker  →  GitHub Actions
```

Si una etapa falla, `main.py` se detiene y CI no construye la imagen. Todos los parámetros
están en `config.yaml`.

## Resultados

Random Forest con undersampling dentro de la validación cruzada, sobre las 7 variables del
notebook del caso. El mejor de los dos runs de tuning (RandomizedSearchCV, CV F1 0.096) se
evalúa una sola vez en el test holdout (8,680 asegurados, 157 infartos):

| Métrica (test) | Valor | Umbral del quality gate |
|---|---|---|
| AUC | **0.828** | ≥ 0.80 |
| Recall | **0.815** (128 de 157 infartos detectados) | ≥ 0.70 |
| Precision | 0.053 | — |
| F1 | 0.099 | ≥ 0.05 |
| Accuracy | 0.733 | — |

![Reporte de evaluación](docs/reporte_evaluacion.png)

**Cómo leer estos números.** Con solo 1.8% de infartos, un modelo que siempre diga "no" tiene
98% de accuracy y no detecta a nadie. Por eso el gate exige recall y AUC. La precisión baja es
el costo esperado de priorizar el recall: por cada infarto detectado entran ~18 asegurados
sanos al programa de prevención. Con los supuestos de costo de `config.yaml`
(US$20,000 por infarto no detectado y US$300 por chequeo preventivo), el modelo cuesta
US$1.27 M frente a los US$3.14 M de no hacer nada (157 × US$20,000).

## Uso en producción

```bash
curl -X POST http://localhost:8000/predecir -H 'Content-Type: application/json' -d '{
  "Edad": 67, "Flag_hipertension": 1, "Flag_problem_cardiaco": 1,
  "Promedio_nivel_glucosa": 228.7, "IMC": 36.6,
  "Estados_civil": "Si", "Tipo_trabajo": "Empresa_privada"}'
```

```json
{"probabilidad_infarto": 0.8087, "decision": "PROGRAMA PREVENCION",
 "nivel_riesgo": "MUY ALTO", "umbral_usado": 0.5, "modelo": "RandomForestClassifier"}
```

| Probabilidad | Decisión | Acción sugerida |
|---|---|---|
| ≥ 0.50 | PROGRAMA PREVENCION | Chequeo cardiológico y seguimiento activo |
| 0.30 – 0.50 | SEGUIMIENTO | Control anual reforzado |
| < 0.30 | RUTINA | Sin acción adicional |

La API recibe los datos **crudos** del asegurado y les aplica la misma función
`transformar_variables()` del entrenamiento. El IMC es opcional: si falta, se imputa con el
promedio guardado en `artifacts/preprocesamiento.json`. Pydantic rechaza con HTTP 422 los valores
fuera de rango. `GET /health` informa el modelo cargado, su origen (`pickle` o
`registry:Production`) y sus métricas de test.

**Límites del modelo.** Usa variables agrupadas en tramos (edad, glucosa, IMC), así que no
distingue entre dos personas del mismo tramo. Las probabilidades no están calibradas, porque el
undersampling las infla: sirven para ordenar el riesgo, no como probabilidad real de infarto.

## Re-entrenamiento

| Disparador | Cómo se detecta |
|---|---|
| Drift en los datos | `monitoreo_drift.yml` corre cada lunes. Falla si más del 30% de las variables derivan o si alguna tiene PSI ≥ 0.25 |
| Nuevos datos etiquetados | Reemplazar `data/raw/dataset_infartos.csv` y hacer push. CI re-entrena y evalúa |
| Calendario | Re-entrenamiento trimestral recomendado, aunque no haya drift |

El modelo nuevo solo pasa a Production si supera el quality gate. Si no lo supera, queda en
Staging y la versión anterior sigue en servicio. Para un rollback manual se usa
`promover_a_produccion()` en `src/manage_versions.py`, o `make rollback`/`deploy.sh` para la
imagen Docker.

## Cobertura de las 10 unidades del curso

| Unidad | Tema | Dónde está en este proyecto |
|---|---|---|
| 1 | Introducción a MLOps, repositorio base | Repositorio Git, `README.md`, `requirements.txt`, `.devcontainer/` para Codespaces |
| 2 | Ingeniería de software para DS | `src/` modular (`cargar_datos`, `COLUMNAS_REQUERIDAS`, `imputar_imc`, `obtener_balanceador` con el `sampler_map` de `dividir_y_balancear`), logging, `tests/test_preprocessing.py` |
| 3 | Versionamiento de datos y modelos | `dvc.yaml` + `dvc.lock` (hash de cada dato y modelo), MLflow tracking, `src/comparar_runs.py` |
| 4 | Pipelines reproducibles | Etapas separadas con validación de salida (`validar_schema`), gate de recall, reporte PNG y costo FN/FP de `evaluate/run.py` |
| 5 | Docker + FastAPI | `api/` (`app.py`, `predictor.py` con la clase `Predictor`, `schemas.py`), `preprocesamiento.json`, `Dockerfile`, `tests/test_api.py` |
| 6 | Ajuste y seguimiento | GridSearchCV y RandomizedSearchCV como runs comparables, selección del mejor, Model Registry → Staging |
| 7 | CI/CD | `.github/workflows/ml_pipeline.yml` (lint-and-test → ml-pipeline → build-docker), `setup.cfg`, `Makefile` |
| 8 | Deployment pre-producción | `docker-compose.preprod.yml` (mlflow → trainer → api), `Dockerfile.trainer`, `deploy.sh` con rollback, `src/manage_versions.py`, `tests/smoke/` |
| 9 | Monitoreo y mantenimiento | `drift/run.py` (Evidently `DataDriftPreset`, `calcular_psi`, `clasificar_psi`), `.github/workflows/monitoreo_drift.yml` semanal |
| 10 | Integración end-to-end | `main.py` (orquestador de 7 etapas), `config.yaml`, `check_data/` |

## Correcciones al template de la Unidad 10

El PDF de la Unidad 10 documenta cuatro problemas del código de referencia. Aquí se corrigen
con patrones del propio curso:

| Problema en la Unidad 10 | Corrección |
|---|---|
| Desajuste train/serve: la API mandaba valores crudos a un modelo entrenado con datos normalizados, y todas las predicciones salían iguales | La API importa `transformar_variables()` de `src/preprocessing.py`, la misma función del entrenamiento. `test_alto_riesgo_va_a_prevencion` detecta si se rompe |
| Evidently con la API antigua (`evidently.report`), que falla en ≥ 0.5 | API nueva de la Unidad 9 (`from evidently import Report`), con fallback KS |
| Gate de F1 ≥ 0.70, imposible con datos desbalanceados | Gate de AUC + Recall (`validar_recall` de la Unidad 4). F1 queda como piso mínimo |
| Rutas `_data/` en el Makefile y en CI | Todas las rutas apuntan a `data/run.py` |

Otros ajustes:

- El balanceo va dentro de un `Pipeline` de imblearn, para no filtrar información del fold de
  validación.
- El modelo se elige por validación cruzada; el test se usa una sola vez, en `evaluate`.
- `main.py` respeta un `MLFLOW_TRACKING_URI` externo, lo que permite usar el servidor MLflow
  del stack pre-prod.
- El servidor MLflow del compose usa `--artifacts-destination`, para que trainer y API
  compartan los modelos.

## Estructura

```
├── main.py                  orquestador (Unidad 10)
├── config.yaml              parámetros del pipeline
├── data/run.py · data/raw/  etapa 1 + CSV original del caso
├── preprocess/ segregate/ check_data/ random_forest/ evaluate/ drift/   etapas 2-7
├── src/                     código compartido (pipeline + API)
├── api/                     FastAPI
├── tests/                   unitarios, API y smoke
├── notebooks/01_eda_infartos.ipynb   análisis exploratorio del caso
├── dvc.yaml · dvc.lock      pipeline DVC
├── Dockerfile · Dockerfile.trainer · docker-compose.preprod.yml · deploy.sh
└── .github/workflows/       CI/CD y monitoreo de drift
```

## Versionado de datos

El CSV crudo (2 MB) va en Git, para que el pipeline corra en Codespaces y en CI sin
configurar un remoto. Las salidas del pipeline (datos procesados, modelo, métricas) no van a
Git: `dvc.lock` guarda su hash md5, y `dvc repro` las regenera y verifica. Para compartir el
cache entre máquinas basta con agregar un remoto (`dvc remote add -d storage gdrive://…`) y
ejecutar `dvc push`.
