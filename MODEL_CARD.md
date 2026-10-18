# Model Card — InfartosRiskModel

Ficha del modelo en producción, según el formato que pide el PDF "Integración y Proyecto
Final" (autor, fecha, datos de entrenamiento, limitaciones y sesgos). Las cifras salen de
`artifacts/eval_metrics.json`, `artifacts/comparativa_modelos.json` y `reportes/fairness.json`,
que regenera `python main.py`.

## Detalles del modelo

| Campo | Valor |
|---|---|
| Nombre en el Model Registry | `InfartosRiskModel` (stage Production) |
| Tipo | Random Forest con undersampling dentro de un `Pipeline` de imblearn |
| Selección | RandomizedSearchCV, 20 combinaciones × 5 folds, por F1 de validación cruzada |
| Autor | Angelo Flores — Proyecto Final, MLOps: Del Modelo al Entorno Productivo (Smart Data) |
| Fecha | 2026-10-18 |
| Código | `random_forest/run.py` (entrenamiento), `evaluate/run.py` (evaluación y gate) |

## Uso previsto

- **Para qué**: ordenar a los asegurados por riesgo de infarto para invitar a los de mayor
  riesgo a un programa de prevención (chequeo cardiológico, seguimiento).
- **Quién lo usa**: el área de salud de la aseguradora, a través de `POST /predecir`.
- **Fuera de alcance**: no es un diagnóstico médico ni debe usarse para fijar primas, negar
  coberturas o decidir sobre una persona sin revisión humana. La probabilidad no está
  calibrada: sirve para ordenar el riesgo, no como probabilidad real de infarto.

## Datos

| | |
|---|---|
| Fuente | Dataset del caso "Prevención de Infartos" (4 años de historial de salud) |
| Tamaño | 43.400 asegurados, 783 infartos (1,8%) |
| Split | 80/20 estratificado (`random_state=123`): 34.720 train · 8.680 test (157 infartos) |
| Preprocesamiento | IMC imputado (3,4% vacíos) con uniforme en promedio ± desviación; `Flag_fumador` descartado (31% vacíos) |
| Variables | `Flag_hipertension`, `Flag_problem_cardiaco`, `Edad_Encoded`, `Gluocosa_Encoded`, `IMC_Encoded`, `Estado_Encoded`, `TipoTrabajo_Encoded` (cortes del notebook del caso) |
| No usadas | `Genero` y `Zona_residencia` (se auditan en fairness), `Flag_fumador` |

## Métricas

Test holdout, evaluado una sola vez, comparado con la línea base:

| Métrica (test) | Modelo (Random Forest) | Línea base (Regresión Logística) | Umbral del gate |
|---|---|---|---|
| AUC | 0,828 | 0,825 | ≥ 0,80 |
| Recall | 0,815 | 0,815 | ≥ 0,70 |
| Precision | 0,053 | 0,054 | — |
| F1 | 0,099 | 0,102 | ≥ 0,05 |

Matriz de confusión del modelo en test: TN 6.232 · FP 2.291 · FN 29 · TP 128.

**Lectura**: el Random Forest no supera a la línea base (diferencias de 0,003 en AUC y
0,002 en F1, dentro del ruido). Con 7 variables codificadas en tramos, el límite lo ponen
los datos y no el algoritmo. Se mantiene el Random Forest porque fue el modelo elegido en
el notebook del caso y aprueba el gate. El siguiente salto de desempeño tiene que venir de
mejores variables (edad y glucosa continuas), no de más tuning.

## Umbrales de decisión (API)

| Probabilidad | Decisión |
|---|---|
| ≥ 0,50 | PROGRAMA PREVENCION |
| 0,30 – 0,50 | SEGUIMIENTO |
| < 0,30 | RUTINA |

## Fairness por subgrupo (test)

Atributos que el modelo **no** usa, auditados para detectar trato desigual (`reportes/fairness.json`):

| Subgrupo | n | Infartos | Recall | Marcados alto riesgo |
|---|---|---|---|---|
| Hombre | 3.533 | 65 | 0,846 | 29,0% |
| Mujer | 5.145 | 92 | 0,793 | 27,1% |
| Rural | 4.369 | 77 | 0,883 | 27,8% |
| Urbano | 4.311 | 80 | 0,750 | 28,0% |

El género `Other` tiene 2 casos en test y ningún infarto, así que no es evaluable.

**Lectura**: la tasa de asegurados marcados es pareja entre grupos (27–29%), así que el
modelo no concentra el programa en un grupo. El recall muestra brechas: 5 puntos entre
hombres y mujeres, y **13 puntos entre zona rural y urbana**. Con 65 a 92 infartos por
grupo, la incertidumbre de cada recall es de ±0,08 a ±0,10, así que las brechas pueden ser
ruido muestral. Aun así, la zona urbana queda como **riesgo a vigilar**: se re-evalúa en
cada ejecución de `evaluate` y se registra en MLflow (`recall_Zona_residencia_*`).

## Limitaciones

- Variables agrupadas en tramos: dos personas del mismo tramo reciben la misma probabilidad.
- Precisión baja (~5%) por construcción: por cada infarto detectado entran ~18 asegurados
  sanos al programa. Es aceptable solo si el costo de un chequeo es mucho menor que el de un
  infarto no atendido (supuestos en `config.yaml → negocio`).
- Los costos de US$20.000 por infarto no detectado y US$300 por chequeo son supuestos de escenario.
- Probabilidades no calibradas: el undersampling las infla.
- Datos de un solo período y una sola aseguradora; no validado en otras poblaciones.

## Mantenimiento

- **Monitoreo**: `monitoreo_drift.yml`, cada lunes. Hay alerta si alguna variable tiene
  PSI ≥ 0,25 o si más del 30% de las variables tienen drift.
- **Re-entrenamiento**: el modelo nuevo solo pasa a Production si aprueba el quality gate;
  si no, sigue en servicio la versión anterior.
- **Reproducir**: `python main.py` (o `dvc repro`).
