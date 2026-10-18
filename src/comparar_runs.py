"""
src/comparar_runs.py — Comparación programática de runs con mlflow.search_runs()
(Unidad 3 + Unidad 6).

Ejecutar: python src/comparar_runs.py
"""
import os

import mlflow

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "prevencion_infartos"

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    experimento = mlflow.get_experiment_by_name(EXPERIMENT)
    if experimento is None:
        raise RuntimeError(f"Experimento '{EXPERIMENT}' no encontrado. Ejecuta: python main.py")

    runs_df = mlflow.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="tags.etapa = 'random_forest'",
        order_by=['metrics.cv_score DESC'],
    )
    cols = ['tags.mlflow.runName', 'params.metodo_busqueda', 'metrics.cv_score',
            'metrics.train_recall', 'metrics.train_auc', 'params.n_estimators',
            'params.max_depth', 'params.min_samples_split', 'start_time']
    cols = [c for c in cols if c in runs_df.columns]

    print('COMPARACIÓN DE RUNS — ordenado por score de validación cruzada:')
    print(runs_df[cols].to_string(index=False))

    mejor_run = runs_df.iloc[0]
    print(f'\nMEJOR RUN: {mejor_run["tags.mlflow.runName"]}')
    print(f' CV score : {mejor_run["metrics.cv_score"]:.4f}')
    print(f' Run ID   : {mejor_run["run_id"]}')
