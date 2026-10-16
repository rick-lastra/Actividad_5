# CHANGELOG

Historial de versiones del proyecto. Los datos y el código se marcan con tags de Git.

## v1.0 (2026-10-16) - Reporte final

- `fuentes/entrena.ipynb`: notebook con una sección por punto del reporte técnico (A a J).
- `fuentes/train.py`: entrenamiento de 3 modelos con Grid Search y registro en MLflow.
- Resultados en el 20 % de prueba (atletas nuevos), por ROC-AUC: regresión logística 0.859, XGBoost 0.855,
  Random Forest 0.854. La t pareada y Wilcoxon (10 folds) no encuentran diferencias significativas, así que se
  elige el modelo más simple: regresión logística.

## datos-v1.1 (2026-10-16) - Datos limpios

- `fuentes/datos_prep.py`: funciones de limpieza.
- `datos/datos_limp/sesiones_limpias.csv`: 81,901 sesiones.
  - 1,283 duplicados exactos eliminados.
  - 427 duplicados por atleta + fecha eliminados (se conserva la captura más reciente).
  - 3,647 sesiones sin etiqueta eliminadas (primeros 28 días de cada atleta).
  - 2,474 valores de `level` corregidos.
  - 251 valores de `prev_rpe` fuera de 1-10 convertidos en vacío.
  - 14 columnas de la misma sesión eliminadas (fuga de información).
- `datos/datos_limp/sugarwod_limpio.csv`: fechas, tiempos en segundos y `score_type` corregido.

## datos-v1.0 (2026-10-16) - Datos originales

- `datos/datos_ini/`: `sessions.csv` (87,258 filas), `athletes.csv` (280) y `workouts_sugarwod.csv` (792, real).
- Publicados en Hugging Face: `rick-lastra/wodpredict-synth`, tag `v1.0`.
