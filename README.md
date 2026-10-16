# Actividad 5. WODPredict-AI: predicción de sobrecarga con MLflow

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rick-lastra/Actividad_5/blob/main/fuentes/entrena.ipynb)

Gestión de datos para entrenamiento, entrenamiento y ajuste de modelos.

**Problema:** predecir si un atleta de CrossFit tendrá **sobrecarga al día siguiente** (clasificación), con datos de
280 atletas. Se comparan regresión logística, Random Forest y XGBoost con Grid Search, validación cruzada y MLflow.

## Estructura

```
Actividad_5/
├── datos/
│   ├── datos_ini/        # datos originales
│   └── datos_limp/       # datos limpios
├── fuentes/
│   ├── entrena.ipynb     # notebook de Colab (entrenamiento y evaluación)
│   ├── datos_prep.py     # funciones de limpieza
│   └── train.py          # script de entrenamiento y registro en MLflow
├── README.md             # guía de reproducción y decisiones
└── CHANGELOG.md          # reporte final
```

## Datos

- **Fuente:** [huggingface.co/datasets/rick-lastra/wodpredict-synth](https://huggingface.co/datasets/rick-lastra/wodpredict-synth)
  (público, versión `v1.0`). Copia de respaldo en `datos/datos_ini/`.
- `sessions.csv`: una fila por atleta y día entrenado (sintético, 87,258 filas).
- `athletes.csv`: ficha de los 280 atletas (sintético).
- `workouts_sugarwod.csv`: exportación real de SugarWOD de un atleta (Actividad 4).

## Cómo reproducir

**Opción 1, Colab:** dar clic en el botón **Abrir en Colab** (arriba) y ejecutar todas las celdas
(Entorno de ejecución > Ejecutar todas). No necesita GPU: funciona igual con CPU o con T4. La primera celda descarga el repositorio y las librerías.

**Opción 2, terminal (WSL / Linux):**
```bash
pip install pandas scikit-learn xgboost mlflow huggingface_hub matplotlib scipy
python fuentes/datos_prep.py      # limpia datos_ini -> datos_limp
python fuentes/train.py           # entrena los 3 modelos y registra en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000    # abrir http://127.0.0.1:5000
```

## Decisiones

| Decisión | Por qué |
|---|---|
| Separar entrenamiento y prueba **por atleta** | Las sesiones de un mismo atleta se parecen; mezclarlas inflaría los resultados |
| Quitar RPE, carga y comentario **de la misma sesión** | Se conocen después de entrenar: sería hacer trampa |
| Imputar, escalar y codificar **dentro del modelo** (Pipeline) | Así se ajustan solo con los datos de entrenamiento |
| Elegir hiperparámetros por **ROC-AUC** | Solo 12.5 % de las sesiones tienen sobrecarga; accuracy engaña con clases desbalanceadas |
| t pareada y Wilcoxon con **10 folds** | Con 5 folds Wilcoxon nunca llega a p < 0.05 |
| Versionar datos con Hugging Face (tag `v1.0`) y Git, cambios en `CHANGELOG.md` | Saber exactamente qué datos usó cada experimento |
