# 🏋️ WODPredict-AI: Predicción de Sobrecarga en Atletas con MLflow

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rick-lastra/Actividad_5/blob/main/fuentes/entrena.ipynb)

## Título y Descripción
**Nombre del proyecto:** WODPredict-AI (Módulo de Predicción de Sobrecarga)
Este repositorio contiene la Actividad 5 del proyecto WODPredict-AI para la red de centros **UnderCover Fitness**. Gestión de datos, entrenamiento y ajuste de modelos. Se preparan y versionan los datos, se entrenan tres modelos con Grid Search y validación cruzada, y todos los experimentos se registran en **MLflow**.

**Objetivo General:**
Predecir si un atleta de CrossFit tendrá **sobrecarga al día siguiente** (`overreaching_label` = 1) usando solo información que se conoce **antes** de entrenar: carga y frecuencia de los últimos días, ACWR, RPE de la sesión anterior y la probabilidad de fatiga de sus comentarios (módulo NLP de la Actividad 4). Así el coach puede ajustar la carga antes de que aparezca la fatiga.

**Contexto Profesional:**
Iniciativa alineada al *Business Case* de UnderCover Fitness para reducir la deserción de atletas en un 25% derivada de lesiones por acumulación de fatiga.

**Organización del notebook:** `fuentes/entrena.ipynb` tiene una sección por cada punto del reporte técnico.

| Sección | Punto del reporte |
|---|---|
| 0 | Preparación del entorno |
| A | Descripción del dataset |
| B | Limpieza y estandarización |
| C | Estrategia de versionamiento |
| D | Visualizaciones |
| E | Modelos entrenados |
| F | Configuración de hiperparámetros |
| G | Métricas obtenidas |
| H | Comparación estadística |
| I | Evidencia de MLflow |
| J | Deuda técnica y MLOps |

## 🤗 Ecosistema Hugging Face
| Componente | Uso en el proyecto |
|---|---|
| **Datasets** | El dataset sintético se publica como dataset **público** `rick-lastra/wodpredict-synth`. El notebook lo descarga con `huggingface_hub.hf_hub_download`. Si no hay internet, usa la copia de `datos/datos_ini/` |
| **Versionado (Hub)** | El tag `v1.0` del dataset fija la versión exacta de los datos (`revision="v1.0"`). Cada run de MLflow guarda el parámetro `dataset = rick-lastra/wodpredict-synth@v1.0` |
| **Models** | Las variables `prob_fatiga_*` reproducen la salida del modelo elegido en la Actividad 4 (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) sobre los comentarios de los atletas |
| **Spaces** | Despliegue futuro: demo en Gradio para que el coach capture la carga de la semana y vea el riesgo de sobrecarga |

## 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python 3.10+
*   **Entorno:** VS Code con WSL (Ubuntu) y Google Colab. No requiere GPU: corre igual en CPU o con una T4 de Colab.
*   **Librerías principales:** `pandas`, `scikit-learn`, `xgboost`, `mlflow`, `huggingface_hub`, `scipy`, `matplotlib`
*   **Modelos entrenados (clasificación binaria):**
    * Regresión logística (`sklearn.linear_model.LogisticRegression`)
    * Random Forest (`sklearn.ensemble.RandomForestClassifier`)
    * XGBoost (`xgboost.XGBClassifier`)
*   **Seguimiento de experimentos:** MLflow con base de datos local SQLite (`mlflow.db`)
*   **Versionado:** Git con tags (`datos-v1.0`, `datos-v1.1`, `v1.0`), `CHANGELOG.md` y tag `v1.0` en Hugging Face

## 📁 Estructura del Proyecto

```
Actividad_5/
├── datos/
│   ├── datos_ini/        # datos originales
│   └── datos_limp/       # datos limpios
├── fuentes/
│   ├── entrena.ipynb     # notebook de Colab (entrenamiento y evaluación)
│   ├── datos_prep.py     # funciones de limpieza
│   └── train.py          # script de entrenamiento y registro en MLflow
├── README.md             # descripción y guía de uso
└── CHANGELOG.md          # historial de versiones
```

*   **`datos/datos_ini/`:** datos originales tal como llegan, sin modificar. Es la copia de respaldo del dataset de Hugging Face (`v1.0`).
    * `sessions.csv`: una fila por atleta y día entrenado (sintético, 87,258 filas, con duplicados y errores de captura).
    * `athletes.csv`: ficha de los 280 atletas: nivel, sexo, edad e idioma (sintético).
    * `workouts_sugarwod.csv`: exportación real de SugarWOD de un atleta (792 filas, Actividad 4).
*   **`datos/datos_limp/`:** datos después de la limpieza, listos para entrenar. Se generan con `fuentes/datos_prep.py`.
    * `sesiones_limpias.csv`: 81,901 sesiones sin duplicados, con texto estandarizado y sin columnas con fuga de información.
    * `sugarwod_limpio.csv`: exportación de SugarWOD con fechas, tiempos en segundos y `score_type` corregido.
*   **`fuentes/`:** todo el código del proyecto.
    * `entrena.ipynb`: notebook principal. Contiene todo el entrenamiento y la evaluación, con una sección por punto del reporte técnico (A a J).
    * `datos_prep.py`: funciones de carga (Hugging Face o local) y de limpieza. El notebook las importa.
    * `train.py`: versión en script del entrenamiento (secciones E a H del notebook) para correrlo desde la terminal.
*   **`README.md`:** este documento.
*   **`CHANGELOG.md`:** historial de versiones de los datos y del código, ligado a los tags de Git.

## ⚙️ Instrucciones de Uso

1. **Clonar el repositorio e instalar dependencias:**
   ```bash
   git clone https://github.com/rick-lastra/Actividad_5.git
   cd Actividad_5
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pandas scikit-learn xgboost mlflow huggingface_hub matplotlib scipy ipykernel
   ```
2. **Ejecutar en local (VS Code):** abre la carpeta con `code .`, abre `fuentes/entrena.ipynb`, selecciona el kernel `.venv` y ejecuta todas las celdas en orden. Las rutas son relativas al repositorio.
3. **Ejecutar en Google Colab:** da clic en el botón **Abrir en Colab** y luego en *Entorno de ejecución > Ejecutar todas*. La primera celda clona el repositorio e instala las librerías; no hay que subir archivos a mano.
4. **Ejecutar con scripts (opcional):**
   ```bash
   python fuentes/datos_prep.py      # limpia datos_ini -> datos_limp
   python fuentes/train.py           # entrena los 3 modelos y registra en MLflow
   ```
5. **Ver los experimentos en MLflow:** desde la carpeta del repositorio:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```
   y abre http://127.0.0.1:5000. El experimento se llama `actividad5-sobrecarga`.

## 🧪 Metodología
*   **Datos:** 280 atletas sintéticos (novato, intermedio y élite) entre octubre de 2024 y septiembre de 2026. La dificultad de cada WOD se calibró con los resultados del atleta real de la exportación de SugarWOD. El 12.5% de las sesiones termina en sobrecarga al día siguiente.
*   **Limpieza:** se eliminan 1,283 duplicados exactos y 427 recapturas del mismo atleta y fecha (se conserva la más reciente). Se corrigen 2,474 valores mal escritos de `level`, los valores fuera de rango (por ejemplo, RPE fuera de 1 a 10) se convierten en vacíos, y se quitan las 3,647 sesiones sin etiqueta. Resultado: de 87,258 a 81,901 sesiones.
*   **Fuga de información:** se eliminan 14 columnas que describen la misma sesión que se quiere predecir (RPE, minutos, carga, PRs, comentario), porque se conocen después de entrenar.
*   **Versionamiento:** datos originales y limpios marcados con tags de Git (`datos-v1.0`, `datos-v1.1`), dataset fijado en Hugging Face con `revision="v1.0"`, huella SHA-256 de cada archivo y cambios descritos en `CHANGELOG.md`.
*   **Split:** 80% entrenamiento y 20% prueba **por atleta** (`GroupShuffleSplit`), para que un mismo atleta no aparezca en ambos conjuntos. Entrenamiento: 65,403 sesiones de 224 atletas; prueba: 16,498 sesiones de 56 atletas.
*   **Preprocesamiento:** dentro de un `Pipeline` (imputación con la mediana, escalado para la regresión logística y One-Hot para `level`, `sex` y `language`), para que se ajuste solo con datos de entrenamiento.
*   **Ajuste de hiperparámetros:** `GridSearchCV` con `GroupKFold` de 5 folds por atleta y **ROC-AUC** como métrica de selección, porque las clases están desbalanceadas. En total se probaron 24 combinaciones (8 por modelo).
*   **Métricas en prueba:** accuracy, precisión, recall, F1 y ROC-AUC, más matriz de confusión y curva ROC.
*   **Comparación estadística:** los tres modelos se evalúan en los mismos 10 folds; se aplican t pareada y Wilcoxon con corrección de Bonferroni (α = 0.05 / 3).
*   **MLflow:** un run por modelo con sus mejores hiperparámetros, métricas, dataset y el modelo guardado, más un run hijo por cada combinación del Grid Search y un run de comparación estadística.

## 📊 Resultados Obtenidos (prueba, 16,498 sesiones de 56 atletas nuevos)

| Modelo | Mejores hiperparámetros | Accuracy | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Regresión logística | `C=0.1`, `class_weight=None` | 0.890 | 0.649 | 0.266 | 0.378 | **0.859** |
| XGBoost | `max_depth=3`, `learning_rate=0.05`, `scale_pos_weight=1` | 0.889 | 0.618 | 0.305 | 0.408 | 0.855 |
| Random Forest | `max_depth=16`, `min_samples_leaf=25`, `max_features=sqrt` | 0.818 | 0.383 | 0.729 | 0.502 | 0.854 |

**Comparación estadística** (ROC-AUC en 10 folds por atleta):

| Comparación | Diferencia de ROC-AUC | p t pareada | p Wilcoxon | ¿Significativa? |
|---|---|---|---|---|
| Regresión logística vs Random Forest | 0.0003 | 0.750 | 0.432 | No |
| Regresión logística vs XGBoost | 0.0008 | 0.698 | 0.695 | No |
| Random Forest vs XGBoost | 0.0005 | 0.799 | 0.922 | No |

*   Los tres modelos tienen un ROC-AUC de 0.85 a 0.86 y **ninguna diferencia es significativa**, así que se elige el modelo más simple e interpretable: **regresión logística**.
*   Con el umbral de 0.5, la regresión logística y XGBoost tienen buena precisión pero bajo recall (detectan 3 de cada 10 casos de sobrecarga). Random Forest, con pesos balanceados, detecta 7 de cada 10, a cambio de más falsas alarmas.
*   Las variables más relacionadas con la sobrecarga son la carga de los últimos 3 y 7 días y el ACWR por carga.
*   El recall baja con el nivel del atleta (novato 0.34, intermedio 0.26, élite 0.15), porque en los atletas élite la sobrecarga es menos frecuente.

## 🔗 Enlaces
*   Repositorio: https://github.com/rick-lastra/Actividad_5
*   Dataset sintético en Hugging Face (público, versión `v1.0`): https://huggingface.co/datasets/rick-lastra/wodpredict-synth
*   Notebook en Google Colab: https://colab.research.google.com/github/rick-lastra/Actividad_5/blob/main/fuentes/entrena.ipynb
*   Actividad 4 (módulo NLP de fatiga): https://github.com/rick-lastra/WODPredictAI

## 👥 Contacto y Créditos
*   **Autor:** Ricardo Alejandro Lastra Dueñas
*   **Empresa (Caso de Uso):** UnderCover Fitness
*   **Fecha:** Octubre 2026
*   **Institución:** Universidad Tecmilenio (Gestión de Proyectos de IA)
