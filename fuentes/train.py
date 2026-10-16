"""
train.py - Entrenamiento de 3 modelos con Grid Search y registro en MLflow (Actividad 5).

Hace lo mismo que las secciones E a H del notebook, pero desde la terminal:

    python fuentes/datos_prep.py      # primero: genera datos/datos_limp
    python fuentes/train.py           # entrena y registra en MLflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # abrir http://127.0.0.1:5000
"""
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from datos_prep import CARPETA_LIMP, CARPETA_RAIZ, CATEGORICAS, ETIQUETA, HF_DATASET, HF_VERSION, NUMERICAS

SEMILLA = 42
EXPERIMENTO = "actividad5-sobrecarga"


def crear_preprocesador(escalar):
    """Imputa vacíos con la mediana, escala (opcional) y convierte categorías en columnas 0/1."""
    pasos_numericos = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        pasos_numericos.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(pasos_numericos), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def definir_modelos(proporcion_negativos):
    """Cada modelo: pipeline (preprocesamiento + clasificador) y rejilla de hiperparámetros."""
    return {
        "regresion_logistica": (
            Pipeline([("prep", crear_preprocesador(escalar=True)),
                      ("modelo", LogisticRegression(max_iter=3000, random_state=SEMILLA))]),
            {"modelo__C": [0.01, 0.1, 1, 10],
             "modelo__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            Pipeline([("prep", crear_preprocesador(escalar=False)),
                      ("modelo", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=SEMILLA))]),
            {"modelo__max_depth": [8, 16],
             "modelo__min_samples_leaf": [5, 25],
             "modelo__max_features": ["sqrt", 0.5]},
        ),
        "xgboost": (
            Pipeline([("prep", crear_preprocesador(escalar=False)),
                      ("modelo", XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                                               eval_metric="logloss", n_jobs=-1, random_state=SEMILLA))]),
            {"modelo__max_depth": [3, 5],
             "modelo__learning_rate": [0.05, 0.1],
             "modelo__scale_pos_weight": [1, round(proporcion_negativos, 2)]},
        ),
    }


def metricas_test(y_real, probabilidad):
    """Las 5 métricas que pide la actividad. Se predice 'sobrecarga' si la probabilidad >= 0.5."""
    prediccion = (probabilidad >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_real, prediccion),
        "precision": precision_score(y_real, prediccion),
        "recall": recall_score(y_real, prediccion),
        "f1": f1_score(y_real, prediccion),
        "roc_auc": roc_auc_score(y_real, probabilidad),
    }


def main():
    # 1. Datos limpios y separación 80/20 POR ATLETA (un atleta no puede estar en train y en test).
    datos = pd.read_csv(CARPETA_LIMP / "sesiones_limpias.csv")
    X = datos[NUMERICAS + CATEGORICAS]
    y = datos[ETIQUETA]
    separador = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    idx_train, idx_test = next(separador.split(X, y, groups=datos["athlete_id"]))
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    grupos_train = datos["athlete_id"].iloc[idx_train]

    # 2. MLflow local: la información queda en mlflow.db y los archivos en mlruns/.
    mlflow.set_tracking_uri(f"sqlite:///{CARPETA_RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)

    # 3. Grid Search con validación cruzada (5 folds por atleta) para cada modelo.
    proporcion = (y_train == 0).sum() / (y_train == 1).sum()
    mejores = {}
    for nombre, (pipeline, rejilla) in definir_modelos(proporcion).items():
        print("Entrenando", nombre)
        busqueda = GridSearchCV(pipeline, rejilla, cv=GroupKFold(n_splits=5), scoring="roc_auc")
        with mlflow.start_run(run_name=nombre):
            busqueda.fit(X_train, y_train, groups=grupos_train)

            # Un run "hijo" por cada combinación probada.
            resultados = pd.DataFrame(busqueda.cv_results_)
            for i in range(len(resultados)):
                with mlflow.start_run(run_name=f"{nombre}_{i + 1}", nested=True):
                    mlflow.log_params(resultados.loc[i, "params"])
                    mlflow.log_metric("cv_roc_auc", resultados.loc[i, "mean_test_score"])

            # En el run del modelo: mejores hiperparámetros, métricas y el modelo entrenado.
            probabilidad = busqueda.predict_proba(X_test)[:, 1]
            mlflow.log_params(busqueda.best_params_)
            mlflow.log_param("dataset", f"{HF_DATASET}@{HF_VERSION}")
            mlflow.log_metric("cv_roc_auc", busqueda.best_score_)
            for metrica, valor in metricas_test(y_test, probabilidad).items():
                mlflow.log_metric("test_" + metrica, valor)
            mlflow.sklearn.log_model(busqueda.best_estimator_, name="modelo",
                                     serialization_format="cloudpickle",
                                     signature=infer_signature(X_test, probabilidad))
        mejores[nombre] = busqueda.best_estimator_
        print("  mejores hiperparámetros:", busqueda.best_params_)

    # 4. Comparación estadística: mismos 10 folds para los 3 modelos, t pareada y Wilcoxon.
    puntajes = {}
    for nombre, modelo in mejores.items():
        puntajes[nombre] = cross_val_score(modelo, X_train, y_train, groups=grupos_train,
                                           cv=GroupKFold(n_splits=10), scoring="roc_auc")
    with mlflow.start_run(run_name="comparacion_estadistica"):
        nombres = list(puntajes)
        for i in range(len(nombres)):
            for j in range(i + 1, len(nombres)):
                a, b = nombres[i], nombres[j]
                p_t = stats.ttest_rel(puntajes[a], puntajes[b]).pvalue
                p_w = stats.wilcoxon(puntajes[a], puntajes[b]).pvalue
                mlflow.log_metric(f"p_t_{a}_vs_{b}", p_t)
                mlflow.log_metric(f"p_wilcoxon_{a}_vs_{b}", p_w)
                print(f"{a} vs {b}: p t pareada = {p_t:.3f} | p Wilcoxon = {p_w:.3f}")


if __name__ == "__main__":
    main()
