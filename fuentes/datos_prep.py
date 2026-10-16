"""
datos_prep.py - Funciones de carga y limpieza de datos (Actividad 5, WODPredict-AI).

Lee los datos originales (datos/datos_ini), los limpia y guarda el resultado en datos/datos_limp.
Se puede usar desde el notebook (import datos_prep) o desde la terminal:

    python fuentes/datos_prep.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
CARPETA_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_INI = CARPETA_RAIZ / "datos" / "datos_ini"
CARPETA_LIMP = CARPETA_RAIZ / "datos" / "datos_limp"

# Dataset publicado en Hugging Face. El tag "v1.0" fija la versión de los datos.
HF_DATASET = "rick-lastra/wodpredict-synth"
HF_VERSION = "v1.0"

# Columna que queremos predecir: 1 = el atleta presenta sobrecarga al día siguiente.
ETIQUETA = "overreaching_label"

# Columnas que describen la MISMA sesión que queremos predecir. Se conocen después de entrenar,
# así que usarlas sería hacer trampa (fuga de información). Se eliminan.
COLUMNAS_FUGA = [
    "n_components", "has_metcon", "session_minutes", "rpe", "session_load", "pr_count",
    "overreaching_h1", "overreaching_h3", "overreaching_h7",
    "has_note", "note_id", "prob_fatiga", "source", "captured_at",
]

# Variables de entrada del modelo (todas se conocen ANTES de la sesión).
NUMERICAS = [
    "age", "days_since_prev_session", "sessions_last_3d", "sessions_last_7d", "sessions_last_28d",
    "load_last_3d", "load_last_7d", "mean_session_load_28d",
    "acwr3_freq", "acwr3_load", "acwr_freq", "acwr_load",
    "days_since_first_session", "prev_rpe", "prev_session_load", "prev_has_note",
    "prob_fatiga_prev_session", "prob_fatiga_last_note", "days_since_last_note", "notes_last_28d",
]
CATEGORICAS = ["level", "sex", "language"]

# Reglas de negocio: rango válido de algunas variables. Un valor fuera de rango es un error de captura.
RANGOS_VALIDOS = {
    "prev_rpe": (1, 10),        # la escala de RPE va de 1 a 10
    "age": (16, 80),
    "acwr_freq": (0, 4),
    "acwr_load": (0, 4),
}


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def cargar_csv(nombre_archivo):
    """Lee un CSV de Hugging Face. Si no hay internet, usa la copia de datos/datos_ini."""
    try:
        from huggingface_hub import hf_hub_download
        ruta = hf_hub_download(HF_DATASET, "datos_ini/" + nombre_archivo,
                               repo_type="dataset", revision=HF_VERSION)
        print(f"{nombre_archivo}: leído de Hugging Face ({HF_DATASET}, versión {HF_VERSION})")
    except Exception:
        ruta = CARPETA_INI / nombre_archivo
        print(f"{nombre_archivo}: sin acceso a Hugging Face, se usa la copia local")
    return pd.read_csv(ruta)


# ---------------------------------------------------------------------------
# 2. Limpieza de sesiones y atletas
# ---------------------------------------------------------------------------
def normalizar_nivel(columna):
    """'ELITE ', 'Elite', 'élite' -> 'élite'; 'Novato' -> 'novato'."""
    texto = columna.str.strip().str.lower()            # quitar espacios y pasar a minúsculas
    return texto.replace({"elite": "élite"})           # unificar la palabra con acento


def limpiar_sesiones(sesiones, atletas):
    """Limpia sessions.csv, lo une con athletes.csv y devuelve (tabla limpia, bitácora)."""
    bitacora = []                                      # aquí se anota qué hizo cada paso
    df = sesiones.copy()

    # Paso 1: duplicados exactos (la app sincronizó dos veces la misma fila).
    antes = len(df)
    df = df.drop_duplicates()
    bitacora.append(["Duplicados exactos eliminados", antes - len(df)])

    # Paso 2: duplicados por atleta + fecha (el atleta corrigió su registro).
    # Se ordena por hora de captura y se conserva la última versión.
    antes = len(df)
    df = df.sort_values("captured_at")
    df = df.drop_duplicates(subset=["athlete_id", "date"], keep="last")
    bitacora.append(["Duplicados por atleta + fecha eliminados", antes - len(df)])

    # Paso 3: sesiones sin etiqueta (primeros 28 días de cada atleta, sin historial suficiente).
    antes = len(df)
    df = df[df[ETIQUETA].notna()]
    bitacora.append(["Sesiones sin etiqueta eliminadas", antes - len(df)])

    # Paso 4: estandarizar el texto de 'level'.
    mal_escritos = (~df["level"].isin(["novato", "intermedio", "elite"])).sum()
    df["level"] = normalizar_nivel(df["level"])
    bitacora.append(["Valores de level corregidos", int(mal_escritos)])

    # Paso 5: limpiar la tabla de atletas y unirla (sexo, edad, idioma).
    atl = atletas.copy()
    atl["sex"] = atl["sex"].str.strip().str.upper()    # 'f' -> 'F'
    atl["language"] = atl["language"].str.strip().str.lower()
    atl = atl.drop_duplicates(subset="athlete_id")
    df = df.merge(atl[["athlete_id", "sex", "age", "language"]], on="athlete_id", how="inner")
    bitacora.append(["Atletas sin edad (se imputa la mediana en el modelo)", int(atl["age"].isna().sum())])

    # Paso 6: reglas de negocio. Un valor fuera de rango se convierte en vacío (NaN).
    for columna, (minimo, maximo) in RANGOS_VALIDOS.items():
        fuera = df[columna].notna() & ~df[columna].between(minimo, maximo)
        df.loc[fuera, columna] = np.nan
        bitacora.append([f"{columna} fuera de rango {minimo}-{maximo} -> vacío", int(fuera.sum())])

    # Paso 7: tipos de datos.
    df["date"] = pd.to_datetime(df["date"])
    df[ETIQUETA] = df[ETIQUETA].astype(int)

    # Paso 8: quitar columnas con fuga de información y dejar solo lo necesario.
    df = df.drop(columns=COLUMNAS_FUGA)
    df = df[["session_id", "athlete_id", "date"] + NUMERICAS + CATEGORICAS + [ETIQUETA]]
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    bitacora = pd.DataFrame(bitacora, columns=["paso", "filas o valores afectados"])
    return df, bitacora


# ---------------------------------------------------------------------------
# 3. Limpieza de la exportación real de SugarWOD (Actividad 4)
# ---------------------------------------------------------------------------
def tiempo_a_segundos(texto):
    """'7:08' o '07:08' -> 428 segundos. Si no es un tiempo, devuelve NaN."""
    if not isinstance(texto, str) or ":" not in texto:
        return np.nan
    partes = texto.strip().split(":")
    if not all(p.isdigit() for p in partes):
        return np.nan
    segundos = 0
    for p in partes:                                   # sirve para mm:ss y para hh:mm:ss
        segundos = segundos * 60 + int(p)
    return float(segundos)


def limpiar_sugarwod(sugarwod):
    """Fechas, duplicados, score_type vacío y tiempos en segundos."""
    df = sugarwod.copy()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.drop_duplicates(subset=["date", "title", "best_result_raw"])
    df["tiempo_s"] = df["best_result_display"].apply(tiempo_a_segundos)
    # SugarWOD deja score_type vacío en los WOD "For Time"; si el resultado es un tiempo, es "Time".
    df.loc[df["score_type"].isna() & df["tiempo_s"].notna(), "score_type"] = "Time"
    df["rx"] = (df["rx_or_scaled"] == "RX").astype(int)
    df["pr"] = (df["pr"] == "PR").astype(int)
    df["notes"] = df["notes"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Ejecución desde la terminal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sesiones_ini = cargar_csv("sessions.csv")
    atletas_ini = cargar_csv("athletes.csv")
    sugarwod_ini = cargar_csv("workouts_sugarwod.csv")

    sesiones_limp, bitacora = limpiar_sesiones(sesiones_ini, atletas_ini)
    sugarwod_limp = limpiar_sugarwod(sugarwod_ini)

    CARPETA_LIMP.mkdir(parents=True, exist_ok=True)
    sesiones_limp.to_csv(CARPETA_LIMP / "sesiones_limpias.csv", index=False)
    sugarwod_limp.to_csv(CARPETA_LIMP / "sugarwod_limpio.csv", index=False)

    print(bitacora.to_string(index=False))
    print(f"Sesiones: {len(sesiones_ini):,} originales -> {len(sesiones_limp):,} limpias")
