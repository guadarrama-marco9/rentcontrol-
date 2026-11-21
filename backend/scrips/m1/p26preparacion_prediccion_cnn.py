# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 26: PREPARACIÓN PARA PREDECIR LA PUNTUALIDAD DEL MES 3
#
# Archivo:
# p26preparacion_prediccion_cnn.py
#
# Objetivo:
# Preparar un problema de REGRESIÓN: predecir el puntaje de
# puntualidad del mes 3 usando solo los dos primeros meses.
#
# Entrada X: [2 canales, 2 meses]
#   Canal 0: puntualidad mes 1 y mes 2.
#   Canal 1: cobertura de renta mes 1 y mes 2.
# Objetivo y: puntualidad del mes 3.
#
# El mes 3 NO forma parte de X (se evita la fuga de información).
# Separación 70/15/15 por inquilino; normalización solo con
# estadísticos del conjunto de entrenamiento.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

ARCHIVO_CSV = RUTA_PROYECTO / "resultados_bigdata" / "rentcontrol_50000.csv"

RUTA_SALIDA = RUTA_PROYECTO / "resultados_prediccion_cnn"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_NPZ = RUTA_SALIDA / "rentcontrol_prediccion_mes3.npz"

ARCHIVO_JSON = RUTA_SALIDA / "metadatos_prediccion.json"

SEMILLA = 42

PROPORCION_ENTRENAMIENTO = 0.70


def mostrar_titulo(mensaje):
    print("\n" + "=" * 65)
    print(mensaje)
    print("=" * 65)


mostrar_titulo("RENTCONTROL - PRÁCTICA 26 (REGRESIÓN)")

# ------------------------------------------------------------
# 1. CARGAR Y VALIDAR EL DATASET
# ------------------------------------------------------------

print("\nCARGANDO DATASET ORIGINAL...")

if not ARCHIVO_CSV.exists():
    raise FileNotFoundError(
        f"No existe el archivo: {ARCHIVO_CSV}. "
        "Ejecuta primero p20generador_datos_masivos.py"
    )

datos = pd.read_csv(ARCHIVO_CSV)

print(f"Registros encontrados: {len(datos):,}")
print(f"Columnas encontradas: {len(datos.columns)}")

columnas_requeridas = [
    "id_registro",
    "id_inquilino",
    "tipo_inmueble",
    "puntualidad_m1",
    "puntualidad_m2",
    "puntualidad_m3",
    "cobertura_m1",
    "cobertura_m2",
    "cobertura_m3"
]

faltantes = [c for c in columnas_requeridas if c not in datos.columns]

if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

mostrar_titulo("VALIDACIÓN DE INTEGRIDAD")

if datos["id_inquilino"].isnull().any():
    raise ValueError("Existen registros sin identificador de inquilino.")

if datos["tipo_inmueble"].isnull().any():
    raise ValueError("Existen registros sin tipo de inmueble.")

if datos["id_registro"].duplicated().any():
    raise ValueError("Se encontraron identificadores duplicados.")

if datos.duplicated(subset=["id_inquilino", "tipo_inmueble"]).any():
    raise ValueError("Existen tipos de inmueble duplicados por inquilino.")

if not (datos.groupby("id_inquilino")["tipo_inmueble"].nunique() == 5).all():
    raise ValueError("No todos los inquilinos tienen cinco tipos de inmueble.")

print(f"Inquilinos encontrados: {datos['id_inquilino'].nunique():,}")

columnas_numericas = [
    "puntualidad_m1", "puntualidad_m2", "puntualidad_m3",
    "cobertura_m1", "cobertura_m2", "cobertura_m3"
]

for columna in columnas_numericas:
    datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

if datos[columnas_numericas].isnull().any().any():
    raise ValueError("Existen valores nulos o no numéricos.")

if not np.isfinite(datos[columnas_numericas].to_numpy()).all():
    raise ValueError("Se encontraron valores infinitos.")

for columna in ["puntualidad_m1", "puntualidad_m2", "puntualidad_m3"]:
    if not datos[columna].between(0, 10).all():
        raise ValueError(f"Puntualidad fuera de rango en {columna}")

for columna in ["cobertura_m1", "cobertura_m2", "cobertura_m3"]:
    if not datos[columna].between(0, 100).all():
        raise ValueError(f"Cobertura fuera de rango en {columna}")

print("Datos válidos.")

# ------------------------------------------------------------
# 2. CONSTRUIR X (2 canales x 2 meses) Y EL OBJETIVO y
# ------------------------------------------------------------

mostrar_titulo("CONSTRUCCIÓN DEL CONJUNTO DE PREDICCIÓN")

X = np.zeros((len(datos), 2, 2), dtype=np.float32)

X[:, 0, 0] = datos["puntualidad_m1"].to_numpy()
X[:, 0, 1] = datos["puntualidad_m2"].to_numpy()
X[:, 1, 0] = datos["cobertura_m1"].to_numpy()
X[:, 1, 1] = datos["cobertura_m2"].to_numpy()

y = datos["puntualidad_m3"].to_numpy(dtype=np.float32)

grupos = datos["id_inquilino"].to_numpy()

tipos = datos["tipo_inmueble"].to_numpy()

print(f"Entradas X: {X.shape}")
print(f"Objetivo y: {y.shape}")
print("IMPORTANTE: la puntualidad ni la cobertura del mes 3 forman parte de X.")

# ------------------------------------------------------------
# 3. SEPARACIÓN POR INQUILINO
# ------------------------------------------------------------

mostrar_titulo("SEPARACIÓN DE ENTRENAMIENTO, VALIDACIÓN Y PRUEBA")

separador_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=1 - PROPORCION_ENTRENAMIENTO,
    random_state=SEMILLA
)

idx_train, idx_temporal = next(separador_1.split(X, y, groups=grupos))

separador_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEMILLA
)

idx_val_rel, idx_test_rel = next(
    separador_2.split(
        idx_temporal,
        y[idx_temporal],
        groups=grupos[idx_temporal]
    )
)

idx_val = idx_temporal[idx_val_rel]
idx_test = idx_temporal[idx_test_rel]

ids_train = set(grupos[idx_train])
ids_val = set(grupos[idx_val])
ids_test = set(grupos[idx_test])

assert ids_train.isdisjoint(ids_val)
assert ids_train.isdisjoint(ids_test)
assert ids_val.isdisjoint(ids_test)

print("Separación por inquilino correcta.")
print(f"Inquilinos entrenamiento: {len(ids_train):,}")
print(f"Inquilinos validación:    {len(ids_val):,}")
print(f"Inquilinos prueba:        {len(ids_test):,}")
print(
    f"Registros: {len(idx_train):,} / {len(idx_val):,} / {len(idx_test):,}"
)

X_train_raw, X_val_raw, X_test_raw = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

# ------------------------------------------------------------
# 4. NORMALIZACIÓN (solo con entrenamiento)
# ------------------------------------------------------------

mostrar_titulo("NORMALIZACIÓN DE VARIABLES")

media_entrenamiento = X_train_raw.mean(axis=(0, 2), keepdims=True)

desviacion_entrenamiento = X_train_raw.std(axis=(0, 2), keepdims=True)

desviacion_entrenamiento = np.where(
    desviacion_entrenamiento == 0,
    1.0,
    desviacion_entrenamiento
)

X_train = ((X_train_raw - media_entrenamiento) / desviacion_entrenamiento).astype(np.float32)
X_val = ((X_val_raw - media_entrenamiento) / desviacion_entrenamiento).astype(np.float32)
X_test = ((X_test_raw - media_entrenamiento) / desviacion_entrenamiento).astype(np.float32)

print(f"Media puntualidad: {media_entrenamiento[0, 0, 0]:.4f}")
print(f"Desviación puntualidad: {desviacion_entrenamiento[0, 0, 0]:.4f}")
print(f"Media cobertura: {media_entrenamiento[0, 1, 0]:.4f}")
print(f"Desviación cobertura: {desviacion_entrenamiento[0, 1, 0]:.4f}")

# ------------------------------------------------------------
# 5. VALIDACIÓN FINAL DE TENSORES
# ------------------------------------------------------------

mostrar_titulo("VALIDACIÓN FINAL DE TENSORES")

for nombre, Xc, yc in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test)
]:
    assert Xc.ndim == 3 and Xc.shape[1:] == (2, 2)
    assert yc.ndim == 1 and len(Xc) == len(yc)
    assert np.isfinite(Xc).all() and np.isfinite(yc).all()
    print(f"X_{nombre}: {Xc.shape} | y_{nombre}: {yc.shape}")

mostrar_titulo("ANÁLISIS DE LA PUNTUALIDAD DEL MES 3")

print(f"Promedio entrenamiento: {y_train.mean():.4f}")
print(f"Promedio validación: {y_val.mean():.4f}")
print(f"Promedio prueba: {y_test.mean():.4f}")
print(f"Mínimo: {y.min():.2f} | Máximo: {y.max():.2f}")
print(f"Desviación estándar: {y.std():.4f}")

# ------------------------------------------------------------
# 6. GUARDAR
# ------------------------------------------------------------

mostrar_titulo("GUARDANDO DATOS PARA PYTORCH")

np.savez_compressed(
    ARCHIVO_NPZ,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    ids_train=grupos[idx_train].astype(str),
    ids_val=grupos[idx_val].astype(str),
    ids_test=grupos[idx_test].astype(str),
    tipos_train=tipos[idx_train].astype(str),
    tipos_val=tipos[idx_val].astype(str),
    tipos_test=tipos[idx_test].astype(str),
    media_entrenamiento=media_entrenamiento,
    desviacion_entrenamiento=desviacion_entrenamiento
)

print(f"Archivo NPZ guardado: {ARCHIVO_NPZ}")

metadatos = {
    "proyecto": "RentControl",
    "practica": 26,
    "tipo_problema": "Regresión",
    "objetivo": "Predicción de la puntualidad de pago del mes 3",
    "total_registros": int(len(datos)),
    "total_inquilinos": int(datos["id_inquilino"].nunique()),
    "tipos_por_inquilino": 5,
    "variables_entrada": [
        "puntualidad_m1",
        "puntualidad_m2",
        "cobertura_m1",
        "cobertura_m2"
    ],
    "variable_objetivo": "puntualidad_m3",
    "variables_excluidas": [
        "puntualidad_m3 como entrada",
        "cobertura_m3",
        "promedio_puntualidad",
        "tendencia",
        "resultado",
        "id_inquilino",
        "tipo_inmueble"
    ],
    "forma_entrada": [2, 2],
    "separacion": {
        "entrenamiento": 0.70,
        "validacion": 0.15,
        "prueba": 0.15,
        "metodo": "GroupShuffleSplit",
        "grupo": "id_inquilino"
    },
    "semilla": SEMILLA,
    "normalizacion": {
        "metodo": "Z-score",
        "calculada_con": "Solo entrenamiento",
        "media": media_entrenamiento.tolist(),
        "desviacion": desviacion_entrenamiento.tolist()
    },
    "advertencia": (
        "Los datos son sintéticos. Los resultados son una validación "
        "experimental, no evidencia de rendimiento con datos reales."
    )
}

with open(ARCHIVO_JSON, "w", encoding="utf-8") as archivo:
    json.dump(metadatos, archivo, indent=4, ensure_ascii=False)

print(f"Metadatos guardados: {ARCHIVO_JSON}")

mostrar_titulo("PRÁCTICA 26 FINALIZADA")

print(f"Entrenamiento: {len(X_train):,}")
print(f"Validación: {len(X_val):,}")
print(f"Prueba: {len(X_test):,}")
print("\nENTRADA DE LA CNN: (2 canales, 2 meses)")
print("OBJETIVO: puntualidad de pago del mes 3")
print("NO SE DETECTÓ FUGA DE INFORMACIÓN POR INQUILINO.")
print("=" * 65)
