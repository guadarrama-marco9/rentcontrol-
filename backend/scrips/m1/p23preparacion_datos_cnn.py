# ==============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 23: PREPARACIÓN DE DATOS PARA CNN 1D
#
# Archivo:
# p23preparacion_datos_cnn.py
#
# Objetivo:
# Convertir los registros de cobranza procesados con Spark en
# tensores para una CNN 1D que clasifica la tendencia de pago
# (Deterioro, Estable, Mejora).
#
# Forma de cada muestra: [2 canales, 3 meses]
#   Canal 0: puntualidad de pago.
#   Canal 1: porcentaje de la renta cubierta.
#
# Pasos: leer Parquet -> validar -> tensores -> separar 70/15/15
# POR INQUILINO -> normalizar solo con entrenamiento -> guardar.
#
# NOTA IMPORTANTE:
# La etiqueta de tendencia se calcula con la puntualidad de los
# propios meses de entrada. Una métrica alta no demuestra que el
# modelo pueda predecir el comportamiento futuro de un inquilino.
# ==============================================================

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import GroupShuffleSplit
import torch

# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_PARQUET = RUTA_PROYECTO / "resultados_bigdata" / "spark_procesado"

RUTA_SALIDA = RUTA_PROYECTO / "resultados_cnn"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
REGISTROS_ESPERADOS = 50000
TIPOS_POR_INQUILINO = 5

MAPA_ETIQUETAS = {
    "Deterioro": 0,
    "Estable": 1,
    "Mejora": 2
}

COLUMNAS_PUNTUALIDAD = ["puntualidad_m1", "puntualidad_m2", "puntualidad_m3"]

COLUMNAS_COBERTURA = ["cobertura_m1", "cobertura_m2", "cobertura_m3"]

COLUMNAS_NECESARIAS = (
    ["id_registro", "id_inquilino"]
    + COLUMNAS_PUNTUALIDAD
    + COLUMNAS_COBERTURA
    + ["tendencia"]
)


def crear_sesion_spark():
    spark = (
        SparkSession.builder
        .appName("RentControl_Preparacion_CNN")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def cargar_datos(spark):
    print("\n1. LEYENDO DATOS PARQUET")
    print("-" * 60)

    if not RUTA_PARQUET.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {RUTA_PARQUET}. "
            "Ejecuta primero p21carga_masiva_spark.py"
        )

    datos_spark = spark.read.parquet(str(RUTA_PARQUET))

    faltantes = set(COLUMNAS_NECESARIAS) - set(datos_spark.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas necesarias: {faltantes}")

    datos_spark = datos_spark.select(*COLUMNAS_NECESARIAS)

    total = datos_spark.count()

    print(f"Registros encontrados: {total:,}")

    if total != REGISTROS_ESPERADOS:
        raise ValueError(
            f"Esperados: {REGISTROS_ESPERADOS:,}; encontrados: {total:,}"
        )

    print("Archivos Parquet leídos correctamente.")

    return datos_spark.toPandas()


def validar_datos(datos):
    print("\n2. VALIDANDO CALIDAD DE LOS DATOS")
    print("-" * 60)

    nulos = int(datos[COLUMNAS_NECESARIAS].isnull().sum().sum())

    print(f"Valores nulos: {nulos}")

    if nulos > 0:
        raise ValueError("Se encontraron valores nulos.")

    duplicados = int(datos["id_registro"].duplicated().sum())

    print(f"Identificadores duplicados: {duplicados}")

    if duplicados > 0:
        raise ValueError("Existen identificadores duplicados.")

    puntualidad = datos[COLUMNAS_PUNTUALIDAD].to_numpy(dtype=np.float64)

    if not np.isfinite(puntualidad).all():
        raise ValueError("Hay valores de puntualidad no numéricos.")

    if (puntualidad < 0).any() or (puntualidad > 10).any():
        raise ValueError("Hay puntualidad fuera del rango 0-10.")

    cobertura = datos[COLUMNAS_COBERTURA].to_numpy(dtype=np.float64)

    if not np.isfinite(cobertura).all():
        raise ValueError("Hay valores de cobertura no numéricos.")

    if (cobertura < 0).any() or (cobertura > 100).any():
        raise ValueError("Hay cobertura fuera del rango 0-100.")

    encontradas = set(datos["tendencia"].unique())

    if encontradas != set(MAPA_ETIQUETAS.keys()):
        raise ValueError(
            f"Las categorías de tendencia no coinciden: {encontradas}"
        )

    por_inquilino = datos.groupby("id_inquilino")["id_registro"].count()

    incorrectos = int((por_inquilino != TIPOS_POR_INQUILINO).sum())

    print(
        "Inquilinos con cinco registros: "
        f"{len(por_inquilino) - incorrectos:,}"
    )

    if incorrectos > 0:
        raise ValueError("Hay inquilinos sin exactamente cinco registros.")

    print("Validación de datos completada.")


def construir_tensores(datos):
    """
    X = [50000, 2, 3]  |  y = [50000]
    """

    print("\n3. CONSTRUYENDO TENSORES")
    print("-" * 60)

    puntualidad = datos[COLUMNAS_PUNTUALIDAD].to_numpy(dtype=np.float32)

    cobertura = datos[COLUMNAS_COBERTURA].to_numpy(dtype=np.float32)

    X = np.stack([puntualidad, cobertura], axis=1)

    y = datos["tendencia"].map(MAPA_ETIQUETAS).to_numpy(dtype=np.int64)

    grupos = datos["id_inquilino"].astype(str).to_numpy()

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print("\nEjemplo de un registro:")
    print(X[0])
    print("Etiqueta numérica:", y[0])

    if X.shape != (REGISTROS_ESPERADOS, 2, 3):
        raise ValueError(f"Forma inesperada para X: {X.shape}")

    if y.shape != (REGISTROS_ESPERADOS,):
        raise ValueError(f"Forma inesperada para y: {y.shape}")

    return X, y, grupos


def dividir_datos(X, y, grupos):
    """
    70% entrenamiento, 15% validación, 15% prueba, separados
    por inquilino: un mismo inquilino no aparece en dos conjuntos
    (cada inquilino tiene cinco registros).
    """

    print("\n4. DIVIDIENDO LOS CONJUNTOS")
    print("-" * 60)

    separador_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEMILLA
    )

    idx_train_val, idx_test = next(separador_1.split(X, y, groups=grupos))

    separador_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1764705882,
        random_state=SEMILLA
    )

    idx_train_rel, idx_val_rel = next(
        separador_2.split(
            X[idx_train_val],
            y[idx_train_val],
            groups=grupos[idx_train_val]
        )
    )

    idx_train = idx_train_val[idx_train_rel]
    idx_val = idx_train_val[idx_val_rel]

    conjuntos = {
        "Entrenamiento": idx_train,
        "Validación": idx_val,
        "Prueba": idx_test
    }

    ids = {nombre: set(grupos[idx]) for nombre, idx in conjuntos.items()}

    if (
        ids["Entrenamiento"] & ids["Validación"]
        or ids["Entrenamiento"] & ids["Prueba"]
        or ids["Validación"] & ids["Prueba"]
    ):
        raise ValueError("Hay inquilinos compartidos entre conjuntos.")

    for nombre, idx in conjuntos.items():
        print(
            f"{nombre}: {len(idx):,} registros "
            f"({len(ids[nombre]):,} inquilinos)"
        )

    print("Separación por inquilino verificada.")

    return (
        X[idx_train], y[idx_train], grupos[idx_train],
        X[idx_val], y[idx_val], grupos[idx_val],
        X[idx_test], y[idx_test], grupos[idx_test]
    )


def normalizar_datos(X_train, X_val, X_test):
    """
    Z-score por canal. La media y la desviación se calculan solo
    con entrenamiento para evitar fuga de información.
    """

    print("\n5. NORMALIZANDO LOS DATOS")
    print("-" * 60)

    datos_train = X_train.transpose(0, 2, 1).reshape(-1, 2)

    medias = datos_train.mean(axis=0)

    desviaciones = datos_train.std(axis=0)

    desviaciones = np.where(desviaciones == 0, 1.0, desviaciones)

    def aplicar(X):
        return (
            (X - medias[None, :, None]) / desviaciones[None, :, None]
        ).astype(np.float32)

    print(f"Media puntualidad: {medias[0]:.4f}")
    print(f"Desviación puntualidad: {desviaciones[0]:.4f}")
    print(f"Media cobertura: {medias[1]:.4f}")
    print(f"Desviación cobertura: {desviaciones[1]:.4f}")

    return aplicar(X_train), aplicar(X_val), aplicar(X_test), medias, desviaciones


def mostrar_distribucion(y, nombre):
    print(f"\n{nombre}")

    for etiqueta, codigo in MAPA_ETIQUETAS.items():
        cantidad = int(np.sum(y == codigo))

        print(
            f"{etiqueta:12}: {cantidad:6,} registros "
            f"({cantidad / len(y) * 100:.2f}%)"
        )


def validar_clases(y_train, y_val, y_test):
    print("\n6. DISTRIBUCIÓN DE CLASES")
    print("-" * 60)

    mostrar_distribucion(y_train, "ENTRENAMIENTO")
    mostrar_distribucion(y_val, "VALIDACIÓN")
    mostrar_distribucion(y_test, "PRUEBA")

    esperadas = set(MAPA_ETIQUETAS.values())

    for nombre, conjunto in [
        ("Entrenamiento", y_train),
        ("Validación", y_val),
        ("Prueba", y_test)
    ]:
        if set(np.unique(conjunto).tolist()) != esperadas:
            raise ValueError(f"Faltan categorías en {nombre}.")

    print("\nLas tres clases están presentes en los conjuntos.")


def guardar_datos(
    X_train, y_train, grupos_train,
    X_val, y_val, grupos_val,
    X_test, y_test, grupos_test,
    medias, desviaciones
):
    print("\n7. GUARDANDO LOS RESULTADOS")
    print("-" * 60)

    ruta_datos = RUTA_SALIDA / "rentcontrol_datos_cnn.npz"

    ruta_metadatos = RUTA_SALIDA / "metadatos_cnn.json"

    np.savez_compressed(
        ruta_datos,
        X_train=X_train, y_train=y_train, grupos_train=grupos_train,
        X_val=X_val, y_val=y_val, grupos_val=grupos_val,
        X_test=X_test, y_test=y_test, grupos_test=grupos_test
    )

    metadatos = {
        "proyecto": "RentControl",
        "practica": 23,
        "total_registros": REGISTROS_ESPERADOS,
        "forma_entrada": [2, 3],
        "canales": ["puntualidad_pago", "cobertura_renta"],
        "momentos_temporales": ["mes1", "mes2", "mes3"],
        "mapa_etiquetas": MAPA_ETIQUETAS,
        "media_normalizacion": medias.tolist(),
        "desviacion_normalizacion": desviaciones.tolist(),
        "semilla": SEMILLA,
        "separacion": {
            "entrenamiento": 0.70,
            "validacion": 0.15,
            "prueba": 0.15,
            "metodo": "GroupShuffleSplit por inquilino"
        },
        "framework": "PyTorch",
        "version_pytorch": torch.__version__,
        "cuda_disponible": torch.cuda.is_available()
    }

    with open(ruta_metadatos, "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, indent=4, ensure_ascii=False)

    print(f"Datos guardados: {ruta_datos}")
    print(f"Metadatos guardados: {ruta_metadatos}")


def verificar_pytorch():
    print("\n8. VERIFICANDO PYTORCH")
    print("-" * 60)
    print(f"Versión PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Se utilizará CPU cuando se entrene el modelo.")


def main():
    print("=" * 65)
    print("RENTCONTROL")
    print("PRÁCTICA 23 - PREPARACIÓN DE DATOS PARA CNN 1D")
    print("=" * 65)

    spark = None

    try:
        spark = crear_sesion_spark()

        print("\nSESIÓN SPARK INICIADA")

        datos = cargar_datos(spark)

        validar_datos(datos)

        X, y, grupos = construir_tensores(datos)

        (
            X_train, y_train, g_train,
            X_val, y_val, g_val,
            X_test, y_test, g_test
        ) = dividir_datos(X, y, grupos)

        X_train, X_val, X_test, medias, desviaciones = normalizar_datos(
            X_train, X_val, X_test
        )

        validar_clases(y_train, y_val, y_test)

        guardar_datos(
            X_train, y_train, g_train,
            X_val, y_val, g_val,
            X_test, y_test, g_test,
            medias, desviaciones
        )

        verificar_pytorch()

        print("\n" + "=" * 65)
        print("PRÁCTICA 23 FINALIZADA CORRECTAMENTE")
        print("Los datos están preparados para la CNN 1D.")
        print("=" * 65)

    except Exception as error:
        print("\nERROR EN LA PRÁCTICA 23")
        print(f"Detalle: {error}")
        raise

    finally:
        if spark is not None:
            spark.stop()
            print("\nSESIÓN SPARK CERRADA")


if __name__ == "__main__":
    main()
