# ==============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 21: PROCESAMIENTO MASIVO CON APACHE SPARK
#
# Archivo:
# p21carga_masiva_spark.py
#
# Objetivo:
# Leer los 50,000 registros de cobranza, validarlos, calcular
# nuevas variables y guardarlos en formato Parquet.
#
# Entrada:  resultados_bigdata/rentcontrol_50000.csv
# Salida:   resultados_bigdata/spark_procesado  (Parquet)
#           resultados_bigdata/resumen_spark    (CSV)
#
# Requisitos: Java 17, pyspark. En Windows, Spark puede pedir
# HADOOP_HOME con winutils.exe para escribir archivos locales.
# ==============================================================

import os
import sys
from pathlib import Path

# Usar el mismo Python del entorno virtual en los workers de Spark.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType
)

# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_CSV = RUTA_PROYECTO / "resultados_bigdata" / "rentcontrol_50000.csv"

RUTA_PARQUET = RUTA_PROYECTO / "resultados_bigdata" / "spark_procesado"

RUTA_RESUMEN = RUTA_PROYECTO / "resultados_bigdata" / "resumen_spark"

TOTAL_ESPERADO = 50000
TIPOS_POR_INQUILINO = 5
NUM_PARTICIONES = 8

# Esquema explícito: el orden coincide con el CSV de la práctica 20.
ESQUEMA = StructType([
    StructField("id_registro", LongType(), False),
    StructField("id_inquilino", StringType(), False),
    StructField("tipo_inmueble", StringType(), False),
    StructField("renta_mensual", DoubleType(), False),
    StructField("puntualidad_m1", DoubleType(), False),
    StructField("puntualidad_m2", DoubleType(), False),
    StructField("puntualidad_m3", DoubleType(), False),
    StructField("cobertura_m1", DoubleType(), False),
    StructField("cobertura_m2", DoubleType(), False),
    StructField("cobertura_m3", DoubleType(), False),
    StructField("promedio_puntualidad", DoubleType(), False),
    StructField("tendencia", StringType(), False),
    StructField("resultado", StringType(), False)
])


def crear_sesion_spark():
    """
    local[*] usa todos los procesadores del equipo.
    Se simula procesamiento distribuido en una sola máquina.
    """

    spark = (
        SparkSession.builder
        .appName("RENTCONTROL_CARGA_MASIVA")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(NUM_PARTICIONES))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


def cargar_datos(spark):
    print("\n1. LEYENDO ARCHIVO CSV")
    print("-" * 65)

    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {RUTA_CSV}. "
            "Ejecuta primero p20generador_datos_masivos.py"
        )

    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("encoding", "UTF-8")
        .option("mode", "FAILFAST")
        .schema(ESQUEMA)
        .load(str(RUTA_CSV))
    )

    print("Archivo leído correctamente.")
    df.printSchema()

    return df


def rango_invalido(columnas, minimo, maximo):
    """
    Construye la condición: alguna columna fuera de [minimo, maximo].
    """

    condicion = None

    for nombre in columnas:
        actual = (F.col(nombre) < minimo) | (F.col(nombre) > maximo)

        condicion = actual if condicion is None else (condicion | actual)

    return condicion


def validar_datos(df):
    print("\n2. VALIDANDO INFORMACIÓN")
    print("-" * 65)

    total = df.count()

    print(f"Registros encontrados: {total:,}")
    print(f"Registros esperados: {TOTAL_ESPERADO:,}")

    if total != TOTAL_ESPERADO:
        raise ValueError("La cantidad de registros no coincide.")

    duplicados = total - df.select("id_registro").distinct().count()

    print(f"Registros duplicados: {duplicados}")

    if duplicados > 0:
        raise ValueError("Se encontraron identificadores duplicados.")

    condicion_nulos = None

    for nombre in df.columns:
        actual = F.col(nombre).isNull()

        condicion_nulos = (
            actual if condicion_nulos is None else (condicion_nulos | actual)
        )

    nulos = df.filter(condicion_nulos).count()

    print(f"Registros con valores nulos: {nulos}")

    if nulos > 0:
        raise ValueError("Se encontraron registros con valores nulos.")

    puntualidad_invalida = df.filter(
        rango_invalido(
            [
                "puntualidad_m1",
                "puntualidad_m2",
                "puntualidad_m3",
                "promedio_puntualidad"
            ],
            0,
            10
        )
    ).count()

    print(f"Registros con puntualidad inválida: {puntualidad_invalida}")

    if puntualidad_invalida > 0:
        raise ValueError("Existe puntualidad fuera del rango 0-10.")

    cobertura_invalida = df.filter(
        rango_invalido(
            ["cobertura_m1", "cobertura_m2", "cobertura_m3"],
            0,
            100
        )
    ).count()

    print(f"Registros con cobertura inválida: {cobertura_invalida}")

    if cobertura_invalida > 0:
        raise ValueError("Existe cobertura fuera del rango 0-100.")

    rentas_invalidas = df.filter(F.col("renta_mensual") <= 0).count()

    print(f"Registros con renta inválida: {rentas_invalidas}")

    if rentas_invalidas > 0:
        raise ValueError("Existen rentas menores o iguales a cero.")

    incorrectos = (
        df.groupBy("id_inquilino")
        .agg(F.countDistinct("tipo_inmueble").alias("total_tipos"))
        .filter(F.col("total_tipos") != TIPOS_POR_INQUILINO)
        .count()
    )

    print(f"Inquilinos que no tienen cinco tipos de inmueble: {incorrectos}")

    if incorrectos > 0:
        raise ValueError("Hay inquilinos con cantidad incorrecta de tipos.")

    print("\nVALIDACIÓN COMPLETADA CORRECTAMENTE.")

    return total


def transformar_datos(df):
    """
    Nuevas columnas:
    - cobertura_promedio
    - variacion_puntualidad
    - tendencia_recalculada
    - monto_esperado_trimestre / monto_cobrado_estimado /
      monto_pendiente_estimado
    """

    print("\n3. TRANSFORMANDO DATOS")
    print("-" * 65)

    df = df.withColumn(
        "cobertura_promedio",
        F.round(
            (F.col("cobertura_m1") + F.col("cobertura_m2")
             + F.col("cobertura_m3")) / 3,
            2
        )
    )

    df = df.withColumn(
        "variacion_puntualidad",
        F.round(F.col("puntualidad_m3") - F.col("puntualidad_m1"), 2)
    )

    df = df.withColumn(
        "tendencia_recalculada",
        F.when(F.col("variacion_puntualidad") >= 1, F.lit("Mejora"))
        .when(F.col("variacion_puntualidad") <= -1, F.lit("Deterioro"))
        .otherwise(F.lit("Estable"))
    )

    df = df.withColumn(
        "monto_esperado_trimestre",
        F.round(F.col("renta_mensual") * 3, 2)
    )

    df = df.withColumn(
        "monto_cobrado_estimado",
        F.round(
            F.col("renta_mensual")
            * (F.col("cobertura_m1") + F.col("cobertura_m2")
               + F.col("cobertura_m3")) / 100,
            2
        )
    )

    df = df.withColumn(
        "monto_pendiente_estimado",
        F.round(
            F.col("monto_esperado_trimestre")
            - F.col("monto_cobrado_estimado"),
            2
        )
    )

    print("Nuevas variables calculadas.")

    return df


def analizar_datos(df):
    print("\n4. ANÁLISIS ESTADÍSTICO CON SPARK")
    print("-" * 65)

    print("\nPROMEDIO GENERAL DE PUNTUALIDAD")
    df.select(
        F.round(F.avg("promedio_puntualidad"), 2).alias("promedio_general")
    ).show()

    print("\nPROMEDIO POR TIPO DE INMUEBLE")
    (
        df.groupBy("tipo_inmueble")
        .agg(
            F.round(F.avg("promedio_puntualidad"), 2).alias("puntualidad"),
            F.round(F.avg("renta_mensual"), 2).alias("renta_promedio"),
            F.count("*").alias("total_registros")
        )
        .orderBy("tipo_inmueble")
        .show(truncate=False)
    )

    print("\nDISTRIBUCIÓN DE TENDENCIAS")
    (
        df.groupBy("tendencia")
        .agg(F.count("*").alias("cantidad"))
        .orderBy("tendencia")
        .show()
    )

    print("\nRESULTADOS")
    df.groupBy("resultado").agg(F.count("*").alias("cantidad")).show()

    print("\nCARTERA ESTIMADA (TRIMESTRE)")
    df.agg(
        F.round(F.sum("monto_esperado_trimestre"), 2).alias("esperado"),
        F.round(F.sum("monto_cobrado_estimado"), 2).alias("cobrado"),
        F.round(F.sum("monto_pendiente_estimado"), 2).alias("pendiente")
    ).show(truncate=False)

    print("\nCOMPROBACIÓN DE ETIQUETAS")
    diferencias = df.filter(
        F.col("tendencia") != F.col("tendencia_recalculada")
    ).count()

    print(f"Diferencias entre tendencia original y recalculada: {diferencias}")


def guardar_resultados(df):
    """
    Guarda en Parquet particionado por tipo de inmueble.
    """

    print("\n5. GUARDANDO RESULTADOS")
    print("-" * 65)

    (
        df.repartition(NUM_PARTICIONES, "tipo_inmueble")
        .write
        .mode("overwrite")
        .partitionBy("tipo_inmueble")
        .parquet(str(RUTA_PARQUET))
    )

    print(f"Datos Parquet guardados en: {RUTA_PARQUET}")

    resumen = (
        df.groupBy("tipo_inmueble")
        .agg(
            F.count("*").alias("registros"),
            F.round(F.avg("promedio_puntualidad"), 2).alias("puntualidad"),
            F.round(F.avg("cobertura_promedio"), 2).alias("cobertura"),
            F.round(F.avg("renta_mensual"), 2).alias("renta_promedio")
        )
    )

    (
        resumen.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(str(RUTA_RESUMEN))
    )

    print(f"Resumen guardado en: {RUTA_RESUMEN}")


def main():
    print("\n" + "=" * 65)
    print("RENTCONTROL")
    print("PRÁCTICA 21 - PROCESAMIENTO MASIVO CON APACHE SPARK")
    print("=" * 65)

    spark = None

    try:
        spark = crear_sesion_spark()

        print("\nSESIÓN SPARK INICIADA")
        print("Versión de Spark:", spark.version)

        df = cargar_datos(spark)

        print("\nParticiones iniciales:", df.rdd.getNumPartitions())

        validar_datos(df)

        df = transformar_datos(df)

        df = df.repartition(NUM_PARTICIONES)

        print("\nParticiones de trabajo:", df.rdd.getNumPartitions())

        analizar_datos(df)

        guardar_resultados(df)

        print("\n" + "=" * 65)
        print("PRÁCTICA 21 FINALIZADA CORRECTAMENTE")
        print("=" * 65)

    except Exception as error:
        print("\nERROR DURANTE EL PROCESAMIENTO")
        print("-" * 65)
        print(type(error).__name__, ":", error)
        raise

    finally:
        if spark is not None:
            spark.stop()
            print("\nSESIÓN SPARK CERRADA")


if __name__ == "__main__":
    main()
