# ==============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 22: PERSISTENCIA MASIVA CON SPARK Y MONGODB LOCAL
#
# Archivo:
# p22persistencia_spark_mongodb.py
#
# Objetivo:
# Almacenar en MongoDB los 50,000 registros de cobranza que
# fueron procesados con Apache Spark (práctica 21).
#
# ORIGEN:  resultados_bigdata/spark_procesado
# DESTINO: base rentcontrol, colección registros_cobranza_masivos
#
# Características:
# - Lectura de archivos Parquet.
# - Escritura por particiones Spark con bulk_write.
# - Upsert (ReplaceOne) para evitar duplicados al repetir la carga.
# - Índices de búsqueda y validación final.
#
# IMPORTANTE:
# Se usa una colección independiente para no mezclar datos
# sintéticos con los datos reales de la aplicación.
# ==============================================================

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pymongo import MongoClient, ReplaceOne
from pyspark.sql import SparkSession

import rc_mongo

# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_PARQUET = RUTA_PROYECTO / "resultados_bigdata" / "spark_procesado"

NOMBRE_COLECCION = "registros_cobranza_masivos"

# Si se repite la carga, los registros del mismo lote se actualizan.
LOTE_ID = "RENTCONTROL-50000-V1"

TAMANO_LOTE = 500
NUM_PARTICIONES = 8
TOTAL_ESPERADO = 50000
INQUILINOS_ESPERADOS = 10000


def conectar_mongodb():
    print("\n1. CONECTANDO CON MONGODB LOCAL")
    print("-" * 65)

    cliente, db = rc_mongo.conectar()

    coleccion = db[NOMBRE_COLECCION]

    coleccion.create_index(
        [("id_lote", 1), ("id_registro", 1)],
        unique=True,
        name="indice_lote_registro_unico"
    )
    coleccion.create_index("id_inquilino", name="indice_inquilino")
    coleccion.create_index("tipo_inmueble", name="indice_tipo_inmueble")

    print("Conexión establecida correctamente.")
    print(f"Base de datos: {db.name}")
    print(f"Colección: {NOMBRE_COLECCION}")
    print("Índices verificados correctamente.")

    return cliente, db, coleccion, rc_mongo.construir_uri()


def crear_sesion_spark():
    spark = (
        SparkSession.builder
        .appName("RENTCONTROL_SPARK_MONGODB")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(NUM_PARTICIONES))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ==============================================================
# 2. ESCRITURA POR PARTICIÓN
# ==============================================================

def insertar_particion(
    filas,
    uri,
    nombre_db,
    nombre_coleccion,
    lote_id,
    tamano_lote
):
    """
    Recibe una partición de Spark y escribe sus registros en
    MongoDB con bulk_write.

    La conexión se crea dentro de cada partición: un MongoClient
    no se comparte entre procesos de Spark.
    """

    cliente = MongoClient(uri, serverSelectionTimeoutMS=20000)

    try:
        coleccion = cliente[nombre_db][nombre_coleccion]

        operaciones = []

        fecha_carga = datetime.now(timezone.utc)

        for fila in filas:

            documento = fila.asDict(recursive=True)

            documento["id_lote"] = lote_id
            documento["sistema"] = "RentControl"
            documento["practica"] = 22
            documento["fecha_carga"] = fecha_carga

            operaciones.append(
                ReplaceOne(
                    {
                        "id_lote": lote_id,
                        "id_registro": documento["id_registro"]
                    },
                    documento,
                    upsert=True
                )
            )

            if len(operaciones) >= tamano_lote:
                coleccion.bulk_write(operaciones, ordered=False)
                operaciones.clear()

        if operaciones:
            coleccion.bulk_write(operaciones, ordered=False)

    finally:
        cliente.close()


def cargar_parquet(spark):
    print("\n2. LEYENDO ARCHIVOS PARQUET")
    print("-" * 65)

    if not RUTA_PARQUET.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {RUTA_PARQUET}. "
            "Ejecuta primero p21carga_masiva_spark.py"
        )

    df = spark.read.parquet(str(RUTA_PARQUET))

    total = df.count()

    print(f"Registros leídos: {total:,}")

    if total != TOTAL_ESPERADO:
        raise ValueError(
            f"Se esperaban {TOTAL_ESPERADO} registros "
            f"y se encontraron {total}."
        )

    print("Archivos Parquet validados.")

    return df


def ejecutar_carga(df, uri, nombre_db):
    print("\n3. INICIANDO CARGA MASIVA")
    print("-" * 65)

    df_distribuido = df.repartition(NUM_PARTICIONES)

    print("Particiones de escritura:", df_distribuido.rdd.getNumPartitions())
    print("Tamaño del lote de escritura:", TAMANO_LOTE)
    print("Identificador de lote:", LOTE_ID)

    # No se usa collect(): los registros se escriben desde
    # las particiones de Spark.
    df_distribuido.foreachPartition(
        lambda filas: insertar_particion(
            filas,
            uri,
            nombre_db,
            NOMBRE_COLECCION,
            LOTE_ID,
            TAMANO_LOTE
        )
    )

    print("Proceso de escritura terminado.")


def validar_persistencia(coleccion):
    print("\n4. VALIDANDO PERSISTENCIA")
    print("-" * 65)

    filtro = {"id_lote": LOTE_ID}

    total = coleccion.count_documents(filtro)

    print(f"Documentos encontrados: {total:,}")
    print(f"Documentos esperados: {TOTAL_ESPERADO:,}")

    if total != TOTAL_ESPERADO:
        raise ValueError(
            "La cantidad de documentos no coincide con lo esperado."
        )

    tipos = coleccion.distinct("tipo_inmueble", filtro)

    print(f"Tipos de inmueble almacenados: {len(tipos)}")

    inquilinos = coleccion.distinct("id_inquilino", filtro)

    print(f"Inquilinos almacenados: {len(inquilinos):,}")

    if len(inquilinos) != INQUILINOS_ESPERADOS:
        raise ValueError("La cantidad de inquilinos no coincide.")

    ejemplo = coleccion.find_one(
        filtro,
        {
            "_id": 0,
            "id_registro": 1,
            "id_inquilino": 1,
            "tipo_inmueble": 1,
            "renta_mensual": 1,
            "promedio_puntualidad": 1,
            "tendencia": 1,
            "resultado": 1,
            "id_lote": 1
        }
    )

    print("\nDOCUMENTO DE EJEMPLO")
    print("-" * 65)
    print(ejemplo)

    print("\nVALIDACIÓN COMPLETADA.")

    return total


def main():
    print("\n" + "=" * 65)
    print("RENTCONTROL")
    print("PRÁCTICA 22 - PERSISTENCIA MASIVA EN MONGODB LOCAL")
    print("=" * 65)

    spark = None
    cliente = None

    try:
        cliente, db, coleccion, uri = conectar_mongodb()

        spark = crear_sesion_spark()

        print("\nSESIÓN SPARK INICIADA")
        print("Versión:", spark.version)

        df = cargar_parquet(spark)

        ejecutar_carga(df, uri, db.name)

        total = validar_persistencia(coleccion)

        print("\n" + "=" * 65)
        print("PRÁCTICA 22 FINALIZADA CORRECTAMENTE")
        print("=" * 65)
        print(f"Registros almacenados: {total:,}")
        print(f"Base de datos: {db.name}")
        print(f"Colección: {NOMBRE_COLECCION}")

    except Exception as error:
        print("\nERROR DURANTE LA PERSISTENCIA")
        print("-" * 65)
        print(type(error).__name__, ":", error)
        raise

    finally:
        if spark is not None:
            spark.stop()
            print("\nSESIÓN SPARK CERRADA")

        if cliente is not None:
            cliente.close()
            print("CONEXIÓN MONGODB CERRADA")


if __name__ == "__main__":
    main()
