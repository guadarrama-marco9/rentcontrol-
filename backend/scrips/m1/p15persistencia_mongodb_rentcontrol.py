# ============================================================
# RENTCONTROL - PRACTICA 15
# PERSISTENCIA DEL REPORTE EN MONGODB
# ============================================================

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


RUTA_BASE = Path(__file__).resolve().parents[2]
load_dotenv(RUTA_BASE / ".env")
ARCHIVO_REPORTE = RUTA_BASE / "resultados_p10" / "p14_reporte_inquilinos.csv"
NOMBRE_BASE_DATOS = os.getenv("MONGO_DB", "rentcontrol")
NOMBRE_COLECCION = os.getenv(
    "MONGO_COLLECTION",
    "practica15_persistencia",
)


def leer_reporte():
    if not ARCHIVO_REPORTE.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_REPORTE}. Ejecuta primero "
            "p14motor_recomendaciones_rentcontrol.py."
        )
    reporte = pd.read_csv(ARCHIVO_REPORTE)
    if reporte.empty:
        raise ValueError("El reporte no contiene inquilinos para guardar.")
    return reporte


def convertir_documentos(reporte):
    filas = json.loads(reporte.to_json(orient="records", force_ascii=False))
    fecha = datetime.now(timezone.utc)
    return [
        {
            "id_analisis": str(uuid4()),
            "sistema": "RentControl",
            "practica": 15,
            "fecha_registro": fecha,
            "archivo_origen": ARCHIVO_REPORTE.name,
            "analisis": fila,
        }
        for fila in filas
    ]


def main():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError(
            "Configura MONGO_URI en el entorno. Usa las variables "
            "de backend/.env.example y ejecuta el script manualmente; "
            "esta practica escribe datos en MongoDB."
        )

    documentos = convertir_documentos(leer_reporte())
    with MongoClient(uri, serverSelectionTimeoutMS=5000) as cliente:
        cliente.admin.command("ping")
        coleccion = cliente[NOMBRE_BASE_DATOS][NOMBRE_COLECCION]
        coleccion.create_index("id_analisis", unique=True)
        coleccion.create_index("fecha_registro")
        resultado = coleccion.insert_many(documentos)
        recientes = list(
            coleccion.find(
                {},
                {
                    "_id": 0,
                    "id_analisis": 1,
                    "fecha_registro": 1,
                    "analisis.id_inquilino": 1,
                    "analisis.perfil": 1,
                },
            )
            .sort("fecha_registro", -1)
            .limit(5)
        )

    print("=" * 60)
    print("RENTCONTROL - PRACTICA 15: PERSISTENCIA MONGODB")
    print("=" * 60)
    print("Base de datos:", NOMBRE_BASE_DATOS)
    print("Coleccion:", NOMBRE_COLECCION)
    print("Documentos almacenados:", len(resultado.inserted_ids))
    print("\nUltimos analisis:")
    for registro in recientes:
        print(registro)


if __name__ == "__main__":
    main()
