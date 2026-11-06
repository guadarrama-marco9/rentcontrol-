# ============================================================
# PRACTICA 2 - PARTE A
# CREACION DEL CONJUNTO DE DATOS DE RENTCONTROL
# ============================================================

import json
from pathlib import Path
import pandas as pd

columnas = [
    "inquilino",
    "tipo_inmueble",
    "renta_mensual",
    "atrasos_previos",
    "porcentaje_pagado",
    "dias_atraso",
    "mora",
]

datos = json.loads(
    Path("rentcontrol_data.json").read_text(encoding="utf-8")
)
df = pd.DataFrame(datos, columns=columnas)

print("=== REGISTROS ANONIMIZADOS DE RENTCONTROL ===")
print(df)
print("\nDimensiones:", df.shape)
print("Numero de registros:", df.shape[0])
print("Numero de variables:", df.shape[1])

df.to_csv("rentcontrol.csv", index=False, encoding="utf-8")
print("\nArchivo temporal rentcontrol.csv creado correctamente.")
