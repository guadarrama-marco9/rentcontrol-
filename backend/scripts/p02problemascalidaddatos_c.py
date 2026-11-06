# ============================================================
# PRACTICA 2 - PARTE C
# DETECCION DE PROBLEMAS EN DATOS DE RENTCONTROL
# ============================================================

import json
from pathlib import Path
import pandas as pd

columnas = [
    "inquilino",
    "renta_mensual",
    "porcentaje_pagado",
    "dias_atraso",
]

datos = json.loads(
    Path("rentcontrol_data.json").read_text(encoding="utf-8")
)
df = pd.DataFrame(datos, columns=columnas)
print("=== DATOS CON POSIBLES PROBLEMAS ===")
print(df)

print("\n=== VALORES FALTANTES POR COLUMNA ===")
print(df.isnull().sum())
print("\nFilas con al menos un dato faltante:", int(df.isnull().any(axis=1).sum()))

print("\n=== FILAS COMPLETAMENTE DUPLICADAS ===")
print(df[df.duplicated()])

porcentaje_invalido = ~df["porcentaje_pagado"].between(0, 100)
print("\n=== PORCENTAJES PAGADOS FUERA DE RANGO ===")
print(df[porcentaje_invalido])

renta_invalida = df["renta_mensual"].notna() & (df["renta_mensual"] < 0)
print("\n=== RENTAS NEGATIVAS ===")
print(df[renta_invalida])

# En un sistema real, los datos financieros deben validarse y
# corregirse con una fuente autorizada, no modificarse a ciegas.
