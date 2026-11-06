# ============================================================
# PRACTICA 2 - PARTE D
# LIMPIEZA Y EXPORTACION DE DATOS DE RENTCONTROL
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
df_limpio = df.drop_duplicates().copy()

columnas_requeridas = [
    "inquilino", "renta_mensual", "porcentaje_pagado", "dias_atraso"
]
completo = df_limpio[columnas_requeridas].notna().all(axis=1)
renta_valida = df_limpio["renta_mensual"] >= 0
porcentaje_valido = df_limpio["porcentaje_pagado"].between(0, 100)
dias_atraso_validos = df_limpio["dias_atraso"] >= 0

valido = completo & renta_valida & porcentaje_valido & dias_atraso_validos
df_validos = df_limpio.loc[valido].copy()
df_revision = df_limpio.loc[~valido].copy()

print("=== REGISTROS VALIDOS ===")
print(df_validos)
print("\n=== REGISTROS QUE REQUIEREN REVISION ===")
print(df_revision)

df_validos.to_csv("rentcontrol_limpios.csv", index=False, encoding="utf-8")
df_revision.to_csv("rentcontrol_revision.csv", index=False, encoding="utf-8")

print("\nArchivos generados:")
print("- rentcontrol_limpios.csv")
print("- rentcontrol_revision.csv")
