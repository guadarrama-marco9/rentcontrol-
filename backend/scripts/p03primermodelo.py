# ============================================================
# PRACTICA 03: PREPARAR DATOS PARA CLASIFICACION
# CONTEXTO: REGISTROS ANONIMIZADOS DE RENTCONTROL
# ============================================================

import json
from pathlib import Path

import pandas as pd


columnas = [
    "inquilino",
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

print("=== REGISTROS ACTUALES Y ANONIMIZADOS ===")
print(df.to_string(index=False))
print("\nDimensiones:", df.shape)
print("\nDistribucion del estado actual:")
print(
    df["mora"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(index={0: "Sin atraso vencido", 1: "Con atraso vencido"})
)

print(
    "\nNo se entrena ni se evalua un modelo: estos datos son una "
    "fotografia actual, no un historial independiente."
)
print(
    "La clasificacion describe cargos vencidos y no debe usarse "
    "para tomar decisiones sobre inquilinos."
)
