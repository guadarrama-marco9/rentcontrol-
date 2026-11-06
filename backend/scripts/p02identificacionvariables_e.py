# ============================================================
# PRACTICA 2 - PARTE E
# PREPARAR VARIABLES PARA UN ANALISIS DE RENTCONTROL
# ============================================================

# Prepararemos datos para explorar si hay pagos atrasados.
# En un producto real, un modelo no debe reemplazar la revision
# humana ni usarse como unico criterio para decisiones importantes.

import pandas as pd

# La practica A genera el CSV a partir de los datos visibles en RentControl.
df = pd.read_csv("rentcontrol.csv")

X = df[
    [
        "renta_mensual",
        "atrasos_previos",
        "porcentaje_pagado",
        "dias_atraso"
    ]
]
y = df["mora"]

print("=== VARIABLES DE ENTRADA: X ===")
print(X.head())
print("\n=== VARIABLE OBJETIVO: y (mora registrada) ===")
print(y.head())
print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

assert len(X) == len(y), "X e y tienen diferente cantidad de filas"
print("\nLas entradas y el objetivo estan alineados.")

# Son datos actuales y anonimizados; no sirven para evaluar un predictor.
