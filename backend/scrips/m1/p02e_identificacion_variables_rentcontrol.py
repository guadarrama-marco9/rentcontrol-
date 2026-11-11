# ============================================================
# PRACTICA 2 - PARTE E
# PREPARAR LAS VARIABLES DE ENTRADA Y EL OBJETIVO
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Queremos entrenar un modelo para estimar los dias de retraso de
# pago de un inquilino.
#
# - Variables de entrada (X): meses de contrato, porcentaje de pagos
#   a tiempo y porcentaje de avisos atendidos.
# - Variable objetivo (y): dias_retraso_promedio.
#
# Todavia no entrenamos un modelo; solo preparamos la estructura
# de los datos.
#
# Requiere haber ejecutado antes p02a_crear_datos_rentcontrol.py

import pandas as pd

df = pd.read_csv("inquilinos.csv")

# 1. VARIABLES DE ENTRADA
X = df[
    [
        "meses_contrato",
        "pagos_a_tiempo",
        "avisos_atendidos"
    ]
]

# 2. VARIABLE OBJETIVO
y = df["dias_retraso_promedio"]

# 3. MOSTRAR LOS DATOS
print("=== VARIABLES DE ENTRADA: X ===")
print(X.head())

print("\n=== VARIABLE OBJETIVO: y ===")
print(y.head())

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# 4. COMPROBAR LA CORRESPONDENCIA ENTRE X E y
assert len(X) == len(y), "X e y tienen diferente cantidad de filas"

print("\nLas entradas y el objetivo estan alineados.")

# Un modelo supervisado usa ejemplos en los que se conocen tanto las
# entradas como el resultado esperado. Durante el entrenamiento busca
# relaciones entre ambos.
#
# Cuidado: los datos de esta practica son ficticios y muy pequeños.
# Sirven para aprender el procedimiento, no para crear un predictor
# fiable de morosidad en RENTCONTROL.
