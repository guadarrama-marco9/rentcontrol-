# ============================================================
# PRACTICA 2 - PARTE B
# EXPLORACION DE LOS DATOS DE RENTCONTROL
# ============================================================

import pandas as pd

# La practica A genera el CSV a partir de los datos visibles en RentControl.
df = pd.read_csv("rentcontrol.csv")

print("=== PRIMEROS REGISTROS ===")
print(df.head())

print("\n=== ULTIMOS REGISTROS ===")
print(df.tail())

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== INFORMACION GENERAL ===")
df.info()

print("\n=== ESTADISTICAS DE VARIABLES NUMERICAS ===")
print(df.describe())

print("\n=== INQUILINO Y ATRASO ===")
print(df[["inquilino", "dias_atraso"]])

print("\nPromedio de renta mensual:", round(df["renta_mensual"].mean(), 2))
print("Casos marcados con mora:", int(df["mora"].sum()))

# Las estadisticas describen el estado actual y no predicen casos futuros.
