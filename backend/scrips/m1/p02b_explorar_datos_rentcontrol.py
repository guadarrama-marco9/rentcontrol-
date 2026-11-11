# ============================================================
# PRACTICA 2 - PARTE B
# EXPLORACION DEL CONJUNTO DE DATOS
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# No basta con tener datos. Antes de usarlos hay que conocer su
# estructura, sus tipos y sus caracteristicas estadisticas.

import pandas as pd

df = pd.read_csv("inquilinos.csv")

# 1. PRIMEROS REGISTROS
print("=== PRIMEROS REGISTROS ===")
print(df.head())

# 2. ULTIMOS REGISTROS
print("\n=== ULTIMOS REGISTROS ===")
print(df.tail())

# 3. TIPOS DE DATOS
print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

# 4. INFORMACION GENERAL
print("\n=== INFORMACION GENERAL ===")
df.info()

# 5. ESTADISTICAS DESCRIPTIVAS
print("\n=== ESTADISTICAS DESCRIPTIVAS ===")
print(df.describe())

# 6. CONSULTAR VARIAS COLUMNAS
# NOTA: para elegir varias columnas se usan DOBLES corchetes.
# (En el original, df["a", "b"] provoca KeyError.)
print("\n=== MESES DE CONTRATO POR INQUILINO ===")
print(df[["inquilino", "meses_contrato"]])

# 7. PROMEDIO DE DIAS DE RETRASO
promedio = df["dias_retraso_promedio"].mean()
print("\nPromedio de dias de retraso:", round(promedio, 2))

# Funcion      Para que sirve
# head()       Muestra las primeras filas.
# tail()       Muestra las ultimas filas.
# shape        Devuelve filas y columnas.
# dtypes       Indica el tipo de cada columna.
# info()       Resume estructura y valores no nulos.
# describe()   Calcula estadisticas descriptivas.
# mean()       Calcula la media aritmetica.

# Las estadisticas permiten conocer los datos, pero no demuestran por
# si mismas que una variable cause otra. Que los inquilinos con mas
# pagos a tiempo tengan menos dias de retraso en esta tabla no prueba
# una relacion causal ni garantiza el mismo comportamiento en otros
# grupos de inquilinos.
