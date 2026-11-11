# ============================================================
# PRACTICA 2 - PARTE C
# DETECCION DE ERRORES EN LOS DATOS
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# ¿Por que limpiar los datos? Imagina que alguien registra por error:
# - Un 110 % de pagos a tiempo.
# - Un dato vacio de dias de retraso.
# - El mismo inquilino dos veces.
# - Un valor numerico escrito como texto.
# Si usamos esos datos sin revisarlos, el analisis puede ser incorrecto
# (por ejemplo, RENTCONTROL podria marcar mal a un inquilino).

import pandas as pd

datos_prueba = {
    "inquilino": ["Ana", "Luis", "Maria", "Luis", "Carlos"],
    "meses_contrato": [24, 3, None, 3, 10],
    "pagos_a_tiempo": [95, 60, 98, 60, 110],
    "dias_retraso_promedio": [0, 9, 0, 9, None]
}

df = pd.DataFrame(datos_prueba)

print("=== DATOS CON PROBLEMAS ===")
print(df)

# 1. VALORES FALTANTES
print("\n=== VALORES FALTANTES POR COLUMNA ===")
print(df.isnull().sum())

# 2. FILAS CON ALGUN DATO FALTANTE
filas_incompletas = df.isnull().any(axis=1).sum()
print("\nFilas con al menos un valor faltante:")
print(filas_incompletas)

# 3. REGISTROS DUPLICADOS
# duplicated() detecta filas completamente repetidas; no identifica
# personas repetidas cuyos otros datos difieren.
print("\n=== FILAS DUPLICADAS ===")
print(df[df.duplicated()])

# 4. RANGOS DE LAS VARIABLES
# pagos_a_tiempo debe estar entre 0 y 100.
# dias_retraso_promedio no puede ser negativo.
pagos_invalidos = (df["pagos_a_tiempo"] < 0) | (df["pagos_a_tiempo"] > 100)

print("\n=== PAGOS A TIEMPO FUERA DE RANGO ===")
print(df[pagos_invalidos])

retraso_invalido = df["dias_retraso_promedio"] < 0

print("\n=== DIAS DE RETRASO NEGATIVOS ===")
print(df[retraso_invalido])

# 5. VALORES NUMERICOS ESCRITOS COMO TEXTO
# to_numeric con errors="coerce" convierte lo no numerico en NaN.
meses_texto = pd.Series(["24", "tres", "30"])
print("\n=== CONVERSION A NUMERO ===")
print(pd.to_numeric(meses_texto, errors="coerce"))
