# ============================================================
# PRACTICA 2 - PARTE A
# CREACION DE UN CONJUNTO DE DATOS
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Al ejecutar el codigo se crea una tabla de inquilinos ficticios
# y se guarda en un archivo CSV, independiente del programa.
#
# Actividad rapida: agrega mas inquilinos con datos inventados y
# comprueba como cambian las dimensiones de la tabla.
#
# Variables:
# inquilino                Nombre ficticio
# meses_contrato           Antiguedad en el contrato (meses)
# pagos_a_tiempo           Porcentaje de pagos hechos a tiempo
# avisos_atendidos         Porcentaje de avisos/recordatorios atendidos
# dias_retraso_promedio    Dias promedio de retraso al pagar

import pandas as pd

# ------------------------------------------------------------
# PASO 1. CREAR LOS DATOS
# ------------------------------------------------------------
datos = {
    "inquilino": [
        "Ana", "Luis", "Maria", "Carlos", "Sofia",
        "Pedro", "Elena", "Jorge"
    ],
    "meses_contrato": [24, 3, 30, 10, 14, 2, 20, 8],
    "pagos_a_tiempo": [95, 60, 98, 75, 85, 55, 90, 70],
    "avisos_atendidos": [100, 50, 100, 80, 90, 40, 95, 65],
    "dias_retraso_promedio": [0, 9, 0, 4, 2, 12, 1, 6]
}

# ------------------------------------------------------------
# PASO 2. CONVERTIR A DATAFRAME (como una hoja de calculo)
# ------------------------------------------------------------
df = pd.DataFrame(datos)

# ------------------------------------------------------------
# PASO 3. MOSTRAR LA TABLA
# ------------------------------------------------------------
print("=== CONJUNTO DE DATOS ORIGINAL ===")
print(df)

# ------------------------------------------------------------
# PASO 4. CONSULTAR EL TAMANO DE LA TABLA
# ------------------------------------------------------------
print("\nDimensiones de la tabla:")
print(df.shape)
print("Numero de inquilinos:", df.shape[0])
print("Numero de variables:", df.shape[1])

# ------------------------------------------------------------
# PASO 5. GUARDAR EN CSV
# ------------------------------------------------------------
df.to_csv("inquilinos.csv", index=False, encoding="utf-8")

print("\nArchivo inquilinos.csv creado correctamente.")
