# ============================================================
# PRACTICA 2 - PARTE D
# LIMPIEZA BASICA Y EXPORTACION
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Estrategia de limpieza:
# - Eliminar filas completamente duplicadas.
# - Conservar los valores faltantes durante la inspeccion.
# - Marcar registros con valores fuera de rango.
# - Excluir registros incompletos o invalidos del archivo de practica,
#   conservandolos aparte para auditarlos.
#
# En un proyecto real eliminar filas no siempre es lo mejor: depende
# de la cantidad de datos, la causa de los faltantes y el objetivo.

import pandas as pd

datos_prueba = {
    "inquilino": ["Ana", "Luis", "Maria", "Luis", "Carlos"],
    "meses_contrato": [24, 3, None, 3, 10],
    "pagos_a_tiempo": [95, 60, 98, 60, 110],
    "dias_retraso_promedio": [0, 9, 0, 9, None]
}

df = pd.DataFrame(datos_prueba)

# PASO 1. ELIMINAR FILAS COMPLETAMENTE DUPLICADAS
df_limpio = df.drop_duplicates().copy()

# PASO 2. IDENTIFICAR REGISTROS VALIDOS
columnas_requeridas = [
    "inquilino",
    "meses_contrato",
    "pagos_a_tiempo",
    "dias_retraso_promedio"
]

completo = df_limpio[columnas_requeridas].notna().all(axis=1)
pagos_validos = df_limpio["pagos_a_tiempo"].between(0, 100)
retraso_valido = df_limpio["dias_retraso_promedio"] >= 0
meses_validos = df_limpio["meses_contrato"] >= 0

valido = completo & pagos_validos & retraso_valido & meses_validos

# PASO 3. SEPARAR VALIDOS Y A REVISION
df_validos = df_limpio.loc[valido].copy()
df_revision = df_limpio.loc[~valido].copy()

print("=== REGISTROS VALIDOS ===")
print(df_validos)

print("\n=== REGISTROS QUE REQUIEREN REVISION ===")
print(df_revision)

# PASO 4. EXPORTAR
df_validos.to_csv("inquilinos_limpios.csv", index=False, encoding="utf-8")
df_revision.to_csv("inquilinos_revision.csv", index=False, encoding="utf-8")

print("\nArchivos generados:")
print("- inquilinos_limpios.csv")
print("- inquilinos_revision.csv")

# inquilinos_limpios.csv: registros que cumplen las condiciones.
# inquilinos_revision.csv: registros que requieren revision.
# Esta separacion evita que datos problematicos desaparezcan sin
# explicacion. En un sistema profesional tambien convendria registrar
# el motivo de exclusion y revisarlo con el responsable de los datos
# (por ejemplo, el administrador de la propiedad).
