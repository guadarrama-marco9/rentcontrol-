# ============================================================
# PROYECTO: RENTCONTROL
# PRACTICA 10 - GENERACION DE DATOS PARA ANALISIS
# ARCHIVO: p10datos_rentcontrol.py
#
# Genera datos sinteticos de inquilinos para analizar patrones
# de pago, estimar riesgo de morosidad y preparar los datos para
# clasificacion, K-Means y PCA.
#
# La columna "moroso" es una etiqueta sintetica para fines
# educativos; no debe utilizarse para tomar decisiones reales
# sobre una persona ni para negar una vivienda.
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# 1. CONFIGURACION GENERAL
# ------------------------------------------------------------

SEMILLA = 42
TOTAL_INQUILINOS = 500

RUTA_RESULTADOS = (
    Path(__file__).resolve().parents[2] / "resultados_p10"
)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct",
]


# ------------------------------------------------------------
# 2. GENERACION DE DATOS DE INQUILINOS
# ------------------------------------------------------------

def generar_datos_inquilinos():
    """
    Genera informacion sintetica de comportamiento de pago.

    Variables:
    - pagos_puntuales_pct: porcentaje de pagos realizados a tiempo.
    - dias_retraso_promedio: dias promedio de retraso en el pago.
    - antiguedad_meses: meses de permanencia en el contrato.
    - renta_ingreso_pct: porcentaje del ingreso destinado a la renta.
    - moroso: etiqueta sintetica (1 = mayor riesgo, 0 = menor riesgo).

    Los datos se generan con una semilla fija para poder reproducir
    los resultados de la practica.
    """
    rng = np.random.default_rng(SEMILLA)

    pagos_puntuales_pct = rng.uniform(40, 100, TOTAL_INQUILINOS)
    dias_retraso_promedio = rng.uniform(0, 30, TOTAL_INQUILINOS)
    antiguedad_meses = rng.integers(0, 61, TOTAL_INQUILINOS)
    renta_ingreso_pct = rng.uniform(15, 90, TOTAL_INQUILINOS)

    # Una puntuacion sintetica combina los factores asociados con
    # mayor riesgo. El percentil mantiene aproximadamente 35 % de
    # ejemplos positivos para practicar clasificacion.
    ruido = rng.normal(0, 5, TOTAL_INQUILINOS)
    puntuacion_riesgo = (
        (100 - pagos_puntuales_pct) * 0.6
        + dias_retraso_promedio * 1.5
        + renta_ingreso_pct * 0.25
        - antiguedad_meses * 0.2
        + ruido
    )
    umbral = np.percentile(puntuacion_riesgo, 65)
    moroso = (puntuacion_riesgo >= umbral).astype(int)

    datos = pd.DataFrame({
        "pagos_puntuales_pct": pagos_puntuales_pct,
        "dias_retraso_promedio": dias_retraso_promedio,
        "antiguedad_meses": antiguedad_meses,
        "renta_ingreso_pct": renta_ingreso_pct,
        "moroso": moroso,
    })

    datos[VARIABLES] = datos[VARIABLES].round(2)
    datos.insert(
        0,
        "id_inquilino",
        [f"INQ-{indice:04d}" for indice in range(1, TOTAL_INQUILINOS + 1)],
    )

    return datos


# ------------------------------------------------------------
# 3. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("RENTCONTROL - GENERACION DE DATOS PARA PRACTICA 10")
    print("=" * 60)

    datos = generar_datos_inquilinos()

    print("\nPRIMEROS 10 INQUILINOS:")
    print(datos.head(10))

    print("\nDIMENSIONES DEL DATASET:")
    print(datos.shape)

    print("\nTIPOS DE DATOS:")
    print(datos.dtypes)

    print("\nVALORES NULOS:")
    print(datos.isnull().sum())

    print("\nDISTRIBUCION DE MOROSIDAD (0 = menor riesgo, 1 = mayor riesgo):")
    print(datos["moroso"].value_counts().sort_index())

    print("\nPORCENTAJES:")
    print(
        datos["moroso"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    archivo = RUTA_RESULTADOS / "rentcontrol_datos.csv"
    datos.to_csv(archivo, index=False, encoding="utf-8-sig")

    print("\nARCHIVO GENERADO:")
    print(archivo)
    print("\nPROCESO FINALIZADO CORRECTAMENTE.")


if __name__ == "__main__":
    main()
