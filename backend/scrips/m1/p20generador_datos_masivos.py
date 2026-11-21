# ==============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 20: GENERACIÓN DE DATOS DE COBRANZA MASIVOS
#
# Archivo:
# p20generador_datos_masivos.py
#
# Objetivo:
# Generar 50,000 registros sintéticos de cobranza de rentas que
# después se procesan con Apache Spark y se usan para entrenar
# una red neuronal convolucional 1D con PyTorch.
#
# Estructura:
# 10,000 inquilinos x 5 tipos de inmueble = 50,000 registros.
# (Cada inquilino tiene un contrato histórico por cada tipo.)
#
# Cada registro contiene:
# - Tres puntajes de puntualidad de pago (0 a 10), uno por mes.
# - Tres porcentajes de la renta cubierta (0 a 100), uno por mes.
# - Renta mensual, promedio de puntualidad y tendencia.
# - Resultado: "Cumple" o "Riesgo de impago".
#
# IMPORTANTE:
# Los datos son sintéticos y no representan a personas reales.
# ==============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

SEMILLA = 42

rng = np.random.default_rng(SEMILLA)

TOTAL_INQUILINOS = 10000

TIPOS_INMUEBLE = [
    "Departamento",
    "Casa",
    "Cuarto",
    "Local comercial",
    "Bodega"
]

TOTAL_TIPOS = len(TIPOS_INMUEBLE)

TOTAL_REGISTROS = TOTAL_INQUILINOS * TOTAL_TIPOS

# Renta mensual de referencia (MXN) por tipo de inmueble.
RENTA_BASE = {
    "Departamento": 6500,
    "Casa": 9500,
    "Cuarto": 2500,
    "Local comercial": 14000,
    "Bodega": 8000
}

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_RESULTADOS = RUTA_PROYECTO / "resultados_bigdata"

RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

ARCHIVO_CSV = RUTA_RESULTADOS / "rentcontrol_50000.csv"

ARCHIVO_RESUMEN = RUTA_RESULTADOS / "resumen_datos_masivos.csv"


# ==============================================================
# 2. GENERACIÓN DE LOS REGISTROS
# ==============================================================

def generar_dataset():
    """
    Genera todos los registros de forma vectorizada.

    0 = Mejora, 1 = Estable, 2 = Deterioro (patrón de pago).
    """

    # Confiabilidad de cada inquilino (3 a 10). Se repite para
    # sus cinco registros.
    confiabilidad = rng.uniform(3, 10, TOTAL_INQUILINOS)

    conf = np.repeat(confiabilidad, TOTAL_TIPOS)

    ids = np.repeat(np.arange(1, TOTAL_INQUILINOS + 1), TOTAL_TIPOS)

    tipos = np.tile(TIPOS_INMUEBLE, TOTAL_INQUILINOS)

    n = len(conf)

    # Patrón de evolución de cada registro.
    patron = rng.choice([0, 1, 2], size=n, p=[0.40, 0.35, 0.25])

    # Puntualidad del primer mes.
    m1 = np.clip(rng.normal(conf, 0.8), 0, 10)

    cambio = rng.uniform(1.1, 2.2, n)
    ruido_2 = rng.uniform(-0.4, 0.4, n)
    ruido_3 = rng.uniform(-0.4, 0.4, n)

    m2 = np.where(
        patron == 0, m1 + cambio * 0.5,
        np.where(patron == 1, m1 + ruido_2, m1 - cambio * 0.5)
    )

    m3 = np.where(
        patron == 0, m1 + cambio,
        np.where(patron == 1, m1 + ruido_3, m1 - cambio)
    )

    m2 = np.clip(m2, 0, 10)
    m3 = np.clip(m3, 0, 10)

    # Porcentaje de la renta cubierta en cada mes.
    base = 65 + conf * 3.2
    base = base + np.where(patron == 0, 5, np.where(patron == 2, -5, 0))

    cob1 = np.clip(rng.normal(base, 7), 0, 100)
    cob2 = np.clip(rng.normal(base, 7), 0, 100)
    cob3 = np.clip(rng.normal(base, 7), 0, 100)

    # Renta mensual según el tipo de inmueble (múltiplos de $50).
    renta_ref = pd.Series(tipos).map(RENTA_BASE).to_numpy(dtype=float)
    renta = np.clip(rng.normal(renta_ref, renta_ref * 0.20), 1200, 40000)
    renta = np.round(renta / 50) * 50

    datos = pd.DataFrame({
        "id_inquilino": [f"INQ-MASIVO-{i:05d}" for i in ids],
        "tipo_inmueble": tipos,
        "renta_mensual": renta,
        "puntualidad_m1": np.round(m1, 2),
        "puntualidad_m2": np.round(m2, 2),
        "puntualidad_m3": np.round(m3, 2),
        "cobertura_m1": np.round(cob1, 2),
        "cobertura_m2": np.round(cob2, 2),
        "cobertura_m3": np.round(cob3, 2)
    })

    datos["promedio_puntualidad"] = (
        datos[["puntualidad_m1", "puntualidad_m2", "puntualidad_m3"]]
        .mean(axis=1)
        .round(2)
    )

    # Tendencia observada: cambio entre el primer y tercer mes.
    diferencia = (
        datos["puntualidad_m3"] - datos["puntualidad_m1"]
    ).round(2)

    datos["tendencia"] = np.select(
        [diferencia >= 1, diferencia <= -1],
        ["Mejora", "Deterioro"],
        default="Estable"
    )

    # Se usa 7 como puntaje mínimo de cumplimiento en este escenario.
    datos["resultado"] = np.where(
        datos["promedio_puntualidad"] >= 7,
        "Cumple",
        "Riesgo de impago"
    )

    datos.insert(0, "id_registro", np.arange(1, len(datos) + 1))

    return datos


# ==============================================================
# 3. PROGRAMA PRINCIPAL
# ==============================================================

def main():

    print("\n" + "=" * 65)
    print("RENTCONTROL")
    print("PRÁCTICA 20 - GENERACIÓN DE DATOS DE COBRANZA MASIVOS")
    print("=" * 65)

    print("\nCONFIGURACIÓN DEL PROCESO")
    print("-" * 65)
    print(f"Total de inquilinos: {TOTAL_INQUILINOS}")
    print(f"Tipos de inmueble por inquilino: {TOTAL_TIPOS}")
    print(f"Registros esperados: {TOTAL_REGISTROS}")

    print("\nGENERANDO REGISTROS...")

    df = generar_dataset()

    # ----------------------------------------------------------
    # Validaciones
    # ----------------------------------------------------------

    print("\nVALIDANDO INFORMACIÓN")
    print("-" * 65)
    print(f"Registros generados: {len(df):,}")
    print(f"Registros esperados: {TOTAL_REGISTROS:,}")

    assert len(df) == TOTAL_REGISTROS, "ERROR: cantidad de registros."

    assert df["id_registro"].is_unique, "ERROR: identificadores duplicados."

    assert df[
        ["puntualidad_m1", "puntualidad_m2", "puntualidad_m3"]
    ].apply(lambda c: c.between(0, 10).all()).all(), (
        "ERROR: puntualidad fuera de rango."
    )

    assert df[
        ["cobertura_m1", "cobertura_m2", "cobertura_m3"]
    ].apply(lambda c: c.between(0, 100).all()).all(), (
        "ERROR: cobertura fuera de rango."
    )

    assert (
        df.groupby("id_inquilino")["tipo_inmueble"].nunique() == TOTAL_TIPOS
    ).all(), "ERROR: cada inquilino debe tener cinco tipos de inmueble."

    # ----------------------------------------------------------
    # Guardar
    # ----------------------------------------------------------

    print("\nGUARDANDO ARCHIVO CSV...")

    df.to_csv(ARCHIVO_CSV, index=False, encoding="utf-8")

    print(f"Archivo generado: {ARCHIVO_CSV}")

    resumen_tendencia = (
        df["tendencia"]
        .value_counts()
        .rename_axis("tendencia")
        .reset_index(name="cantidad")
    )

    resumen_tendencia["porcentaje"] = (
        resumen_tendencia["cantidad"] / len(df) * 100
    ).round(2)

    resumen_tendencia.to_csv(ARCHIVO_RESUMEN, index=False, encoding="utf-8")

    # ----------------------------------------------------------
    # Resultados en pantalla
    # ----------------------------------------------------------

    print("\nPRIMEROS CINCO REGISTROS")
    print("-" * 65)
    print(df.head().to_string(index=False))

    print("\nDISTRIBUCIÓN DE TENDENCIAS")
    print("-" * 65)
    print(resumen_tendencia.to_string(index=False))

    print("\nDISTRIBUCIÓN DE RESULTADOS")
    print("-" * 65)
    print(df["resultado"].value_counts().to_string())

    print("\nESTADÍSTICAS DE PUNTUALIDAD")
    print("-" * 65)
    print(
        df[
            [
                "puntualidad_m1",
                "puntualidad_m2",
                "puntualidad_m3",
                "promedio_puntualidad"
            ]
        ].describe().round(2)
    )

    print("\n" + "=" * 65)
    print("PRÁCTICA 20 FINALIZADA CORRECTAMENTE")
    print("=" * 65)
    print(f"\nArchivo principal: {ARCHIVO_CSV}")
    print(f"Archivo resumen: {ARCHIVO_RESUMEN}")


if __name__ == "__main__":
    main()
