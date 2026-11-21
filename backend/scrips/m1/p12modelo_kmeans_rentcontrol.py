# ============================================================
# RENTCONTROL - PRACTICA 12
# SEGMENTACION DE INQUILINOS CON K-MEANS
# ============================================================

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RUTA_BASE = Path(__file__).resolve().parents[2]
RUTA_RESULTADOS = RUTA_BASE / "resultados_p10"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"
ARCHIVO_DATOS = RUTA_RESULTADOS / "rentcontrol_datos.csv"
ARCHIVO_MODELO = RUTA_MODELOS / "modelo_kmeans_rentcontrol.joblib"
CARACTERISTICAS = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct",
]
VALORES_K = range(2, 7)
SEMILLA = 42
NOMBRES_PERFIL = [
    "Seguimiento prioritario",
    "Seguimiento preventivo",
    "Comportamiento regular",
    "Buen historial",
    "Historial destacado",
    "Historial sobresaliente",
]


def cargar_datos():
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_DATOS}. Ejecuta primero p10datos_rentcontrol.py."
        )
    datos = pd.read_csv(ARCHIVO_DATOS)
    faltantes = [
        columna for columna in ["id_inquilino", *CARACTERISTICAS]
        if columna not in datos
    ]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    if datos[CARACTERISTICAS].isna().any().any():
        raise ValueError("Las caracteristicas contienen valores nulos.")
    return datos


def puntuar_riesgo(perfil):
    return (
        (100 - perfil["pagos_puntuales_pct"])
        + perfil["dias_retraso_promedio"] * 2
        + perfil["renta_ingreso_pct"] * 0.5
        - perfil["antiguedad_meses"] * 0.1
    )


def main():
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    X = datos[CARACTERISTICAS].astype(float)
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    evaluacion = []
    for k in VALORES_K:
        modelo = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=SEMILLA,
        )
        etiquetas = modelo.fit_predict(X_escalado)
        evaluacion.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(X_escalado, etiquetas),
        })

    tabla = pd.DataFrame(evaluacion)
    tabla.to_csv(
        RUTA_RESULTADOS / "p12_evaluacion_k.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mejor_k = int(tabla.loc[tabla["silhouette"].idxmax(), "k"])

    modelo_final = KMeans(
        n_clusters=mejor_k,
        n_init=20,
        random_state=SEMILLA,
    )
    datos["cluster"] = modelo_final.fit_predict(X_escalado)
    perfiles = datos.groupby("cluster")[CARACTERISTICAS].mean()
    orden_riesgo = sorted(
        perfiles.index,
        key=lambda cluster: puntuar_riesgo(perfiles.loc[cluster]),
        reverse=True,
    )
    mapa_perfiles = {
        int(cluster): NOMBRES_PERFIL[indice]
        for indice, cluster in enumerate(orden_riesgo)
    }
    datos["perfil"] = datos["cluster"].map(mapa_perfiles)
    perfiles.insert(
        0,
        "perfil",
        [mapa_perfiles[int(cluster)] for cluster in perfiles.index],
    )

    datos.to_csv(
        RUTA_RESULTADOS / "p12_inquilinos_segmentados.csv",
        index=False,
        encoding="utf-8-sig",
    )
    perfiles.to_csv(
        RUTA_RESULTADOS / "p12_perfiles.csv",
        encoding="utf-8-sig",
    )
    joblib.dump({
        "modelo": modelo_final,
        "escalador": escalador,
        "caracteristicas": CARACTERISTICAS,
        "mapa_perfiles": mapa_perfiles,
    }, ARCHIVO_MODELO)

    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(tabla["k"], tabla["inercia"], marker="o")
    eje.set(
        title="RENTCONTROL - Metodo del codo",
        xlabel="Numero de grupos (K)",
        ylabel="Inercia",
        xticks=list(VALORES_K),
    )
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(RUTA_RESULTADOS / "p12_metodo_del_codo.png", dpi=180)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(tabla["k"], tabla["silhouette"], marker="o")
    eje.set(
        title="RENTCONTROL - Evaluacion Silhouette",
        xlabel="Numero de grupos (K)",
        ylabel="Silhouette",
        xticks=list(VALORES_K),
    )
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(RUTA_RESULTADOS / "p12_silhouette.png", dpi=180)
    plt.close(figura)

    print("=" * 60)
    print("RENTCONTROL - PRACTICA 12: K-MEANS")
    print("=" * 60)
    print(tabla.round(3).to_string(index=False))
    print("\nK seleccionado por Silhouette:", mejor_k)
    print("\nPerfiles (medias por grupo):")
    print(perfiles.round(2).to_string())
    print("\nSegmentacion y modelo guardados en:", RUTA_RESULTADOS)
    print("Los segmentos sirven para orientar seguimiento, no para negar vivienda.")


if __name__ == "__main__":
    main()
