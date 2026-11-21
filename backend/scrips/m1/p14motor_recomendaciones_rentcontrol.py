# ============================================================
# RENTCONTROL - PRACTICA 14
# ANALISIS INTEGRAL Y RECOMENDACIONES ORIENTATIVAS
# ============================================================

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


RUTA_BASE = Path(__file__).resolve().parents[2]
RUTA_RESULTADOS = RUTA_BASE / "resultados_p10"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"
ARCHIVO_DATOS = RUTA_RESULTADOS / "rentcontrol_datos.csv"
ARCHIVO_SUPERVISADO = RUTA_MODELOS / "modelo_supervisado_rentcontrol.joblib"
ARCHIVO_KMEANS = RUTA_MODELOS / "modelo_kmeans_rentcontrol.joblib"
ARCHIVO_PCA = RUTA_MODELOS / "modelo_pca_rentcontrol.joblib"


def cargar_modelos():
    archivos = [ARCHIVO_SUPERVISADO, ARCHIVO_KMEANS, ARCHIVO_PCA]
    faltantes = [str(archivo) for archivo in archivos if not archivo.is_file()]
    if faltantes:
        raise FileNotFoundError(
            "Ejecuta primero las practicas 11, 12 y 13. "
            "Faltan los modelos: " + ", ".join(faltantes)
        )
    return (
        joblib.load(ARCHIVO_SUPERVISADO),
        joblib.load(ARCHIVO_KMEANS),
        joblib.load(ARCHIVO_PCA),
    )


def recomendar(fila, riesgo_estimado, perfil):
    recomendaciones = []
    if fila["pagos_puntuales_pct"] < 70 or fila["dias_retraso_promedio"] >= 10:
        recomendaciones.append(
            "Ofrecer recordatorios anticipados y un canal de comunicacion claro."
        )
    elif fila["pagos_puntuales_pct"] < 90 or fila["dias_retraso_promedio"] >= 4:
        recomendaciones.append(
            "Mantener seguimiento preventivo y confirmar fechas de pago."
        )
    else:
        recomendaciones.append(
            "Mantener el proceso habitual de comunicacion sobre pagos."
        )

    if fila["renta_ingreso_pct"] >= 40:
        recomendaciones.append(
            "Conversar sobre opciones de apoyo o acuerdos permitidos, "
            "sin asumir la situacion financiera de la persona."
        )
    if fila["antiguedad_meses"] < 6:
        recomendaciones.append(
            "Compartir con claridad fechas, canales y condiciones de pago."
        )
    if riesgo_estimado >= 0.5:
        recomendaciones.append(
            "Priorizar un contacto respetuoso para ofrecer opciones de regularizacion."
        )
    if perfil == "Seguimiento prioritario":
        recomendaciones.append(
            "Revisar el caso de forma individual antes de proponer cualquier accion."
        )

    return " | ".join(dict.fromkeys(recomendaciones))


def main():
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_DATOS}. Ejecuta primero p10datos_rentcontrol.py."
        )
    datos = pd.read_csv(ARCHIVO_DATOS)
    paquete_supervisado, paquete_kmeans, paquete_pca = cargar_modelos()
    caracteristicas = paquete_supervisado["caracteristicas"]
    faltantes = [columna for columna in caracteristicas if columna not in datos]
    if faltantes:
        raise ValueError(f"Faltan caracteristicas requeridas: {faltantes}")

    X = datos[caracteristicas].astype(float)
    modelo_supervisado = paquete_supervisado["modelo"]
    modelo_kmeans = paquete_kmeans["modelo"]
    escalador_kmeans = paquete_kmeans["escalador"]
    mapa_perfiles = paquete_kmeans["mapa_perfiles"]
    modelo_pca = paquete_pca["modelo_pca"]
    escalador_pca = paquete_pca["escalador"]

    predicciones = modelo_supervisado.predict(X)
    probabilidades = modelo_supervisado.predict_proba(X)
    clases = list(modelo_supervisado.classes_)
    indice_riesgo = clases.index(1)
    etiquetas_cluster = modelo_kmeans.predict(
        escalador_kmeans.transform(X)
    )
    coordenadas = modelo_pca.transform(escalador_pca.transform(X))

    reporte = datos[["id_inquilino", *caracteristicas]].copy()
    reporte["riesgo_morosidad_estimado"] = predicciones.astype(int)
    reporte["probabilidad_riesgo"] = probabilidades[:, indice_riesgo]
    reporte["cluster"] = etiquetas_cluster.astype(int)
    reporte["perfil"] = [
        mapa_perfiles[int(cluster)] for cluster in etiquetas_cluster
    ]
    reporte["PC1"] = coordenadas[:, 0]
    reporte["PC2"] = coordenadas[:, 1]
    reporte["recomendaciones"] = [
        recomendar(
            fila,
            float(probabilidades[indice, indice_riesgo]),
            mapa_perfiles[int(etiquetas_cluster[indice])],
        )
        for indice, (_, fila) in enumerate(reporte.iterrows())
    ]
    archivo_reporte = RUTA_RESULTADOS / "p14_reporte_inquilinos.csv"
    reporte.to_csv(archivo_reporte, index=False, encoding="utf-8-sig")

    uri = os.getenv("MONGO_URI")
    if uri:
        from pymongo import MongoClient

        with MongoClient(
            uri,
            serverSelectionTimeoutMS=1500,
        ) as cliente:
            coleccion = cliente[
                os.getenv("MONGO_DB", "rentcontrol")
            ].practica14_recomendaciones
            coleccion.create_index("id_inquilino", unique=True)
            fecha = datetime.now(timezone.utc)
            for fila in reporte.to_dict(orient="records"):
                documento = {
                    "practica": 14,
                    "sistema": "RentControl",
                    "id_inquilino": fila["id_inquilino"],
                    "fecha_actualizacion": fecha,
                    "recomendaciones": fila,
                }
                coleccion.update_one(
                    {"id_inquilino": fila["id_inquilino"]},
                    {"$set": documento},
                    upsert=True,
                )

    print("=" * 60)
    print("RENTCONTROL - PRACTICA 14: MOTOR DE RECOMENDACIONES")
    print("=" * 60)
    print("Inquilinos analizados:", len(reporte))
    print(
        "Casos marcados para seguimiento orientativo:",
        int(reporte["riesgo_morosidad_estimado"].sum()),
    )
    print("\nMuestra del reporte:")
    print(
        reporte[
            [
                "id_inquilino",
                "probabilidad_riesgo",
                "perfil",
                "recomendaciones",
            ]
        ].head(5).to_string(index=False)
    )
    print("\nReporte generado:", archivo_reporte)
    print(
        "Los resultados son orientativos y no deben usarse para "
        "negar o condicionar una vivienda."
    )


if __name__ == "__main__":
    main()
