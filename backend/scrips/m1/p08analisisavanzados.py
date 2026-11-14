# ============================================================
# UNIDAD III - FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# PRACTICA 8:
# ANALISIS AVANZADO Y OPTIMIZACION DE SEGMENTOS DE INQUILINOS
#
# CASO: RENTCONTROL
#
# OBJETIVOS:
# 1. Comparar inicializaciones k-means++ y random.
# 2. Comparar StandardScaler y MinMaxScaler.
# 3. Evaluar estabilidad mediante Adjusted Rand Index (ARI).
# 4. Detectar inquilinos cuyo comportamiento esta lejos del
#    centro de su segmento (candidatos a revision).
# 5. Analizar centroides y distancias.
# 6. Exportar resultados (CSV y, opcionalmente, MongoDB local).
#
# Continua el trabajo de la Practica 7: tres perfiles de
# inquilinos (inquilino puntual, pago regular, riesgo de
# morosidad). Aqui se comprueba que tan confiables son.
#
# IMPORTANTE: estar lejos del centroide NO significa que la
# persona incumpla ni que este mal clasificada. Solo indica que
# su comportamiento es distinto al tipico de su grupo; es un
# aviso para que el administrador lo revise.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score
)

from rc_datos import (
    VARIABLES,
    generar_inquilinos,
    guardar_en_mongo
)


# ------------------------------------------------------------
# 2. CONFIGURACION GENERAL
# ------------------------------------------------------------

NUM_CLUSTERS = 3

SEMILLA = 42

NUM_INICIALIZACIONES = 20

# Se marcan los que superan el percentil 95 de distancias.
PERCENTIL_ATIPICOS = 95

# Cambia a True para guardar el reporte en MongoDB local
# (base rentcontrol, coleccion segmentos_inquilinos).
GUARDAR_EN_MONGO = False

COLECCION_SEGMENTOS = "segmentos_inquilinos"

CARPETA_RESULTADOS = "resultados_p08"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def ruta(nombre):
    return f"{CARPETA_RESULTADOS}/{nombre}"


# ------------------------------------------------------------
# 3. CARGA DE DATOS
# ------------------------------------------------------------

def cargar_datos():

    """
    Intenta recuperar el CSV generado en P7.
    Si no existe, genera datos equivalentes.

    Se eliminan cluster y perfil para no usarlos como entrada.
    """

    ruta_csv = "p07_inquilinos_segmentados.csv"

    if os.path.exists(ruta_csv):

        print("\nArchivo de P7 encontrado.")

        df = pd.read_csv(ruta_csv)

        faltantes = [c for c in VARIABLES if c not in df.columns]

        if faltantes:
            raise ValueError(
                "El archivo de P7 no contiene: " + str(faltantes)
            )

        df = df.drop(columns=["cluster", "perfil"], errors="ignore")

        if "id_inquilino" not in df.columns:
            df.insert(
                0,
                "id_inquilino",
                [f"INQ-{i:04d}" for i in range(1, len(df) + 1)]
            )

        return df

    print("\nNo se encontro el CSV de P7.")
    print("Se generara un conjunto de datos equivalente.")

    return generar_inquilinos(SEMILLA)


# ------------------------------------------------------------
# 4. PREPARACION DE LOS DATOS
# ------------------------------------------------------------

def preparar_datos(df):

    """
    K-Means necesita datos numericos y no admite NaN.
    Los faltantes se sustituyen por la mediana (en un proyecto
    real esta decision debe justificarse).
    """

    print("\n" + "=" * 60)
    print("PREPARACION DE DATOS")
    print("=" * 60)

    print("\nNumero de inquilinos:", len(df))

    print("\nValores faltantes por variable:")

    print(df[VARIABLES].isnull().sum())

    for columna in VARIABLES:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df[VARIABLES] = df[VARIABLES].fillna(df[VARIABLES].median())

    X = df[VARIABLES].to_numpy()

    return df, X


# ------------------------------------------------------------
# 5. COMPARACION DE INICIALIZACIONES
# ------------------------------------------------------------

def comparar_inicializaciones(X):

    """
    k-means++ : centroides iniciales separados entre si.
    random    : centroides iniciales al azar.

    Escalamiento, K y n_init se mantienen constantes.
    """

    print("\n" + "=" * 60)
    print("COMPARACION DE INICIALIZACIONES")
    print("=" * 60)

    X_escalado = StandardScaler().fit_transform(X)

    resultados = []

    for inicializacion in ["k-means++", "random"]:

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init=inicializacion,
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_escalado)

        silhouette = silhouette_score(X_escalado, etiquetas)

        resultados.append({
            "inicializacion": inicializacion,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })

        print(f"\nInicializacion: {inicializacion}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv(ruta("p08_inicializaciones.csv"), index=False)

    plt.figure(figsize=(8, 5))

    plt.bar(df_resultados["inicializacion"], df_resultados["silhouette"])

    plt.title("Comparacion de inicializaciones K-Means")
    plt.xlabel("Metodo de inicializacion")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(ruta("p08_inicializaciones.png"), dpi=300)

    plt.close()

    return df_resultados, X_escalado


# ------------------------------------------------------------
# 6. COMPARACION DE ESCALADORES
# ------------------------------------------------------------

def comparar_escaladores(X):

    """
    StandardScaler: media 0 y desviacion 1.
    MinMaxScaler  : intervalo [0, 1].

    Cambian las distancias, asi que un inquilino puede cambiar
    de segmento. La inercia de ambos NO es comparable (escalas
    distintas): se compara el Silhouette y los perfiles.
    """

    print("\n" + "=" * 60)
    print("COMPARACION DE ESCALADORES")
    print("=" * 60)

    escaladores = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    resultados = []

    etiquetas_por_escalador = {}

    for nombre, escalador in escaladores.items():

        X_transformado = escalador.fit_transform(X)

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_transformado)

        silhouette = silhouette_score(X_transformado, etiquetas)

        resultados.append({
            "escalador": nombre,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })

        etiquetas_por_escalador[nombre] = etiquetas

        print(f"\nEscalador: {nombre}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    # Cuanto coinciden las particiones de ambos escaladores.
    ari = adjusted_rand_score(
        etiquetas_por_escalador["StandardScaler"],
        etiquetas_por_escalador["MinMaxScaler"]
    )

    print("\nARI entre StandardScaler y MinMaxScaler:", round(ari, 4))

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv(
        ruta("p08_comparacion_escaladores.csv"),
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(df_resultados["escalador"], df_resultados["silhouette"])

    plt.title("Comparacion de escaladores")
    plt.xlabel("Escalador")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(ruta("p08_escaladores.png"), dpi=300)

    plt.close()

    return df_resultados


# ------------------------------------------------------------
# 7. ESTABILIDAD MEDIANTE ARI
# ------------------------------------------------------------

def analizar_estabilidad(X_escalado):

    """
    Ejecuta K-Means con distintas semillas y compara las
    asignaciones con el Adjusted Rand Index.

    ARI = 1   : coincidencia perfecta.
    ARI ~ 0   : coincidencia similar a la esperada por azar.
    ARI < 0   : peor que el azar.

    ARI compara particiones aunque los numeros de cluster
    sean distintos.
    """

    print("\n" + "=" * 60)
    print("ANALISIS DE ESTABILIDAD")
    print("=" * 60)

    semillas = [10, 20, 30, 40, 50, 42]

    particiones = {}

    for semilla in semillas:

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=semilla
        )

        particiones[semilla] = modelo.fit_predict(X_escalado)

    resultados = []

    for i in range(len(semillas)):

        for j in range(i + 1, len(semillas)):

            ari = adjusted_rand_score(
                particiones[semillas[i]],
                particiones[semillas[j]]
            )

            resultados.append({
                "semilla_a": semillas[i],
                "semilla_b": semillas[j],
                "ARI": ari
            })

    df_ari = pd.DataFrame(resultados)

    print("\nComparaciones entre ejecuciones:")

    print(df_ari.round(4).to_string(index=False))

    print("\nARI promedio:", round(df_ari["ARI"].mean(), 4))

    df_ari.to_csv(ruta("p08_estabilidad_ari.csv"), index=False)

    matriz = np.eye(len(semillas))

    for _, fila in df_ari.iterrows():

        i = semillas.index(int(fila["semilla_a"]))
        j = semillas.index(int(fila["semilla_b"]))

        matriz[i, j] = fila["ARI"]
        matriz[j, i] = fila["ARI"]

    plt.figure(figsize=(8, 6))

    imagen = plt.imshow(matriz, vmin=0, vmax=1, cmap="viridis")

    plt.colorbar(imagen, label="Adjusted Rand Index")

    plt.xticks(range(len(semillas)), semillas)
    plt.yticks(range(len(semillas)), semillas)

    plt.title("Estabilidad de K-Means mediante ARI")
    plt.xlabel("Semilla")
    plt.ylabel("Semilla")

    for i in range(len(semillas)):
        for j in range(len(semillas)):
            plt.text(
                j, i, f"{matriz[i, j]:.2f}",
                ha="center", va="center", color="white"
            )

    plt.tight_layout()

    plt.savefig(ruta("p08_matriz_estabilidad.png"), dpi=300)

    plt.close()

    return df_ari


# ------------------------------------------------------------
# 8. MODELO FINAL
# ------------------------------------------------------------

def entrenar_modelo_final(X):

    """
    StandardScaler + K-Means con K=3 (justificado en P7).
    En un proyecto real K debe justificarse otra vez con los
    datos disponibles.
    """

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    modelo = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=NUM_INICIALIZACIONES,
        random_state=SEMILLA
    )

    etiquetas = modelo.fit_predict(X_escalado)

    silhouette_individual = silhouette_samples(X_escalado, etiquetas)

    return modelo, escalador, X_escalado, etiquetas, silhouette_individual


# ------------------------------------------------------------
# 9. CENTROIDES Y PERFILES
# ------------------------------------------------------------

def analizar_centroides(df, modelo, escalador, etiquetas):

    """
    Los centroides estan en espacio escalado; inverse_transform
    los devuelve a unidades originales para interpretarlos.
    """

    print("\n" + "=" * 60)
    print("ANALISIS DE CENTROIDES")
    print("=" * 60)

    df_centroides = pd.DataFrame(
        escalador.inverse_transform(modelo.cluster_centers_),
        columns=VARIABLES
    )

    df_centroides.insert(0, "cluster", range(NUM_CLUSTERS))

    print("\nCentroides en unidades originales:")

    print(df_centroides.round(2).to_string(index=False))

    df_perfiles = df[VARIABLES].copy()

    df_perfiles["cluster"] = etiquetas

    perfiles = df_perfiles.groupby("cluster")[VARIABLES].mean().reset_index()

    print("\nPerfiles promedio de los inquilinos:")

    print(perfiles.round(2).to_string(index=False))

    df_centroides.to_csv(ruta("p08_centroides.csv"), index=False)

    perfiles.to_csv(ruta("p08_perfiles.csv"), index=False)

    return df_centroides, perfiles


# ------------------------------------------------------------
# 10. POSIBLES CASOS ATIPICOS
# ------------------------------------------------------------

def detectar_atipicos(df, X_escalado, modelo, etiquetas, silhouette_individual):

    """
    Distancia euclidiana de cada inquilino al centroide de su
    cluster. Se marcan los que superan el percentil 95.

    Es una regla exploratoria, no una prueba estadistica.
    """

    print("\n" + "=" * 60)
    print("DETECCION DE POSIBLES CASOS ATIPICOS")
    print("=" * 60)

    centroides_asignados = modelo.cluster_centers_[etiquetas]

    distancias = np.linalg.norm(X_escalado - centroides_asignados, axis=1)

    umbral = np.percentile(distancias, PERCENTIL_ATIPICOS)

    df_atipicos = df.copy()

    df_atipicos["cluster"] = etiquetas

    df_atipicos["distancia_centroide"] = distancias

    df_atipicos["silhouette_individual"] = silhouette_individual

    df_atipicos["posible_atipico"] = distancias > umbral

    candidatos = df_atipicos[df_atipicos["posible_atipico"]].copy()

    print(f"\nUmbral de distancia (percentil {PERCENTIL_ATIPICOS}):")
    print(round(umbral, 4))

    print("\nTotal de candidatos:", len(candidatos))

    print(
        "Porcentaje de candidatos:",
        round(len(candidatos) / len(df_atipicos) * 100, 2),
        "%"
    )

    print("\nPrimeros candidatos a revision:")

    print(
        candidatos[
            [
                "id_inquilino",
                "cluster",
                "distancia_centroide",
                "silhouette_individual"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    df_atipicos.to_csv(ruta("p08_analisis_distancias.csv"), index=False)

    candidatos.to_csv(ruta("p08_posibles_atipicos.csv"), index=False)

    plt.figure(figsize=(10, 5))

    plt.scatter(range(len(distancias)), distancias)

    plt.axhline(
        y=umbral,
        linestyle="--",
        label=f"Umbral P{PERCENTIL_ATIPICOS}"
    )

    plt.title("Distancia de inquilinos a sus centroides")
    plt.xlabel("Inquilino")
    plt.ylabel("Distancia euclidiana")

    plt.legend()

    plt.tight_layout()

    plt.savefig(ruta("p08_distancias_atipicos.png"), dpi=300)

    plt.close()

    return df_atipicos, candidatos, umbral


# ------------------------------------------------------------
# 11. VISUALIZACION DE LOS SEGMENTOS
# ------------------------------------------------------------

def visualizar_segmentos(df, etiquetas):

    """
    Proyeccion de dos variables (K-Means usa las cuatro).
    """

    plt.figure(figsize=(10, 6))

    dispersion = plt.scatter(
        df["pagos_puntuales_pct"],
        df["dias_retraso_promedio"],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75
    )

    plt.title("Segmentacion de inquilinos de RENTCONTROL")
    plt.xlabel("Pagos puntuales (%)")
    plt.ylabel("Dias de retraso promedio")

    plt.colorbar(dispersion, label="Cluster")

    plt.tight_layout()

    plt.savefig(ruta("p08_segmentacion_final.png"), dpi=300)

    plt.close()


# ------------------------------------------------------------
# 12. REPORTE INTEGRAL
# ------------------------------------------------------------

def exportar_reporte(df, etiquetas, silhouette_individual, df_centroides, candidatos):

    """
    Un archivo con: id, variables, cluster, perfil, Silhouette
    individual e indicador de posible atipico.
    """

    reporte = df.copy()

    reporte["cluster"] = etiquetas

    reporte["silhouette_individual"] = silhouette_individual

    ids_atipicos = set(candidatos["id_inquilino"].tolist())

    reporte["posible_atipico"] = reporte["id_inquilino"].isin(ids_atipicos)

    # Nombre del perfil: centroides ordenados por pagos puntuales.

    orden = df_centroides.sort_values(
        "pagos_puntuales_pct"
    )["cluster"].tolist()

    nombres = {}

    if len(orden) == 3:

        nombres = {
            orden[0]: "Riesgo de morosidad",
            orden[1]: "Pago regular",
            orden[2]: "Inquilino puntual"
        }

    reporte["perfil"] = reporte["cluster"].map(nombres)

    reporte.to_csv(ruta("p08_reporte_integral.csv"), index=False)

    print("\nReporte integral exportado correctamente.")

    return reporte


# ------------------------------------------------------------
# 13. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("RENTCONTROL - PRACTICA 8")
    print("ANALISIS AVANZADO DE SEGMENTOS DE INQUILINOS")
    print("=" * 60)

    df = cargar_datos()

    df, X = preparar_datos(df)

    resultados_init, X_std = comparar_inicializaciones(X)

    resultados_scaler = comparar_escaladores(X)

    df_ari = analizar_estabilidad(X_std)

    (
        modelo,
        escalador,
        X_escalado,
        etiquetas,
        silhouette_individual
    ) = entrenar_modelo_final(X)

    df_centroides, perfiles = analizar_centroides(
        df, modelo, escalador, etiquetas
    )

    df_atipicos, candidatos, umbral = detectar_atipicos(
        df, X_escalado, modelo, etiquetas, silhouette_individual
    )

    visualizar_segmentos(df, etiquetas)

    reporte = exportar_reporte(
        df, etiquetas, silhouette_individual, df_centroides, candidatos
    )

    # Guardado opcional en MongoDB local.

    if GUARDAR_EN_MONGO:

        try:
            guardar_en_mongo(reporte, COLECCION_SEGMENTOS)

        except Exception as error:
            print("No se pudo guardar en MongoDB:", error)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DE LA PRACTICA 8")
    print("=" * 60)

    print("Inquilinos analizados:", len(df))

    print("Clusters:", NUM_CLUSTERS)

    print(
        "Silhouette final:",
        round(silhouette_score(X_escalado, etiquetas), 4)
    )

    print("ARI promedio entre semillas:", round(df_ari["ARI"].mean(), 4))

    print("Posibles atipicos:", len(candidatos))

    print(
        "Porcentaje de posibles atipicos:",
        round(len(candidatos) / len(df) * 100, 2),
        "%"
    )

    print("\nArchivos generados en:", CARPETA_RESULTADOS)

    print("\nPractica 8 finalizada correctamente.")


# ------------------------------------------------------------
# 14. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()


# ------------------------------------------------------------
# ACTIVIDADES
# ------------------------------------------------------------
# 1. NUM_CLUSTERS = 2 y luego 4; comparar resultados.
# 2. NUM_INICIALIZACIONES = 5 y 50.
# 3. Agregar la variable reportes_mantenimiento y evaluar su efecto.
# 4. PERCENTIL_ATIPICOS = 90 y 97.5; comparar cuantos se marcan.
# 5. Comparar los perfiles con StandardScaler y MinMaxScaler.
# 6. Explicar si los segmentos son estables entre semillas.
# 7. Investigar por que dos inquilinos parecidos podrian quedar
#    en grupos distintos.
# 8. Poner GUARDAR_EN_MONGO = True y revisar en Compass la
#    coleccion rentcontrol.segmentos_inquilinos.