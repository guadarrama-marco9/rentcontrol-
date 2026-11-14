# ============================================================
# UNIDAD III - FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# PRACTICA 9:
# REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA
# (Principal Component Analysis / Analisis de Componentes
# Principales)
#
# CASO: RENTCONTROL
#
# Cada inquilino tiene cuatro variables:
#   pagos_puntuales_pct   % de pagos a tiempo
#   dias_retraso_promedio dias promedio de retraso
#   antiguedad_meses      meses rentando
#   renta_ingreso_pct     % del ingreso que representa la renta
#
# Visualizar cuatro variables a la vez es dificil. PCA las
# transforma en componentes principales:
#   PC1: la mayor variabilidad posible.
#   PC2: la mayor variabilidad restante, sin correlacion con PC1.
#
# OBJETIVOS:
# 1. Entender la dimensionalidad de un conjunto de datos.
# 2. Estandarizar antes de aplicar PCA.
# 3. Calcular componentes y revisar varianza explicada.
# 4. Representar inquilinos en 2D y 3D coloreados por segmento.
# 5. Evaluar el error de reconstruccion.
# 6. Exportar coordenadas para analisis posteriores.
#
# Conceptos:
# - Varianza explicada / acumulada.
# - Cargas (loadings): contribucion de cada variable a un PC.
# - Reconstruccion: aproximar los datos con menos componentes.
#
# PCA busca conservar VARIANZA, no separar clusters. Si en 2D los
# grupos se ven mezclados, K-Means no necesariamente fallo: parte
# de la informacion puede estar en PC3 o PC4.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from rc_datos import VARIABLES


# ------------------------------------------------------------
# 2. CONFIGURACION GENERAL
# ------------------------------------------------------------

SEMILLA = 42

NUM_CLUSTERS = 3

# Referencia de trabajo, no una regla obligatoria.
VARIANZA_OBJETIVO = 0.85

CARPETA_RESULTADOS = "resultados_p09"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def ruta(nombre):
    return f"{CARPETA_RESULTADOS}/{nombre}"


# ------------------------------------------------------------
# 3. CARGA DE DATOS
# ------------------------------------------------------------

def cargar_datos():

    """
    Recupera los datos de la practica 7.

    cluster y perfil se conservan solo para colorear las
    graficas; NO entran a PCA.
    """

    archivo = "p07_inquilinos_segmentados.csv"

    if not os.path.exists(archivo):

        raise FileNotFoundError(
            f"No se encontro {archivo}. Ejecuta primero la practica 7."
        )

    df = pd.read_csv(archivo)

    faltantes = [c for c in VARIABLES if c not in df.columns]

    if faltantes:
        raise ValueError("Faltan las variables: " + str(faltantes))

    if "id_inquilino" not in df.columns:
        df.insert(
            0,
            "id_inquilino",
            [f"INQ-{i:04d}" for i in range(1, len(df) + 1)]
        )

    print("\nDatos cargados correctamente.")

    print("Total de inquilinos:", len(df))

    return df


# ------------------------------------------------------------
# 4. PREPARACION DE VARIABLES
# ------------------------------------------------------------

def preparar_datos(df):

    X = df[VARIABLES].copy()

    for columna in VARIABLES:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")

    X = X.fillna(X.median())

    return X.to_numpy(dtype=float)


# ------------------------------------------------------------
# 5. ESTANDARIZACION
# ------------------------------------------------------------

def estandarizar_datos(X):

    """
    PCA es sensible a la escala: sin estandarizar dominarian
    las variables con valores mas grandes (ej. pagos_puntuales_pct
    de 0 a 100 frente a dias_retraso_promedio de 0 a 30).
    """

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    print("\n" + "=" * 60)
    print("ESTANDARIZACION")
    print("=" * 60)

    print("\nMedia de las variables escaladas:")
    print(np.round(X_escalado.mean(axis=0), 4))

    print("\nDesviacion estandar:")
    print(np.round(X_escalado.std(axis=0), 4))

    return X_escalado, escalador


# ------------------------------------------------------------
# 6. ETIQUETAS DE CLUSTER
# ------------------------------------------------------------

def obtener_clusters(df, X_escalado):

    """
    Usa los clusters de P7 si existen; si no, entrena K-Means.
    Solo se usan para colorear.
    """

    if "cluster" in df.columns:

        print("\nSe utilizaran los clusters de la practica 7.")

        etiquetas = df["cluster"].to_numpy()

    else:

        print("\nNo existen etiquetas de P7. Se generaran con K-Means.")

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=20,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_escalado)

    return np.asarray(etiquetas).astype(int)


# ------------------------------------------------------------
# 7. ENTRENAMIENTO DE PCA
# ------------------------------------------------------------

def entrenar_pca(X_escalado):

    """
    Con cuatro variables se obtienen hasta cuatro componentes.
    """

    modelo_pca = PCA(n_components=len(VARIABLES))

    datos_pca = modelo_pca.fit_transform(X_escalado)

    varianza_acumulada = np.cumsum(modelo_pca.explained_variance_ratio_)

    print("\n" + "=" * 60)
    print("RESULTADOS GENERALES DE PCA")
    print("=" * 60)

    print("\nVarianza explicada (%):")
    print(np.round(modelo_pca.explained_variance_ratio_ * 100, 2))

    print("\nVarianza acumulada (%):")
    print(np.round(varianza_acumulada * 100, 2))

    return modelo_pca, datos_pca, varianza_acumulada


# ------------------------------------------------------------
# 8. TABLA DE VARIANZA EXPLICADA
# ------------------------------------------------------------

def generar_tabla_varianza(modelo_pca):

    varianza = modelo_pca.explained_variance_ratio_

    acumulada = np.cumsum(varianza)

    df_varianza = pd.DataFrame({
        "componente": [f"PC{i + 1}" for i in range(len(varianza))],
        "varianza_explicada": varianza,
        "porcentaje_explicado": varianza * 100,
        "varianza_acumulada": acumulada,
        "porcentaje_acumulado": acumulada * 100
    })

    print("\nTabla de varianza explicada:")

    print(df_varianza.round(4).to_string(index=False))

    df_varianza.to_csv(ruta("p09_varianza_explicada.csv"), index=False)

    return df_varianza


# ------------------------------------------------------------
# 9. CARGAS DE LAS COMPONENTES
# ------------------------------------------------------------

def analizar_cargas(modelo_pca):

    """
    Cada fila es una componente y cada columna una variable
    original. Signos y magnitudes indican direccion y peso.
    """

    nombres_pc = [f"PC{i + 1}" for i in range(len(VARIABLES))]

    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=VARIABLES,
        index=nombres_pc
    )

    print("\n" + "=" * 60)
    print("CARGAS DE LAS COMPONENTES")
    print("=" * 60)

    print(cargas.round(4))

    cargas.to_csv(ruta("p09_cargas_componentes.csv"))

    plt.figure(figsize=(10, 5))

    plt.imshow(
        cargas.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Carga de componente")

    plt.xticks(range(len(VARIABLES)), VARIABLES, rotation=25, ha="right")

    plt.yticks(range(len(nombres_pc)), nombres_pc)

    plt.title("Cargas de las componentes principales")

    plt.tight_layout()

    plt.savefig(ruta("p09_cargas_componentes.png"), dpi=300)

    plt.close()

    return cargas


# ------------------------------------------------------------
# 10. NUMERO DE COMPONENTES NECESARIAS
# ------------------------------------------------------------

def determinar_componentes(modelo_pca):

    """
    Minimo de componentes que alcanza la varianza objetivo.
    Si no se alcanza, se conservan todas.
    """

    acumulada = np.cumsum(modelo_pca.explained_variance_ratio_)

    cantidad = int(np.argmax(acumulada >= VARIANZA_OBJETIVO)) + 1

    if acumulada[-1] < VARIANZA_OBJETIVO:
        cantidad = len(acumulada)

    print("\n" + "=" * 60)
    print("SELECCION DE COMPONENTES")
    print("=" * 60)

    print("Varianza objetivo:", f"{VARIANZA_OBJETIVO * 100:.2f}%")

    print("Componentes necesarias:", cantidad)

    print(
        "Varianza acumulada conservada:",
        round(acumulada[cantidad - 1] * 100, 2),
        "%"
    )

    return cantidad


# ------------------------------------------------------------
# 11. GRAFICAS DE VARIANZA
# ------------------------------------------------------------

def graficar_varianza(df_varianza):

    # Sedimentacion.
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_varianza["componente"],
        df_varianza["porcentaje_explicado"],
        marker="o"
    )

    plt.title("Grafica de sedimentacion PCA")
    plt.xlabel("Componentes principales")
    plt.ylabel("Varianza explicada (%)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(ruta("p09_sedimentacion.png"), dpi=300)

    plt.close()

    # Varianza acumulada.
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_varianza["componente"],
        df_varianza["porcentaje_acumulado"],
        marker="o"
    )

    plt.axhline(
        y=VARIANZA_OBJETIVO * 100,
        linestyle="--",
        label="Objetivo de varianza"
    )

    plt.title("Varianza acumulada PCA")
    plt.xlabel("Componentes principales")
    plt.ylabel("Varianza acumulada (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(ruta("p09_varianza_acumulada.png"), dpi=300)

    plt.close()


# ------------------------------------------------------------
# 12. PCA EN 2D
# ------------------------------------------------------------

def visualizar_pca_2d(datos_pca, etiquetas):

    """
    PCA no usa las etiquetas para calcular las componentes;
    los colores solo representan los clusters de K-Means.
    """

    plt.figure(figsize=(10, 7))

    dispersion = plt.scatter(
        datos_pca[:, 0],
        datos_pca[:, 1],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75,
        s=45
    )

    plt.title("RENTCONTROL - Proyeccion PCA en 2D")
    plt.xlabel("Componente principal 1 (PC1)")
    plt.ylabel("Componente principal 2 (PC2)")

    plt.colorbar(dispersion, label="Cluster K-Means")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(ruta("p09_pca_2d.png"), dpi=300)

    plt.close()


# ------------------------------------------------------------
# 13. PCA EN 3D
# ------------------------------------------------------------

def visualizar_pca_3d(datos_pca, etiquetas):

    figura = plt.figure(figsize=(10, 7))

    eje = figura.add_subplot(111, projection="3d")

    dispersion = eje.scatter(
        datos_pca[:, 0],
        datos_pca[:, 1],
        datos_pca[:, 2],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75,
        s=35
    )

    eje.set_title("RENTCONTROL - Proyeccion PCA en 3D")
    eje.set_xlabel("PC1")
    eje.set_ylabel("PC2")
    eje.set_zlabel("PC3")

    figura.colorbar(
        dispersion,
        ax=eje,
        label="Cluster K-Means",
        shrink=0.7
    )

    plt.tight_layout()

    plt.savefig(ruta("p09_pca_3d.png"), dpi=300)

    plt.close()


# ------------------------------------------------------------
# 14. ERROR DE RECONSTRUCCION
# ------------------------------------------------------------

def evaluar_reconstruccion(X_escalado, datos_pca, modelo_pca, cantidad):

    """
    Reconstruye los datos estandarizados con las primeras
    componentes y calcula el error cuadratico medio (MSE).
    Menor MSE = menor perdida de informacion.
    """

    datos_reducidos = datos_pca[:, :cantidad]

    reconstruidos = (
        datos_reducidos @ modelo_pca.components_[:cantidad, :]
    ) + modelo_pca.mean_

    error = mean_squared_error(X_escalado, reconstruidos)

    print("\n" + "=" * 60)
    print("EVALUACION DE RECONSTRUCCION")
    print("=" * 60)

    print("Componentes utilizadas:", cantidad)

    print("Error cuadratico medio:", round(error, 6))

    return reconstruidos, error


# ------------------------------------------------------------
# 15. EXPORTAR COORDENADAS
# ------------------------------------------------------------

def exportar_coordenadas(df, datos_pca, etiquetas):

    """
    Las coordenadas sirven para dashboards o para guardarlas
    despues en MongoDB local.
    """

    resultado = pd.DataFrame({
        "id_inquilino": df["id_inquilino"],
        "cluster": etiquetas
    })

    for i in range(datos_pca.shape[1]):
        resultado[f"PC{i + 1}"] = datos_pca[:, i]

    resultado.to_csv(ruta("p09_coordenadas_pca.csv"), index=False)

    print("\nCoordenadas PCA exportadas correctamente.")

    return resultado


# ------------------------------------------------------------
# 16. EXPORTAR DATOS RECONSTRUIDOS
# ------------------------------------------------------------

def exportar_reconstruccion(df, reconstruidos, escalador):

    """
    Con menos componentes los datos reconstruidos son solo una
    aproximacion de los originales.
    """

    df_reconstruido = pd.DataFrame(
        escalador.inverse_transform(reconstruidos),
        columns=VARIABLES
    )

    df_reconstruido.insert(
        0,
        "id_inquilino",
        df["id_inquilino"].to_numpy()
    )

    df_reconstruido.to_csv(
        ruta("p09_datos_reconstruidos.csv"),
        index=False
    )

    return df_reconstruido


# ------------------------------------------------------------
# 17. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("RENTCONTROL - PRACTICA 9")
    print("REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA")
    print("=" * 65)

    df = cargar_datos()

    X = preparar_datos(df)

    X_escalado, escalador = estandarizar_datos(X)

    etiquetas = obtener_clusters(df, X_escalado)

    modelo_pca, datos_pca, acumulada = entrenar_pca(X_escalado)

    df_varianza = generar_tabla_varianza(modelo_pca)

    cargas = analizar_cargas(modelo_pca)

    cantidad_componentes = determinar_componentes(modelo_pca)

    graficar_varianza(df_varianza)

    visualizar_pca_2d(datos_pca, etiquetas)

    visualizar_pca_3d(datos_pca, etiquetas)

    reconstruidos, error = evaluar_reconstruccion(
        X_escalado, datos_pca, modelo_pca, cantidad_componentes
    )

    coordenadas = exportar_coordenadas(df, datos_pca, etiquetas)

    df_reconstruido = exportar_reconstruccion(df, reconstruidos, escalador)

    print("\n" + "=" * 65)
    print("RESUMEN FINAL DE LA PRACTICA 9")
    print("=" * 65)

    print("Inquilinos analizados:", len(df))

    print("Variables originales:", len(VARIABLES))

    print("Componentes principales:", len(modelo_pca.components_))

    print(
        "Varianza de PC1:",
        round(modelo_pca.explained_variance_ratio_[0] * 100, 2),
        "%"
    )

    print(
        "Varianza acumulada PC1 y PC2:",
        round(acumulada[1] * 100, 2),
        "%"
    )

    print("Componentes seleccionadas:", cantidad_componentes)

    print("Error de reconstruccion:", round(error, 6))

    print("\nArchivos generados en:", CARPETA_RESULTADOS)

    print("\nPRACTICA 9 FINALIZADA CORRECTAMENTE.")


# ------------------------------------------------------------
# 18. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()