# ================================================================
# PROYECTO: RENTCONTROL
# PRACTICA 07: SEGMENTACION DE INQUILINOS MEDIANTE K-MEANS
# ARCHIVO: p07segmentacionkmeans.py
#
# OBJETIVO:
# Identificar grupos de inquilinos con comportamiento de pago
# similar, usando clustering (K-Means), escalamiento de variables,
# seleccion de K e interpretacion de resultados.
#
# CASO:
# RENTCONTROL administra rentas de varias propiedades. El
# administrador quiere perfiles de inquilinos para definir
# estrategias de seguimiento (recordatorios, cobranza amable,
# renovacion de contrato), pero NO sabe cuantos perfiles existen.
#
# No se usa ninguna columna objetivo (como "moroso"): K-Means
# busca similitudes solo con las caracteristicas.
#
# K-MEANS (aprendizaje NO supervisado):
#   1. Inicializacion: elige K centroides iniciales.
#   2. Asignacion: cada inquilino va al centroide mas cercano.
#   3. Actualizacion: recalcula cada centroide como el promedio.
#   4. Convergencia: repite hasta que los grupos casi no cambian.
#
# Conceptos: centroide, inercia (WCSS), Silhouette Score.
#
# POR QUE ESCALAR:
#   pagos_puntuales_pct   0 a 100
#   dias_retraso_promedio 0 a 30
#   antiguedad_meses      0 a 60
#   renta_ingreso_pct     5 a 90
# Sin escalar, las variables con numeros grandes dominan la
# distancia. StandardScaler deja media 0 y desviacion 1.
#
# IMPORTANTE (uso responsable):
# Los perfiles describen patrones de pago; NO determinan si una
# persona incumplira ni deben usarse para negar una renta de forma
# automatica. Sirven para priorizar seguimiento.
# ================================================================


# ================================================================
# 1. IMPORTAR LAS BIBLIOTECAS
# ================================================================

import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from rc_datos import VARIABLES, generar_inquilinos


# ================================================================
# 2. CONFIGURACION
# ================================================================

SEMILLA = 42

K_FINAL = 3

CARPETA_RESULTADOS = "resultados_p07"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def ruta(nombre):
    return os.path.join(CARPETA_RESULTADOS, nombre)


# ================================================================
# 3. GENERAR LOS DATOS
# ================================================================

# Tres perfiles simulados (ver rc_datos.py). Las etiquetas de
# perfil no se entregan al algoritmo.

df = generar_inquilinos(SEMILLA)


# ================================================================
# 4. EXPLORAR LOS DATOS
# ================================================================

print("\n==========================================")
print("1. DATOS GENERADOS")
print("==========================================")

print(df.head(10))

print("\nCantidad de inquilinos:", len(df))

print("\nInformacion estadistica:")

print(df[VARIABLES].describe().round(2))


# ================================================================
# 5. SELECCIONAR Y ESCALAR LAS VARIABLES
# ================================================================

X = df[VARIABLES]

escalador = StandardScaler()

X_escalado = escalador.fit_transform(X)

print("\n==========================================")
print("2. DATOS ESCALADOS")
print("==========================================")

print("Media aproximada:", X_escalado.mean(axis=0).round(2))

print("Desviacion estandar aproximada:", X_escalado.std(axis=0).round(2))


# ================================================================
# 6. METODO DEL CODO
# ================================================================

# La inercia siempre baja al aumentar K. Buscamos el "codo":
# el punto donde la mejora empieza a ser pequena.

valores_k = range(1, 9)

inercias = []

for k in valores_k:

    modelo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=1)

    modelo.fit(X_escalado)

    inercias.append(modelo.inertia_)

    print(f"K={k} | Inercia={modelo.inertia_:.2f}")

plt.figure(figsize=(9, 5))

plt.plot(list(valores_k), inercias, marker="o", linewidth=2)

plt.title("RENTCONTROL - Metodo del codo")

plt.xlabel("Cantidad de grupos (K)")

plt.ylabel("Inercia (WCSS)")

plt.xticks(list(valores_k))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(ruta("p07_metodo_del_codo.png"), dpi=300)

plt.show()


# ================================================================
# 7. SILHOUETTE SCORE
# ================================================================

# Cercano a 1: grupos bien separados.
# Cercano a 0: grupos que se traslapan.
# Negativo: posibles asignaciones inadecuadas.

resultados_silhouette = []

for k in range(2, 9):

    modelo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=1)

    etiquetas = modelo.fit_predict(X_escalado)

    puntuacion = silhouette_score(X_escalado, etiquetas)

    resultados_silhouette.append(puntuacion)

    print(f"K={k} | Silhouette Score={puntuacion:.4f}")

plt.figure(figsize=(9, 5))

plt.plot(list(range(2, 9)), resultados_silhouette, marker="o", linewidth=2)

plt.title("RENTCONTROL - Evaluacion Silhouette")

plt.xlabel("Cantidad de grupos (K)")

plt.ylabel("Silhouette Score")

plt.xticks(list(range(2, 9)))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(ruta("p07_silhouette.png"), dpi=300)

plt.show()


# ================================================================
# 8. ENTRENAR EL MODELO FINAL
# ================================================================

# K=3 es la hipotesis inicial del caso. Debe justificarse con el
# codo, el Silhouette y el conocimiento del negocio.

modelo_final = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=1)

df["cluster"] = modelo_final.fit_predict(X_escalado)


# ================================================================
# 9. INTERPRETAR LOS GRUPOS
# ================================================================

print("\n==========================================")
print("3. CANTIDAD DE INQUILINOS POR GRUPO")
print("==========================================")

print(df["cluster"].value_counts().sort_index())

perfiles = df.groupby("cluster")[VARIABLES].mean().round(2)

print("\n==========================================")
print("4. PERFIL DE CADA GRUPO")
print("==========================================")

print(perfiles)

# Los numeros de cluster son etiquetas arbitrarias.
# Ordenamos por porcentaje de pagos puntuales (menor a mayor)
# para asignar nombres descriptivos.

orden = perfiles["pagos_puntuales_pct"].sort_values().index.tolist()

nombres = [
    "Riesgo de morosidad",
    "Pago regular",
    "Inquilino puntual"
]

mapa_nombres = dict(zip(orden, nombres))

df["perfil"] = df["cluster"].map(mapa_nombres)

print("\n==========================================")
print("5. INTERPRETACION DE LOS PERFILES")
print("==========================================")

print(df[["id_inquilino"] + VARIABLES + ["cluster", "perfil"]].head(15))


# ================================================================
# 10. VISUALIZAR LOS GRUPOS
# ================================================================

# Mostramos 2 de las 4 variables (K-Means usa las cuatro).

plt.figure(figsize=(10, 6))

for cluster in sorted(df["cluster"].unique()):

    datos_cluster = df[df["cluster"] == cluster]

    plt.scatter(
        datos_cluster["pagos_puntuales_pct"],
        datos_cluster["dias_retraso_promedio"],
        label=mapa_nombres[cluster],
        alpha=0.65
    )

# Centroides en escala original.

centroides_df = pd.DataFrame(
    escalador.inverse_transform(modelo_final.cluster_centers_),
    columns=VARIABLES
)

plt.scatter(
    centroides_df["pagos_puntuales_pct"],
    centroides_df["dias_retraso_promedio"],
    marker="X",
    s=250,
    edgecolor="black",
    linewidth=1.5,
    label="Centroides"
)

plt.title("RENTCONTROL - Segmentacion de inquilinos")

plt.xlabel("Pagos puntuales (%)")

plt.ylabel("Dias de retraso promedio")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(ruta("p07_segmentacion_inquilinos.png"), dpi=300)

plt.show()

print("\n==========================================")
print("6. CENTROIDES EN ESCALA ORIGINAL")
print("==========================================")

print(centroides_df.round(2))


# ================================================================
# 11. EVALUAR EL MODELO FINAL
# ================================================================

silhouette_final = silhouette_score(X_escalado, df["cluster"])

print("\n==========================================")
print("7. EVALUACION FINAL DEL MODELO")
print("==========================================")

print("Numero de grupos:", K_FINAL)

print("Inercia final:", round(modelo_final.inertia_, 2))

print("Silhouette Score:", round(silhouette_final, 4))


# ================================================================
# 12. EXPORTAR LOS RESULTADOS
# ================================================================

# P08 y P09 leen este archivo.

df.to_csv(
    "p07_inquilinos_segmentados.csv",
    index=False,
    encoding="utf-8-sig"
)

perfiles.to_csv(
    ruta("p07_perfiles_inquilinos.csv"),
    encoding="utf-8-sig"
)


# ================================================================
# 13. MENSAJE FINAL
# ================================================================

print("\n==========================================")
print("PRACTICA 07 FINALIZADA")
print("==========================================")

print("Archivos generados:")
print("1. p07_inquilinos_segmentados.csv (carpeta actual)")
print(f"2. {CARPETA_RESULTADOS}/p07_metodo_del_codo.png")
print(f"3. {CARPETA_RESULTADOS}/p07_silhouette.png")
print(f"4. {CARPETA_RESULTADOS}/p07_segmentacion_inquilinos.png")
print(f"5. {CARPETA_RESULTADOS}/p07_perfiles_inquilinos.csv")

print("\nRENTCONTROL ha segmentado a sus inquilinos.")
print("El proceso se realizo mediante aprendizaje no supervisado.")


# ================================================================
# ESTRATEGIAS POSIBLES POR PERFIL (ejemplos)
# ================================================================
#
# - Riesgo de morosidad : recordatorio anticipado de pago, opciones
#                         de pago, seguimiento cercano.
# - Pago regular        : recordatorio estandar antes del vencimiento.
# - Inquilino puntual   : propuesta de renovacion de contrato.
#
# ACTIVIDADES
# 1. Cambiar K_FINAL a 2 y a 4; comparar y explicar los cambios.
# 2. Agregar una variable (ej. reportes_mantenimiento) en
#    rc_datos.py: VARIABLES, generador, centroides y exportacion.
# 3. Ejecutar K-Means sin escalar y con StandardScaler; comparar.
# 4. Investigar el efecto de valores atipicos en los centroides
#    (ej. un inquilino con 30 dias de retraso en todos sus pagos).