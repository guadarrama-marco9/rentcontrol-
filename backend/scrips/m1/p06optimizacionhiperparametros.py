# ================================================================
# PROYECTO: RENTCONTROL
# PRACTICA 06: OPTIMIZACION DE HIPERPARAMETROS
#
# Archivo:
# p06optimizacion_hiperparametros.py
#
# Objetivo:
# Encontrar la mejor configuracion de Decision Tree, Random Forest
# y KNN para predecir si un inquilino pagara su renta con retraso
# el proximo mes.
#
# Caso RENTCONTROL:
# El administrador quiere anticiparse a la morosidad para enviar
# recordatorios antes del vencimiento.
#
# Conceptos:
# - Parametros: valores que el algoritmo aprende al entrenar.
# - Hiperparametros: valores que fijamos ANTES de entrenar.
# - Optimizacion: buscar la configuracion con mejores resultados.
# - Validacion cruzada: evaluar con distintas particiones de datos.
#
# Hiperparametros que se optimizan:
#   Decision Tree : max_depth, min_samples_split, min_samples_leaf,
#                   criterion (gini / entropy)
#   Random Forest : n_estimators, max_depth, min_samples_split,
#                   min_samples_leaf
#   KNN           : n_neighbors, weights, metric
#
# Tecnicas:
#   GridSearchCV       prueba TODAS las combinaciones definidas.
#   RandomizedSearchCV prueba n_iter combinaciones al azar.
#   StratifiedKFold    particiones que conservan la proporcion
#                      de clases (morosos / no morosos).
#
# Problema:
# 0 = Paga a tiempo
# 1 = Moroso (pagara con retraso)
#
# Nota: la metrica principal es F1, no solo accuracy, porque lo
# importante es detectar a los inquilinos con riesgo (clase 1).
# ================================================================


# ================================================================
# 1. IMPORTAR BIBLIOTECAS
# ================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ================================================================
# 2. CONFIGURACION GENERAL
# ================================================================

SEMILLA = 42

CANTIDAD_INQUILINOS = 500

CARPETA_RESULTADOS = "resultados_p06"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def ruta(nombre):
    return os.path.join(CARPETA_RESULTADOS, nombre)


rng = np.random.default_rng(SEMILLA)

print("\n==========================================")
print("RENTCONTROL")
print("OPTIMIZACION DE HIPERPARAMETROS")
print("==========================================")


# ================================================================
# 3. GENERAR DATOS SINTETICOS
# ================================================================

pagos_puntuales_pct = rng.uniform(40, 100, CANTIDAD_INQUILINOS)

dias_retraso_promedio = rng.uniform(0, 20, CANTIDAD_INQUILINOS)

antiguedad_meses = rng.uniform(0, 60, CANTIDAD_INQUILINOS)

renta_ingreso_pct = rng.uniform(15, 70, CANTIDAD_INQUILINOS)

# Indicador sintetico de riesgo (0 = sin riesgo, 1 = riesgo maximo).
# Cada termino se lleva a la escala 0-1 y se pondera.

riesgo = (
    ((100 - pagos_puntuales_pct) / 60) * 0.35
    +
    (dias_retraso_promedio / 20) * 0.25
    +
    ((renta_ingreso_pct - 15) / 55) * 0.30
    +
    (1 - antiguedad_meses / 60) * 0.10
)

# Ruido: ningun inquilino es perfectamente predecible.

ruido = rng.normal(0, 0.10, CANTIDAD_INQUILINOS)

riesgo_final = riesgo + ruido

# Variable objetivo: 1 = moroso.

moroso = (riesgo_final >= 0.55).astype(int)


# ================================================================
# 4. CREAR DATAFRAME
# ================================================================

df = pd.DataFrame({
    "pagos_puntuales_pct": pagos_puntuales_pct,
    "dias_retraso_promedio": dias_retraso_promedio,
    "antiguedad_meses": antiguedad_meses,
    "renta_ingreso_pct": renta_ingreso_pct,
    "moroso": moroso
})

print("\n1. DATOS GENERADOS")
print(df.head())

print("\nDistribucion de clases:")
print(df["moroso"].value_counts())


# ================================================================
# 5. SEPARAR VARIABLES
# ================================================================

VARIABLES = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct"
]

X = df[VARIABLES]

y = df["moroso"]


# ================================================================
# 6. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ================================================================

# 20% queda reservado para la evaluacion final.
# El 80% se usa para optimizar con validacion cruzada.

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEMILLA,
    stratify=y
)

print("\n2. DISTRIBUCION DE DATOS")
print("Entrenamiento:", len(X_entrenamiento))
print("Prueba:", len(X_prueba))


# ================================================================
# 7. CONFIGURAR VALIDACION CRUZADA
# ================================================================

validacion_cruzada = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEMILLA
)


# ================================================================
# 8. DEFINIR LOS ALGORITMOS Y SUS HIPERPARAMETROS
# ================================================================

# ---- MODELO 1: DECISION TREE ----

modelo_arbol = DecisionTreeClassifier(random_state=SEMILLA)

parametros_arbol = {
    "max_depth": [3, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1],
    "criterion": ["gini"]
}

# ---- MODELO 2: RANDOM FOREST ----

modelo_bosque = RandomForestClassifier(
    random_state=SEMILLA,
    n_jobs=1
)

parametros_bosque = {
    "n_estimators": [50],
    "max_depth": [3, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1]
}

# ---- MODELO 3: KNN ----
# KNN usa distancias, asi que escalamos dentro de un Pipeline.
# El escalador se ajusta solo con los datos de entrenamiento
# de cada particion (evita fuga de informacion).

modelo_knn = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", KNeighborsClassifier())
])

# Formato de parametros en un Pipeline: nombrepaso__parametro

parametros_knn = {
    "clasificador__n_neighbors": [3, 11],
    "clasificador__weights": ["uniform"],
    "clasificador__metric": ["euclidean"]
}

modelos = {
    "Decision Tree": (modelo_arbol, parametros_arbol),
    "Random Forest": (modelo_bosque, parametros_bosque),
    "KNN": (modelo_knn, parametros_knn)
}


# ================================================================
# 9. GRIDSEARCHCV
# ================================================================

resultados_grid = []

mejores_modelos_grid = {}

print("\n==========================================")
print("3. OPTIMIZACION CON GRIDSEARCHCV")
print("==========================================")

for nombre, (modelo, parametros) in modelos.items():

    print("\n--------------------------------------")
    print("Optimizando:", nombre)
    print("--------------------------------------")

    busqueda = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        scoring="f1",
        cv=validacion_cruzada,
        n_jobs=1,
        return_train_score=True,
        verbose=0
    )

    busqueda.fit(X_entrenamiento, y_entrenamiento)

    mejores_modelos_grid[nombre] = busqueda.best_estimator_

    print("Mejor F1-score:", f"{busqueda.best_score_:.2%}")
    print("\nMejores hiperparametros:")
    print(busqueda.best_params_)

    resultados_grid.append({
        "algoritmo": nombre,
        "mejor_f1": busqueda.best_score_,
        "hiperparametros": str(busqueda.best_params_),
        "combinaciones": len(busqueda.cv_results_["params"])
    })

df_grid = pd.DataFrame(resultados_grid)

print("\n==========================================")
print("4. RESULTADOS GRIDSEARCHCV")
print("==========================================")

print(
    df_grid[["algoritmo", "mejor_f1", "combinaciones"]].to_string(
        index=False,
        formatters={"mejor_f1": "{:.2%}".format}
    )
)


# ================================================================
# 10. RANDOMIZEDSEARCHCV
# ================================================================

# No prueba todas las combinaciones: elige n_iter al azar.
# Util cuando el espacio de busqueda es grande.

parametros_random = {

    "Decision Tree": {
        "max_depth": randint(2, 15),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "criterion": ["gini", "entropy"]
    },

    "Random Forest": {
        "n_estimators": randint(50, 301),
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": randint(2, 12),
        "min_samples_leaf": randint(1, 6)
    },

    "KNN": {
        "clasificador__n_neighbors": [3, 5, 7, 9, 11, 15, 19],
        "clasificador__weights": ["uniform", "distance"],
        "clasificador__metric": ["euclidean", "manhattan"]
    }
}

resultados_random = []

mejores_modelos_random = {}

print("\n==========================================")
print("5. OPTIMIZACION CON RANDOMIZEDSEARCHCV")
print("==========================================")

for nombre, (modelo, _) in modelos.items():

    print("\n--------------------------------------")
    print("Optimizando aleatoriamente:", nombre)
    print("--------------------------------------")

    busqueda_random = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros_random[nombre],
        n_iter=2,
        scoring="f1",
        cv=validacion_cruzada,
        n_jobs=1,
        random_state=SEMILLA,
        verbose=0
    )

    busqueda_random.fit(X_entrenamiento, y_entrenamiento)

    mejores_modelos_random[nombre] = busqueda_random.best_estimator_

    print("Mejor F1-score:", f"{busqueda_random.best_score_:.2%}")
    print("\nMejores hiperparametros:")
    print(busqueda_random.best_params_)

    resultados_random.append({
        "algoritmo": nombre,
        "mejor_f1": busqueda_random.best_score_,
        "hiperparametros": str(busqueda_random.best_params_)
    })

df_random = pd.DataFrame(resultados_random)


# ================================================================
# 11. COMPARAR GRIDSEARCH Y RANDOMIZEDSEARCH
# ================================================================

print("\n==========================================")
print("6. COMPARACION DE METODOS DE OPTIMIZACION")
print("==========================================")

comparacion = pd.merge(
    df_grid[["algoritmo", "mejor_f1"]].rename(
        columns={"mejor_f1": "gridsearch"}
    ),
    df_random[["algoritmo", "mejor_f1"]].rename(
        columns={"mejor_f1": "randomizedsearch"}
    ),
    on="algoritmo"
)

print(
    comparacion.to_string(
        index=False,
        formatters={
            "gridsearch": "{:.2%}".format,
            "randomizedsearch": "{:.2%}".format
        }
    )
)


# ================================================================
# 12. SELECCIONAR EL MEJOR ALGORITMO
# ================================================================

mejor_indice = df_grid["mejor_f1"].idxmax()

mejor_nombre = df_grid.loc[mejor_indice, "algoritmo"]

mejor_modelo = mejores_modelos_grid[mejor_nombre]

print("\n==========================================")
print("7. MEJOR MODELO OPTIMIZADO")
print("==========================================")

print("Algoritmo:", mejor_nombre)
print(
    "F1-score en validacion cruzada:",
    f"{df_grid.loc[mejor_indice, 'mejor_f1']:.2%}"
)
print("\nConfiguracion seleccionada:")
print(mejor_modelo.get_params())


# ================================================================
# 13. EVALUACION FINAL
# ================================================================

# IMPORTANTE: el conjunto de prueba se usa hasta este momento.
# No participo en la optimizacion de hiperparametros.

predicciones = mejor_modelo.predict(X_prueba)

accuracy = accuracy_score(y_prueba, predicciones)
precision = precision_score(y_prueba, predicciones, zero_division=0)
recall = recall_score(y_prueba, predicciones, zero_division=0)
f1 = f1_score(y_prueba, predicciones, zero_division=0)

print("\n==========================================")
print("8. EVALUACION CON DATOS DE PRUEBA")
print("==========================================")

print("Accuracy:", f"{accuracy:.2%}")
print("Precision:", f"{precision:.2%}")
print("Recall:", f"{recall:.2%}")
print("F1-score:", f"{f1:.2%}")

# Interpretacion para RENTCONTROL:
# - Precision: de los inquilinos marcados como morosos, cuantos
#   realmente lo eran (evita molestar a quien paga bien).
# - Recall: de los morosos reales, cuantos detectamos.


# ================================================================
# 14. REPORTE DE CLASIFICACION
# ================================================================

print("\n==========================================")
print("9. REPORTE DE CLASIFICACION")
print("==========================================")

print(
    classification_report(
        y_prueba,
        predicciones,
        labels=[0, 1],
        target_names=["Paga a tiempo", "Moroso"],
        zero_division=0
    )
)


# ================================================================
# 15. MATRIZ DE CONFUSION
# ================================================================

matriz = confusion_matrix(y_prueba, predicciones, labels=[0, 1])

print("\n==========================================")
print("10. MATRIZ DE CONFUSION")
print("==========================================")

print(matriz)

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Paga a tiempo", "Moroso"]
)

visualizacion.plot(cmap="Blues", values_format="d")

plt.title("Matriz de confusion - " + mejor_nombre)

plt.tight_layout()

plt.savefig(ruta("p06_matriz_confusion.png"), dpi=300)

plt.show()


# ================================================================
# 16. GRAFICA COMPARATIVA
# ================================================================

posiciones = np.arange(len(comparacion))

ancho = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    posiciones - ancho / 2,
    comparacion["gridsearch"],
    width=ancho,
    label="GridSearchCV"
)

plt.bar(
    posiciones + ancho / 2,
    comparacion["randomizedsearch"],
    width=ancho,
    label="RandomizedSearchCV"
)

plt.xticks(posiciones, comparacion["algoritmo"])

plt.ylim(0, 1.05)

plt.ylabel("F1-score")

plt.xlabel("Algoritmo")

plt.title("RENTCONTROL - Comparacion de metodos de optimizacion")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(ruta("p06_comparacion_optimizacion.png"), dpi=300)

plt.show()


# ================================================================
# 17. EXPORTAR RESULTADOS
# ================================================================

df_grid.to_csv(
    ruta("p06_resultados_gridsearch.csv"),
    index=False,
    encoding="utf-8"
)

df_random.to_csv(
    ruta("p06_resultados_randomizedsearch.csv"),
    index=False,
    encoding="utf-8"
)

comparacion.to_csv(
    ruta("p06_comparacion_metodos.csv"),
    index=False,
    encoding="utf-8"
)


# ================================================================
# 18. CONCLUSION
# ================================================================

print("\n==========================================")
print("CONCLUSION DEL PROYECTO")
print("==========================================")

print("Mejor algoritmo:", mejor_nombre)
print("Accuracy final:", f"{accuracy:.2%}")
print("Precision final:", f"{precision:.2%}")
print("Recall final:", f"{recall:.2%}")
print("F1-score final:", f"{f1:.2%}")

print("\nArchivos generados en:", CARPETA_RESULTADOS)

print("\nPRACTICA 06 FINALIZADA.")