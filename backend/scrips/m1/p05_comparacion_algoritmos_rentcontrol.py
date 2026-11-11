# ================================================================
# PROYECTO: RENTCONTROL
# PRACTICA 05: COMPARACION DE ALGORITMOS DE MACHINE LEARNING
#
# Se comparan tres algoritmos de clasificacion:
#   1. Decision Tree   - reglas de decision sobre las variables.
#   2. Random Forest   - combina muchos arboles.
#   3. KNN             - clasifica segun los inquilinos mas parecidos.
#
# Problema binario:
#   0 = el inquilino paga a tiempo.
#   1 = el inquilino es moroso (riesgo de pago atrasado).
#
# Variable                   Descripcion
# monto_renta                Renta mensual en MXN
# meses_contrato             Antiguedad en el contrato (meses)
# pagos_atrasados_previos    Pagos atrasados en los ultimos 12 meses
# dias_retraso_promedio      Dias promedio de retraso
# relacion_ingreso_renta     Ingreso mensual / renta
# comunicacion               Respuesta a avisos (0 a 10)
# moroso                     Objetivo (0 = No, 1 = Si)
#
# Pregunta: ¿cual algoritmo detecta mejor a los inquilinos morosos
# cuando trabaja con datos que nunca ha visto?
#
# IMPORTANTE: datos sinteticos con fines educativos.
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
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
rng = np.random.default_rng(SEMILLA)
CANTIDAD_INQUILINOS = 500

print("\n==========================================")
print("RENTCONTROL")
print("COMPARACION DE ALGORITMOS")
print("==========================================")

# ================================================================
# 3. GENERACION DEL CONJUNTO DE DATOS
# ================================================================
monto_renta = rng.uniform(3000, 15000, CANTIDAD_INQUILINOS)
meses_contrato = rng.uniform(1, 36, CANTIDAD_INQUILINOS)
pagos_atrasados_previos = rng.integers(0, 9, CANTIDAD_INQUILINOS)
dias_retraso_promedio = rng.uniform(0, 20, CANTIDAD_INQUILINOS)
relacion_ingreso_renta = rng.uniform(1, 6, CANTIDAD_INQUILINOS)
comunicacion = rng.uniform(0, 10, CANTIDAD_INQUILINOS)

# Indicador sintetico de riesgo (mismas ponderaciones que la practica 04).
ruido = rng.normal(0, 5, CANTIDAD_INQUILINOS)

puntuacion_riesgo = (
    pagos_atrasados_previos * 4.0
    + dias_retraso_promedio * 1.0
    - relacion_ingreso_renta * 5.0
    - meses_contrato * 0.25
    - comunicacion * 1.5
    + (monto_renta / 1000) * 0.8
    + ruido
)

# Percentil 65: ~35 % de inquilinos morosos (clases desbalanceadas).
umbral = np.percentile(puntuacion_riesgo, 65)
moroso = (puntuacion_riesgo >= umbral).astype(int)

# ================================================================
# 4. CONSTRUIR EL DATAFRAME
# ================================================================
df = pd.DataFrame({
    "monto_renta": monto_renta,
    "meses_contrato": meses_contrato,
    "pagos_atrasados_previos": pagos_atrasados_previos,
    "dias_retraso_promedio": dias_retraso_promedio,
    "relacion_ingreso_renta": relacion_ingreso_renta,
    "comunicacion": comunicacion,
    "moroso": moroso
})

print("\n==========================================")
print("1. DATOS GENERADOS")
print("==========================================")
print(df.head(10))
print("\nCantidad de inquilinos:", len(df))
print("\nDistribucion de resultados:")
print(df["moroso"].value_counts().sort_index())

# ================================================================
# 5. VARIABLES PREDICTORAS Y OBJETIVO
# ================================================================
X = df[[
    "monto_renta",
    "meses_contrato",
    "pagos_atrasados_previos",
    "dias_retraso_promedio",
    "relacion_ingreso_renta",
    "comunicacion"
]]
y = df["moroso"]

# ================================================================
# 6. DIVIDIR LOS DATOS (60 / 20 / 20)
# ================================================================
X_entrenamiento_validacion, X_prueba, y_entrenamiento_validacion, y_prueba = (
    train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )
)

X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = (
    train_test_split(
        X_entrenamiento_validacion,
        y_entrenamiento_validacion,
        test_size=0.25,
        random_state=SEMILLA,
        stratify=y_entrenamiento_validacion
    )
)

print("\n==========================================")
print("2. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")
print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))

# ================================================================
# 7. DEFINIR LOS ALGORITMOS
# ================================================================
modelos = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=4,
        random_state=SEMILLA
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEMILLA
    ),

    # KNN necesita variables en escalas comparables (la renta va en
    # miles y la comunicacion de 0 a 10), por eso se usa Pipeline.
    "KNN": Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ))
    ])
}

# ================================================================
# 8. ENTRENAMIENTO Y COMPARACION (con validacion)
# ================================================================
resultados = []
modelos_entrenados = {}

print("\n==========================================")
print("3. ENTRENAMIENTO DE LOS ALGORITMOS")
print("==========================================")

for nombre, modelo in modelos.items():

    print("\n--------------------------------------")
    print("Entrenando:", nombre)
    print("--------------------------------------")

    modelo.fit(X_entrenamiento, y_entrenamiento)
    modelos_entrenados[nombre] = modelo

    predicciones = modelo.predict(X_validacion)

    exactitud = accuracy_score(y_validacion, predicciones)
    precision = precision_score(y_validacion, predicciones, zero_division=0)
    recall = recall_score(y_validacion, predicciones, zero_division=0)
    f1 = f1_score(y_validacion, predicciones, zero_division=0)

    resultados.append({
        "algoritmo": nombre,
        "exactitud": exactitud,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("Exactitud:", f"{exactitud:.2%}")
    print("Precision:", f"{precision:.2%}")
    print("Recall:", f"{recall:.2%}")
    print("F1-score:", f"{f1:.2%}")

# ================================================================
# 9. TABLA COMPARATIVA
# ================================================================
df_resultados = pd.DataFrame(resultados)

df_resultados = df_resultados.sort_values(
    by="f1_score", ascending=False
).reset_index(drop=True)

print("\n==========================================")
print("4. COMPARACION DE ALGORITMOS")
print("==========================================")
print(
    df_resultados.to_string(
        index=False,
        formatters={
            "exactitud": "{:.2%}".format,
            "precision": "{:.2%}".format,
            "recall": "{:.2%}".format,
            "f1_score": "{:.2%}".format
        }
    )
)

# ================================================================
# 10. SELECCIONAR EL MEJOR MODELO (mayor F1-score)
# ================================================================
# F1 equilibra precision y recall; es mas informativo que la
# exactitud cuando hay menos morosos que buenos pagadores.
mejor_registro = df_resultados.iloc[0]
mejor_nombre = mejor_registro["algoritmo"]

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")
print("Algoritmo:", mejor_nombre)
print("F1-score de validacion:", f"{mejor_registro['f1_score']:.2%}")
print("Criterio de seleccion: mayor F1-score")

# ================================================================
# 11. REENTRENAR CON ENTRENAMIENTO + VALIDACION
# ================================================================
X_entrenamiento_final = pd.concat([X_entrenamiento, X_validacion])
y_entrenamiento_final = pd.concat([y_entrenamiento, y_validacion])

modelo_final = clone(modelos[mejor_nombre])
modelo_final.fit(X_entrenamiento_final, y_entrenamiento_final)

# ================================================================
# 12. EVALUACION FINAL CON EL CONJUNTO DE PRUEBA
# ================================================================
predicciones_finales = modelo_final.predict(X_prueba)

accuracy_final = accuracy_score(y_prueba, predicciones_finales)
precision_final = precision_score(y_prueba, predicciones_finales, zero_division=0)
recall_final = recall_score(y_prueba, predicciones_finales, zero_division=0)
f1_final = f1_score(y_prueba, predicciones_finales, zero_division=0)

print("\n==========================================")
print("6. EVALUACION FINAL")
print("==========================================")
print("Algoritmo:", mejor_nombre)
print("Exactitud:", f"{accuracy_final:.2%}")
print("Precision:", f"{precision_final:.2%}")
print("Recall:", f"{recall_final:.2%}")
print("F1-score:", f"{f1_final:.2%}")

# ================================================================
# 13. REPORTE DE CLASIFICACION
# ================================================================
print("\n==========================================")
print("7. REPORTE DE CLASIFICACION")
print("==========================================")
print(
    classification_report(
        y_prueba,
        predicciones_finales,
        labels=[0, 1],
        target_names=["Paga a tiempo", "Moroso"],
        zero_division=0
    )
)

# ================================================================
# 14. MATRIZ DE CONFUSION
# ================================================================
matriz = confusion_matrix(y_prueba, predicciones_finales, labels=[0, 1])

print("\n==========================================")
print("8. MATRIZ DE CONFUSION")
print("==========================================")
print(matriz)

# ================================================================
# 15. VALIDACION CRUZADA ESTRATIFICADA (sin usar prueba)
# ================================================================
particiones = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

resultados_cv = cross_val_score(
    clone(modelos[mejor_nombre]),
    X_entrenamiento_final,
    y_entrenamiento_final,
    cv=particiones,
    scoring="f1",
    n_jobs=-1
)

print("\n==========================================")
print("9. VALIDACION CRUZADA")
print("==========================================")
print("Resultados por particion:")
print(resultados_cv)
print("Promedio F1:", f"{resultados_cv.mean():.2%}")
print("Desviacion estandar:", f"{resultados_cv.std():.2%}")

# ================================================================
# 16. GRAFICA COMPARATIVA DE METRICAS
# ================================================================
metricas = ["exactitud", "precision", "recall", "f1_score"]

plt.figure(figsize=(12, 6))
ancho = 0.22
posiciones = np.arange(len(df_resultados))

for i, metrica in enumerate(metricas):
    desplazamiento = (i - 1.5) * ancho
    plt.bar(
        posiciones + desplazamiento,
        df_resultados[metrica],
        width=ancho,
        label=metrica.replace("_", " ").title()
    )

plt.xticks(posiciones, df_resultados["algoritmo"])
plt.ylim(0, 1.05)
plt.ylabel("Resultado")
plt.xlabel("Algoritmo")
plt.title("RENTCONTROL: comparacion de algoritmos de clasificacion")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("p05_comparacion_algoritmos.png", dpi=300)
plt.show()

# ================================================================
# 17. GRAFICA DE LA MATRIZ DE CONFUSION
# ================================================================
visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Paga a tiempo", "Moroso"]
)
visualizacion.plot(cmap="Blues", values_format="d")
plt.title("Matriz de confusion - " + mejor_nombre)
plt.tight_layout()
plt.savefig("p05_matriz_confusion.png", dpi=300)
plt.show()

# ================================================================
# 18. GUARDAR LOS RESULTADOS
# ================================================================
df_resultados.to_csv(
    "p05_resultados_comparacion.csv", index=False, encoding="utf-8"
)

print("\n==========================================")
print("10. ARCHIVOS GENERADOS")
print("==========================================")
print("p05_comparacion_algoritmos.png")
print("p05_matriz_confusion.png")
print("p05_resultados_comparacion.csv")

# ================================================================
# 19. ACTIVIDAD DE MODIFICACION: TRES EXPERIMENTOS
# ================================================================
# Se evaluan con el conjunto de VALIDACION (la prueba no se toca)
# y se registra el F1-score de cada configuracion.

def f1_validacion(modelo):
    modelo.fit(X_entrenamiento, y_entrenamiento)
    return f1_score(y_validacion, modelo.predict(X_validacion), zero_division=0)

experimentos = []

# Experimento A: n_neighbors de KNN
for k in [3, 5, 7, 9, 15]:
    m = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    experimentos.append(("A: KNN n_neighbors", k, f1_validacion(m)))

# Experimento B: n_estimators de Random Forest
for n in [50, 100, 200, 500]:
    m = RandomForestClassifier(
        n_estimators=n, min_samples_leaf=2, n_jobs=-1, random_state=SEMILLA
    )
    experimentos.append(("B: RF n_estimators", n, f1_validacion(m)))

# Experimento C: max_depth de Decision Tree
for d in [2, 3, 5, 8, None]:
    m = DecisionTreeClassifier(
        max_depth=d, min_samples_leaf=4, random_state=SEMILLA
    )
    experimentos.append(("C: Tree max_depth", str(d), f1_validacion(m)))

df_experimentos = pd.DataFrame(
    experimentos, columns=["experimento", "valor", "f1_validacion"]
)

print("\n==========================================")
print("11. EXPERIMENTOS A, B Y C")
print("==========================================")
print(
    df_experimentos.to_string(
        index=False, formatters={"f1_validacion": "{:.2%}".format}
    )
)

df_experimentos.to_csv("p05_experimentos.csv", index=False, encoding="utf-8")

# ================================================================
# 20. CONCLUSION DEL SISTEMA
# ================================================================
print("\n==========================================")
print("CONCLUSION")
print("==========================================")
print("El algoritmo seleccionado fue:", mejor_nombre)
print("Su F1-score final fue:", f"{f1_final:.2%}")
print(
    "\nRENTCONTROL ha completado la comparacion de algoritmos."
    "\nEl mejor algoritmo depende de los datos, del problema y de la"
    "\nmetrica a optimizar. En este caso interesa detectar morosos,"
    "\npor lo que conviene vigilar el RECALL de la clase Moroso."
)
print("\nPROCESO FINALIZADO.")

# Registra los resultados de los experimentos y explica como cambio el
# rendimiento. No asumas que Random Forest o KNN siempre seran mejores:
# da tus propias conclusiones con base en lo que arrojen.
