# ============================================================
# FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# PRACTICA 04: ENTRENAMIENTO, VALIDACION Y PRUEBA
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# CASO DE ESTUDIO: RENTCONTROL
# RENTCONTROL administra rentas (propiedades, inquilinos, contratos y pagos).
# Se desea un modelo que identifique inquilinos con RIESGO DE PAGO ATRASADO
# (moroso) para que el arrendador pueda dar seguimiento oportuno.
#
# OBJETIVO:
# Evaluar un modelo de Machine Learning y detectar:
#   - Sobreajuste (overfitting): el modelo memoriza ruido del entrenamiento
#     y pierde capacidad de generalizar.
#   - Subajuste (underfitting): el modelo es demasiado simple y no aprende
#     los patrones.
#   - Buen ajuste: aprende patrones utiles y rinde bien con datos nuevos.
#
# Variable                  Descripcion
# monto_renta               Renta mensual en MXN
# meses_contrato            Antiguedad del inquilino en el contrato (meses)
# pagos_atrasados_previos   Pagos atrasados en los ultimos 12 meses (0 a 8)
# dias_retraso_promedio     Dias promedio de retraso cuando se atrasa
# relacion_ingreso_renta    Ingreso mensual / renta (ej. 3 = gana 3 veces la renta)
# comunicacion              Nivel de respuesta a avisos y mensajes (0 a 10)
# moroso                    1 = pagara tarde (riesgo), 0 = pagara a tiempo
#
# Division de los datos: 60% entrenamiento, 20% validacion, 20% prueba
#
# Conjunto        Para que sirve
# Entrenamiento   Aprender patrones.
# Validacion      Ajustar parametros y elegir modelos.
# Prueba          Evaluar el desempeno final con datos no usados antes.
#
# NOTA: los datos son sinteticos y con fines educativos.
# No representan inquilinos ni pagos reales.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTAR BIBLIOTECAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 2. CREAR CARPETA DE RESULTADOS
# ------------------------------------------------------------

os.makedirs("resultados", exist_ok=True)


# ------------------------------------------------------------
# 3. GENERAR DATOS SINTETICOS (inquilinos ficticios)
# ------------------------------------------------------------

np.random.seed(42)

cantidad_inquilinos = 500

monto_renta = np.random.uniform(3000, 15000, cantidad_inquilinos)
meses_contrato = np.random.uniform(1, 36, cantidad_inquilinos)
pagos_atrasados_previos = np.random.randint(0, 9, cantidad_inquilinos)
dias_retraso_promedio = np.random.uniform(0, 20, cantidad_inquilinos)
relacion_ingreso_renta = np.random.uniform(1, 6, cantidad_inquilinos)
comunicacion = np.random.uniform(0, 10, cantidad_inquilinos)

# Puntuacion de riesgo sintetica.
# Mas atrasos y mas dias de retraso => mas riesgo.
# Mayor ingreso, antiguedad y comunicacion => menos riesgo.
# El ruido evita una regla perfectamente determinista.
ruido = np.random.normal(0, 5, cantidad_inquilinos)

puntuacion_riesgo = (
    pagos_atrasados_previos * 4.0
    + dias_retraso_promedio * 1.0
    - relacion_ingreso_renta * 5.0
    - meses_contrato * 0.25
    - comunicacion * 1.5
    + (monto_renta / 1000) * 0.8
    + ruido
)

# Umbral: regla de generacion de datos ficticios, no criterio real.
# Se usa el percentil 65 para que ~35 % de los inquilinos sean "morosos"
# (clases algo desbalanceadas, como suele pasar en la realidad).
umbral = np.percentile(puntuacion_riesgo, 65)
moroso = (puntuacion_riesgo >= umbral).astype(int)

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
print("\nCantidad de registros:", len(df))
print("\nDistribucion de las clases:")
print(df["moroso"].value_counts())


# ------------------------------------------------------------
# (OPCIONAL) CARGAR DATOS REALES DESDE MONGODB
# ------------------------------------------------------------
# Cuando RENTCONTROL tenga datos reales, se pueden reemplazar los
# sinteticos leyendo una coleccion con PyMongo. Ejemplo (ajusta nombres):
#
# from pymongo import MongoClient
# cliente = MongoClient("mongodb://localhost:27017")
# coleccion = cliente["rentcontrol"]["historial_inquilinos"]
# df = pd.DataFrame(list(coleccion.find({}, {"_id": 0})))
#
# La coleccion debe tener las mismas columnas que este DataFrame.


# ------------------------------------------------------------
# 4. VARIABLES DE ENTRADA Y OBJETIVO
# ------------------------------------------------------------

columnas_entrada = [
    "monto_renta",
    "meses_contrato",
    "pagos_atrasados_previos",
    "dias_retraso_promedio",
    "relacion_ingreso_renta",
    "comunicacion"
]

X = df[columnas_entrada]
y = df["moroso"]


# ------------------------------------------------------------
# 5. SEPARAR ENTRENAMIENTO (60 %) Y TEMPORAL (40 %)
# ------------------------------------------------------------

X_entrenamiento, X_temporal, y_entrenamiento, y_temporal = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)


# ------------------------------------------------------------
# 6. DIVIDIR VALIDACION (20 %) Y PRUEBA (20 %)
# ------------------------------------------------------------

X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(
    X_temporal, y_temporal, test_size=0.50, random_state=42, stratify=y_temporal
)

print("\n==========================================")
print("2. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")
print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))


# ------------------------------------------------------------
# 7. ENTRENAR MODELOS CON DIFERENTES PROFUNDIDADES
# ------------------------------------------------------------

profundidades = [1, 2, 3, 5, 10, None]

resultados = []
modelos = {}

for profundidad in profundidades:

    print("\n--------------------------------------")
    print("Entrenando modelo:", profundidad)
    print("--------------------------------------")

    modelo = DecisionTreeClassifier(max_depth=profundidad, random_state=42)

    # Entrena solo con el conjunto de entrenamiento.
    modelo.fit(X_entrenamiento, y_entrenamiento)

    exactitud_entrenamiento = accuracy_score(
        y_entrenamiento, modelo.predict(X_entrenamiento)
    )
    exactitud_validacion = accuracy_score(
        y_validacion, modelo.predict(X_validacion)
    )

    modelos[profundidad] = modelo

    resultados.append({
        "profundidad": "Sin limite" if profundidad is None else str(profundidad),
        # Valor numerico para graficar (12 = posicion visual de "Sin limite").
        "profundidad_grafica": 12 if profundidad is None else profundidad,
        "entrenamiento": exactitud_entrenamiento,
        "validacion": exactitud_validacion
    })

df_resultados = pd.DataFrame(resultados)

print("\n==========================================")
print("3. COMPARACION DE MODELOS")
print("==========================================")
print(
    df_resultados[["profundidad", "entrenamiento", "validacion"]]
    .to_string(index=False)
)


# ------------------------------------------------------------
# 8. IDENTIFICAR POSIBLES PROBLEMAS DE AJUSTE
# ------------------------------------------------------------

print("\n==========================================")
print("4. ANALISIS DE AJUSTE")
print("==========================================")

for _, fila in df_resultados.iterrows():

    diferencia = fila["entrenamiento"] - fila["validacion"]

    print("\nModelo:", fila["profundidad"])
    print("Exactitud entrenamiento:", f"{fila['entrenamiento']:.2%}")
    print("Exactitud validacion:", f"{fila['validacion']:.2%}")
    print("Diferencia:", f"{diferencia:.2%}")

    # Indicadores didacticos, no prueba definitiva.
    if fila["entrenamiento"] < 0.70 and fila["validacion"] < 0.70:
        print("Posible subajuste.")
    elif diferencia > 0.10:
        print("Posible sobreajuste.")
    else:
        print("No se observa una brecha grande.")


# ------------------------------------------------------------
# 9. SELECCIONAR EL MODELO CON MEJOR VALIDACION
# ------------------------------------------------------------

indice_mejor = df_resultados["validacion"].idxmax()
mejor_registro = df_resultados.loc[indice_mejor]

profundidad_seleccionada = int(mejor_registro["profundidad_grafica"])
if profundidad_seleccionada == 12:
    profundidad_seleccionada = None

mejor_modelo = modelos[profundidad_seleccionada]

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")
print("Profundidad seleccionada:", mejor_registro["profundidad"])
print("Exactitud de validacion:", f"{mejor_registro['validacion']:.2%}")
print("Configuracion interna del modelo:", mejor_modelo.get_params()["max_depth"])


# ------------------------------------------------------------
# 10. EVALUACION FINAL CON EL CONJUNTO DE PRUEBA
# ------------------------------------------------------------
# La clase positiva es moroso = 1.
# - Precision: de los inquilinos marcados como riesgosos,
#   cuantos realmente lo eran.
# - Recall: de los inquilinos realmente morosos,
#   cuantos logro detectar el modelo.

pred_prueba = mejor_modelo.predict(X_prueba)

exactitud = accuracy_score(y_prueba, pred_prueba)
precision = precision_score(y_prueba, pred_prueba, zero_division=0)
recall = recall_score(y_prueba, pred_prueba, zero_division=0)
f1 = f1_score(y_prueba, pred_prueba, zero_division=0)

print("\n==========================================")
print("6. EVALUACION FINAL")
print("==========================================")
print(f"Exactitud: {exactitud:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1:.2%}")


# ------------------------------------------------------------
# 11. REPORTE DE CLASIFICACION
# ------------------------------------------------------------

print("\n==========================================")
print("7. REPORTE DE CLASIFICACION")
print("==========================================")
print(
    classification_report(
        y_prueba,
        pred_prueba,
        labels=[0, 1],
        target_names=["Paga a tiempo", "Moroso"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 12. MATRIZ DE CONFUSION
# ------------------------------------------------------------

matriz = confusion_matrix(y_prueba, pred_prueba, labels=[0, 1])

print("\n==========================================")
print("8. MATRIZ DE CONFUSION")
print("==========================================")
print(matriz)

vn, fp, fn, vp = matriz.ravel()
print(f"\nVerdaderos negativos (a tiempo, predicho a tiempo): {vn}")
print(f"Falsos positivos (a tiempo, predicho moroso): {fp}")
print(f"Falsos negativos (moroso, NO detectado): {fn}")
print(f"Verdaderos positivos (moroso, detectado): {vp}")


# ------------------------------------------------------------
# 13. VALIDACION CRUZADA (solo con entrenamiento, estratificada)
# ------------------------------------------------------------
# StratifiedKFold conserva la proporcion de morosos en cada particion.
# No se usa el conjunto de prueba.

modelo_cv = DecisionTreeClassifier(
    max_depth=profundidad_seleccionada, random_state=42
)

particiones = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

puntuaciones_cv = cross_val_score(
    modelo_cv,
    X_entrenamiento,
    y_entrenamiento,
    cv=particiones,
    scoring="accuracy"
)

print("\n==========================================")
print("9. VALIDACION CRUZADA")
print("==========================================")
print("Resultados de cada particion:")
print(puntuaciones_cv)
print("Promedio:", f"{puntuaciones_cv.mean():.2%}")
print("Desviacion estandar:", f"{puntuaciones_cv.std():.2%}")


# ------------------------------------------------------------
# 14. GRAFICA: ENTRENAMIENTO VS VALIDACION
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

x = df_resultados["profundidad_grafica"]

plt.plot(x, df_resultados["entrenamiento"], marker="o",
         linewidth=2, label="Entrenamiento")
plt.plot(x, df_resultados["validacion"], marker="s",
         linewidth=2, label="Validacion")

plt.xticks(x, df_resultados["profundidad"])
plt.title("RENTCONTROL: rendimiento del modelo de riesgo de morosidad",
          fontsize=14)
plt.xlabel("Profundidad del arbol")
plt.ylabel("Exactitud")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("resultados/comparacion_modelos.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# 15. GRAFICA: MATRIZ DE CONFUSION
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))
plt.imshow(matriz)
plt.title("Matriz de confusion - RENTCONTROL")
plt.xlabel("Prediccion del modelo")
plt.ylabel("Resultado real")
plt.xticks([0, 1], ["Paga a tiempo", "Moroso"])
plt.yticks([0, 1], ["Paga a tiempo", "Moroso"])

for fila in range(2):
    for columna in range(2):
        plt.text(columna, fila, str(matriz[fila, columna]),
                 ha="center", va="center", fontsize=14)

plt.colorbar()
plt.tight_layout()
plt.savefig("resultados/matriz_confusion.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 16. IMPORTANCIA DE VARIABLES (extra para RENTCONTROL)
# ------------------------------------------------------------
# Indica que variables uso mas el arbol; util para explicar
# al arrendador que factores pesan mas en el riesgo.

importancias = pd.Series(
    mejor_modelo.feature_importances_, index=columnas_entrada
).sort_values(ascending=False)

print("\n==========================================")
print("10. IMPORTANCIA DE VARIABLES")
print("==========================================")
print(importancias.round(3))


# ------------------------------------------------------------
# 17. EXPORTAR RESULTADOS
# ------------------------------------------------------------

df_resultados.to_csv("resultados/comparacion_modelos.csv", index=False)

df_predicciones = X_prueba.copy()
df_predicciones["resultado_real"] = y_prueba
df_predicciones["prediccion"] = pred_prueba
df_predicciones.to_csv("resultados/predicciones_prueba.csv", index=False)


# ------------------------------------------------------------
# 18. CONCLUSIONES
# ------------------------------------------------------------

print("\n==========================================")
print("11. CONCLUSIONES")
print("==========================================")
print("El entrenamiento permite aprender patrones de pago a partir de ejemplos.")
print("La validacion ayuda a elegir la profundidad sin usar los datos de prueba.")
print("La prueba estima el comportamiento con inquilinos no vistos antes.")
print("Una brecha grande entre entrenamiento y validacion indica sobreajuste.")
print("En RENTCONTROL conviene vigilar el RECALL: un falso negativo es un "
      "inquilino moroso que el sistema no alerto.")
print("La validacion cruzada estratificada mide la estabilidad del rendimiento.")

print("\nArchivos generados en la carpeta resultados.")
print("\nPRACTICA 04 (RENTCONTROL) FINALIZADA.")