# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 25: COMPARACIÓN DE MODELOS DE IA
#
# Modelos:
# 1. CNN 1D - PyTorch (práctica 24)
# 2. Random Forest - Scikit-learn
#
# Objetivo:
# Comparar ambos modelos con exactamente los mismos conjuntos
# de entrenamiento, validación y prueba.
# ============================================================

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Guardar gráficas sin abrir ventanas.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# ============================================================
# 1. RUTAS Y CONFIGURACIÓN
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_DATOS = RUTA_PROYECTO / "resultados_cnn"

RUTA_SALIDA = RUTA_PROYECTO / "resultados_comparacion"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_DATOS = RUTA_DATOS / "rentcontrol_datos_cnn.npz"

ARCHIVO_METRICAS_CNN = RUTA_DATOS / "metricas_cnn.json"

ARCHIVO_MODELO_RF = RUTA_SALIDA / "modelo_random_forest.joblib"

NOMBRE_CLASES = ["Deterioro", "Estable", "Mejora"]

RANDOM_STATE = 42

print("=" * 65)
print("RENTCONTROL")
print("PRÁCTICA 25 - COMPARACIÓN CNN 1D VS RANDOM FOREST")
print("=" * 65)

# ============================================================
# 2. CARGAR LOS DATOS PREPARADOS
# ============================================================

print("\nCARGANDO LOS DATOS...")

if not ARCHIVO_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {ARCHIVO_DATOS}. "
        "Ejecuta primero p23preparacion_datos_cnn.py"
    )

datos = np.load(ARCHIVO_DATOS)

X_train, X_val, X_test = datos["X_train"], datos["X_val"], datos["X_test"]
y_train, y_val, y_test = datos["y_train"], datos["y_val"], datos["y_test"]

print(f"Entrenamiento: {X_train.shape}")
print(f"Validación:    {X_val.shape}")
print(f"Prueba:        {X_test.shape}")

# ============================================================
# 3. TRANSFORMAR LOS DATOS PARA RANDOM FOREST
# ============================================================

# La CNN usa (registros, canales, meses) = (N, 2, 3).
# Random Forest necesita (registros, características) = (N, 6).

X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_val_rf = X_val.reshape(X_val.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

print("\nCaracterísticas utilizadas (6):")
print("Puntualidad mes 1, 2 y 3 | Cobertura de renta mes 1, 2 y 3")

assert X_train_rf.shape[1] == X_val_rf.shape[1] == X_test_rf.shape[1] == 6
assert len(X_train_rf) == len(y_train)
assert set(np.unique(y_train)) == {0, 1, 2}
assert set(np.unique(y_val)) == {0, 1, 2}
assert set(np.unique(y_test)) == {0, 1, 2}

# ============================================================
# 4. ENTRENAR RANDOM FOREST
# ============================================================

print("\nCONFIGURANDO Y ENTRENANDO RANDOM FOREST...")

modelo_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

inicio = time.perf_counter()

modelo_rf.fit(X_train_rf, y_train)

tiempo_entrenamiento_rf = time.perf_counter() - inicio

print(f"Tiempo de entrenamiento: {tiempo_entrenamiento_rf:.4f} segundos")

# ============================================================
# 5. VALIDACIÓN Y PRUEBA
# ============================================================

predicciones_val = modelo_rf.predict(X_val_rf)

print(
    f"\nAccuracy validación: "
    f"{accuracy_score(y_val, predicciones_val):.4f}"
)
print(
    f"F1 Macro validación: "
    f"{f1_score(y_val, predicciones_val, average='macro'):.4f}"
)

inicio = time.perf_counter()

predicciones_test = modelo_rf.predict(X_test_rf)

tiempo_prediccion_rf = time.perf_counter() - inicio

accuracy_rf = accuracy_score(y_test, predicciones_test)

precision_rf = precision_score(
    y_test, predicciones_test, average="macro", zero_division=0
)

recall_rf = recall_score(
    y_test, predicciones_test, average="macro", zero_division=0
)

f1_rf = f1_score(
    y_test, predicciones_test, average="macro", zero_division=0
)

print("\nRESULTADOS RANDOM FOREST (PRUEBA)")
print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1 Score:  {f1_rf:.4f}")

print("\nMATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_test, predicciones_test, labels=[0, 1, 2]))

print("\nREPORTE DE CLASIFICACIÓN")
print(
    classification_report(
        y_test,
        predicciones_test,
        labels=[0, 1, 2],
        target_names=NOMBRE_CLASES,
        zero_division=0
    )
)

# ============================================================
# 6. RECUPERAR RESULTADOS DE LA CNN
# ============================================================

if not ARCHIVO_METRICAS_CNN.exists():
    raise FileNotFoundError(
        "No se encontró metricas_cnn.json. "
        "Ejecuta primero p24entrenamiento_cnn1d.py"
    )

with open(ARCHIVO_METRICAS_CNN, "r", encoding="utf-8") as archivo:
    metricas_cnn = json.load(archivo)

metricas_prueba_cnn = metricas_cnn["metricas_prueba"]

accuracy_cnn = float(metricas_prueba_cnn["accuracy"])
precision_cnn = float(metricas_prueba_cnn["precision_macro"])
recall_cnn = float(metricas_prueba_cnn["recall_macro"])
f1_cnn = float(metricas_prueba_cnn["f1_macro"])

print("\nMÉTRICAS CNN RECUPERADAS CORRECTAMENTE")
print(f"Accuracy:  {accuracy_cnn:.6f}")
print(f"F1 Score:  {f1_cnn:.6f}")

# ============================================================
# 7. TABLA COMPARATIVA
# ============================================================

comparacion = pd.DataFrame({
    "Modelo": ["CNN 1D", "Random Forest"],
    "Accuracy": [accuracy_cnn, accuracy_rf],
    "Precision Macro": [precision_cnn, precision_rf],
    "Recall Macro": [recall_cnn, recall_rf],
    "F1 Macro": [f1_cnn, f1_rf]
})

print("\nRESULTADOS COMPARATIVOS")
print(
    comparacion.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

archivo_csv = RUTA_SALIDA / "comparacion_cnn_randomforest.csv"

comparacion.to_csv(archivo_csv, index=False, encoding="utf-8")

print(f"\nTabla guardada: {archivo_csv}")

# ============================================================
# 8. GUARDAR MÉTRICAS EN JSON
# ============================================================

resultados_json = {
    "proyecto": "RentControl",
    "practica": 25,
    "modelos": {
        "CNN_1D": {
            "accuracy": accuracy_cnn,
            "precision_macro": precision_cnn,
            "recall_macro": recall_cnn,
            "f1_macro": f1_cnn
        },
        "Random_Forest": {
            "accuracy": accuracy_rf,
            "precision_macro": precision_rf,
            "recall_macro": recall_rf,
            "f1_macro": f1_rf,
            "tiempo_entrenamiento_segundos": tiempo_entrenamiento_rf,
            "tiempo_prediccion_segundos": tiempo_prediccion_rf
        }
    },
    "observacion": (
        "Ambos modelos usan los mismos conjuntos de datos. "
        "La CNN y Random Forest clasifican una tendencia derivada "
        "de la puntualidad observada en los mismos meses de entrada."
    )
}

archivo_json = RUTA_SALIDA / "metricas_comparacion.json"

with open(archivo_json, "w", encoding="utf-8") as archivo:
    json.dump(resultados_json, archivo, indent=4, ensure_ascii=False)

print(f"Métricas guardadas: {archivo_json}")

# ============================================================
# 9. GRÁFICA COMPARATIVA
# ============================================================

metricas = ["Accuracy", "Precision Macro", "Recall Macro", "F1 Macro"]

posiciones = np.arange(len(metricas))

ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    posiciones - ancho / 2,
    comparacion.iloc[0, 1:].values.astype(float),
    ancho,
    label="CNN 1D"
)

ax.bar(
    posiciones + ancho / 2,
    comparacion.iloc[1, 1:].values.astype(float),
    ancho,
    label="Random Forest"
)

ax.set_title("RentControl: CNN 1D vs Random Forest")
ax.set_ylabel("Valor de la métrica")
ax.set_xlabel("Métricas de evaluación")
ax.set_xticks(posiciones)
ax.set_xticklabels(metricas)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

archivo_grafica = RUTA_SALIDA / "comparacion_modelos.png"

plt.savefig(archivo_grafica, dpi=300)

plt.close()

print(f"Gráfica guardada: {archivo_grafica}")

# ============================================================
# 10. GUARDAR MODELO RANDOM FOREST
# ============================================================

joblib.dump(modelo_rf, ARCHIVO_MODELO_RF)

print(f"Modelo guardado: {ARCHIVO_MODELO_RF}")

# ============================================================
# 11. CONCLUSIÓN AUTOMÁTICA
# ============================================================

print("\n" + "=" * 65)
print("CONCLUSIÓN DE LA COMPARACIÓN")
print("=" * 65)

for nombre, valor_cnn, valor_rf in [
    ("accuracy", accuracy_cnn, accuracy_rf),
    ("F1 Macro", f1_cnn, f1_rf)
]:
    if valor_cnn > valor_rf:
        print(f"CNN 1D obtuvo mayor {nombre}.")
    elif valor_rf > valor_cnn:
        print(f"Random Forest obtuvo mayor {nombre}.")
    else:
        print(f"Ambos modelos obtuvieron el mismo {nombre}.")

print("\nIMPORTANTE:")
print(
    "Una métrica elevada no demuestra por sí sola capacidad de "
    "predecir el comportamiento futuro de pago."
)
print(
    "La etiqueta de tendencia se deriva de la puntualidad "
    "utilizada como entrada."
)

print("\nPRÁCTICA 25 FINALIZADA")
print("=" * 65)
