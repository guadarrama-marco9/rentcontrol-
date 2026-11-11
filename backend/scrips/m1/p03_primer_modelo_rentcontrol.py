# ============================================================
# PRACTICA 3: PRIMER MODELO (REGRESION)
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Objetivo: estimar los DIAS DE RETRASO de pago (numero continuo)
# de un inquilino con un arbol de regresion.
# Datos sinteticos con fines educativos.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 2. CONFIGURACION
np.random.seed(42)
os.makedirs("resultados", exist_ok=True)

# 3. GENERAR 500 INQUILINOS
cantidad = 500

monto_renta = np.random.randint(3000, 15001, cantidad)
meses_contrato = np.random.randint(1, 37, cantidad)
pagos_atrasados_previos = np.random.randint(0, 9, cantidad)
relacion_ingreso_renta = np.random.uniform(1, 6, cantidad)
comunicacion = np.random.randint(0, 11, cantidad)

# 4. GENERAR DIAS DE RETRASO
ruido = np.random.normal(0, 1.5, cantidad)

dias_retraso = (
    8
    + pagos_atrasados_previos * 1.8
    + (monto_renta / 1000) * 0.3
    - relacion_ingreso_renta * 1.5
    - comunicacion * 0.5
    - meses_contrato * 0.05
    + ruido
)

# Los dias de retraso no pueden ser negativos ni exceder 30.
dias_retraso = np.clip(dias_retraso, 0, 30)

# 5. ORGANIZAR CON PANDAS
df = pd.DataFrame({
    "monto_renta": monto_renta,
    "meses_contrato": meses_contrato,
    "pagos_atrasados_previos": pagos_atrasados_previos,
    "relacion_ingreso_renta": relacion_ingreso_renta,
    "comunicacion": comunicacion,
    "dias_retraso": dias_retraso
})

print("\nDATOS DE LOS INQUILINOS")
print(df.head(10))

print("\nTAMAÑO DE LOS DATOS")
print(df.shape)

df.to_csv("resultados/inquilinos_sinteticos.csv", index=False)

# 6. DEFINIR VARIABLES
columnas = [
    "monto_renta",
    "meses_contrato",
    "pagos_atrasados_previos",
    "relacion_ingreso_renta",
    "comunicacion"
]

X = df[columnas]
y = df["dias_retraso"]

# 7. DIVIDIR: 60 % entrenamiento, 20 % validacion, 20 % prueba
X_entrenamiento, X_temporal, y_entrenamiento, y_temporal = train_test_split(
    X, y, test_size=0.40, random_state=42
)

X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(
    X_temporal, y_temporal, test_size=0.50, random_state=42
)

print("\nDIVISION DE LOS DATOS")
print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))

# 8. ENTRENAMIENTO Y VALIDACION (distintas profundidades)
mejor_modelo = None
mejor_resultado = -float("inf")
mejor_profundidad = None

for profundidad in [2, 3, 4, 5, 6]:

    modelo = DecisionTreeRegressor(max_depth=profundidad, random_state=42)
    modelo.fit(X_entrenamiento, y_entrenamiento)

    pred_validacion = modelo.predict(X_validacion)
    resultado = r2_score(y_validacion, pred_validacion)

    print("Profundidad:", profundidad, "R2 validacion:", round(resultado, 3))

    if resultado > mejor_resultado:
        mejor_resultado = resultado
        mejor_profundidad = profundidad
        mejor_modelo = modelo

print("\nMEJOR PROFUNDIDAD:", mejor_profundidad)

# 9. REVISAR SOBREAJUSTE Y SUBAJUSTE
r2_entrenamiento = r2_score(
    y_entrenamiento, mejor_modelo.predict(X_entrenamiento)
)
r2_validacion = r2_score(
    y_validacion, mejor_modelo.predict(X_validacion)
)

print("\nCOMPARACION DEL MODELO")
print("R2 entrenamiento:", round(r2_entrenamiento, 3))
print("R2 validacion:", round(r2_validacion, 3))

# 10. MODELO FINAL (entrenamiento + validacion)
X_final = pd.concat([X_entrenamiento, X_validacion])
y_final = pd.concat([y_entrenamiento, y_validacion])

modelo_final = DecisionTreeRegressor(max_depth=mejor_profundidad, random_state=42)
modelo_final.fit(X_final, y_final)

# 11. PRUEBA FINAL
predicciones = modelo_final.predict(X_prueba)

mae = mean_absolute_error(y_prueba, predicciones)
mse = mean_squared_error(y_prueba, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_prueba, predicciones)

print("\nRESULTADOS DE LA PRUEBA")
print("Error absoluto medio (dias):", round(mae, 3))
print("Error cuadratico medio:", round(mse, 3))
print("Raiz del error cuadratico (dias):", round(rmse, 3))
print("R2:", round(r2, 3))

# 12. COMPARAR RESULTADOS
comparacion = pd.DataFrame({
    "Dias_retraso_real": y_prueba,
    "Dias_retraso_predicho": predicciones
})

print("\nCOMPARACION DE DIAS DE RETRASO")
print(comparacion.head(15))

comparacion.to_csv("resultados/comparacion.csv", index=False)

# 13. GRAFICA REAL VS PREDICHO
plt.figure(figsize=(8, 5))
plt.scatter(y_prueba, predicciones)
plt.xlabel("Dias de retraso reales")
plt.ylabel("Dias de retraso predichos")
plt.title("RENTCONTROL: resultados del modelo")
plt.plot([0, 30], [0, 30], linestyle="--")
plt.savefig("resultados/grafica_predicciones.png")
plt.show()

# 14. IMPORTANCIA DE VARIABLES
plt.figure(figsize=(10, 5))
plt.bar(X.columns, modelo_final.feature_importances_)
plt.title("Importancia de las variables")
plt.xlabel("Variables")
plt.ylabel("Importancia")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("resultados/importancia_variables.png")
plt.show()

# 15. PREDICCION DE UN NUEVO INQUILINO
nuevo_inquilino = pd.DataFrame({
    "monto_renta": [8500],
    "meses_contrato": [12],
    "pagos_atrasados_previos": [2],
    "relacion_ingreso_renta": [3.0],
    "comunicacion": [8]
})

resultado = modelo_final.predict(nuevo_inquilino)

print("\nPREDICCION DE NUEVO INQUILINO")
print("Dias de retraso estimados:", round(resultado[0], 2))

print("\nPRACTICA TERMINADA")
