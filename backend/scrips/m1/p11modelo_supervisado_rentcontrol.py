# ============================================================
# RENTCONTROL - PRACTICA 11
# CLASIFICACION SUPERVISADA DEL RIESGO DE MOROSIDAD
# ============================================================

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RUTA_BASE = Path(__file__).resolve().parents[2]
RUTA_RESULTADOS = RUTA_BASE / "resultados_p10"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"
ARCHIVO_DATOS = RUTA_RESULTADOS / "rentcontrol_datos.csv"
ARCHIVO_MODELO = RUTA_MODELOS / "modelo_supervisado_rentcontrol.joblib"
CARACTERISTICAS = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct",
]
OBJETIVO = "moroso"
SEMILLA = 42


def cargar_datos():
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_DATOS}. Ejecuta primero p10datos_rentcontrol.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    columnas = CARACTERISTICAS + [OBJETIVO]
    faltantes = [columna for columna in columnas if columna not in datos]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    if datos[columnas].isna().any().any():
        raise ValueError("El conjunto contiene valores nulos.")
    if set(datos[OBJETIVO].unique()) != {0, 1}:
        raise ValueError("La variable moroso debe contener las clases 0 y 1.")
    return datos


def main():
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    X = datos[CARACTERISTICAS]
    y = datos[OBJETIVO].astype(int)

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEMILLA,
        stratify=y,
    )

    modelos = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=SEMILLA,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=SEMILLA,
            n_jobs=1,
        ),
    }

    resultados = []
    modelos_entrenados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_entrenamiento, y_entrenamiento)
        predicciones = modelo.predict(X_prueba)
        resultados.append({
            "modelo": nombre,
            "accuracy": accuracy_score(y_prueba, predicciones),
            "precision_moroso": precision_score(
                y_prueba, predicciones, zero_division=0
            ),
            "recall_moroso": recall_score(
                y_prueba, predicciones, zero_division=0
            ),
            "f1_moroso": f1_score(
                y_prueba, predicciones, zero_division=0
            ),
        })
        modelos_entrenados[nombre] = modelo

    tabla = pd.DataFrame(resultados).sort_values(
        ["f1_moroso", "recall_moroso", "accuracy"],
        ascending=False,
    )
    mejor_nombre = tabla.iloc[0]["modelo"]
    mejor_modelo = modelos_entrenados[mejor_nombre]
    predicciones = mejor_modelo.predict(X_prueba)

    tabla.to_csv(
        RUTA_RESULTADOS / "p11_comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump({
        "modelo": mejor_modelo,
        "caracteristicas": CARACTERISTICAS,
        "objetivo": OBJETIVO,
        "nombre_modelo": mejor_nombre,
    }, ARCHIVO_MODELO)

    matriz = confusion_matrix(y_prueba, predicciones, labels=[0, 1])
    figura, eje = plt.subplots(figsize=(6, 5))
    imagen = eje.imshow(matriz, cmap="Blues")
    figura.colorbar(imagen, ax=eje)
    eje.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Menor riesgo", "Mayor riesgo"],
        yticklabels=["Menor riesgo", "Mayor riesgo"],
        xlabel="Prediccion",
        ylabel="Valor real",
        title="RENTCONTROL - Matriz de confusion",
    )
    for (fila, columna), valor in np.ndenumerate(matriz):
        eje.text(columna, fila, str(valor), ha="center", va="center")
    figura.tight_layout()
    figura.savefig(RUTA_RESULTADOS / "p11_matriz_confusion.png", dpi=180)
    plt.close(figura)

    print("=" * 60)
    print("RENTCONTROL - PRACTICA 11: CLASIFICACION")
    print("=" * 60)
    print("Registros:", len(datos))
    print("\nComparacion en el conjunto de prueba:")
    print(tabla.round(3).to_string(index=False))
    print("\nMejor modelo por F1 de morosidad:", mejor_nombre)
    print(classification_report(
        y_prueba,
        predicciones,
        labels=[0, 1],
        target_names=["Menor riesgo", "Mayor riesgo"],
        zero_division=0,
    ))
    print("Modelo guardado en:", ARCHIVO_MODELO)
    print("Resultados guardados en:", RUTA_RESULTADOS)
    print("La prediccion es orientativa; no debe usarse para excluir inquilinos.")


if __name__ == "__main__":
    main()
