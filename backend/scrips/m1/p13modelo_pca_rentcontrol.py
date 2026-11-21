# ============================================================
# RENTCONTROL - PRACTICA 13
# REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA
# ============================================================

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RUTA_BASE = Path(__file__).resolve().parents[2]
RUTA_RESULTADOS = RUTA_BASE / "resultados_p10"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"
ARCHIVO_DATOS = RUTA_RESULTADOS / "p12_inquilinos_segmentados.csv"
ARCHIVO_MODELO = RUTA_MODELOS / "modelo_pca_rentcontrol.joblib"
CARACTERISTICAS = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct",
]


def main():
    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_DATOS}. Ejecuta primero p12modelo_kmeans_rentcontrol.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    columnas = ["id_inquilino", "cluster", "perfil", *CARACTERISTICAS]
    faltantes = [columna for columna in columnas if columna not in datos]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    if datos[CARACTERISTICAS].isna().any().any():
        raise ValueError("Las caracteristicas contienen valores nulos.")

    X = datos[CARACTERISTICAS].astype(float)
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    modelo_pca = PCA(n_components=2)
    componentes = modelo_pca.fit_transform(X_escalado)

    coordenadas = datos[["id_inquilino", "cluster", "perfil"]].copy()
    coordenadas["PC1"] = componentes[:, 0]
    coordenadas["PC2"] = componentes[:, 1]
    coordenadas.to_csv(
        RUTA_RESULTADOS / "p13_coordenadas_pca.csv",
        index=False,
        encoding="utf-8-sig",
    )

    varianzas = pd.DataFrame({
        "componente": ["PC1", "PC2"],
        "varianza_explicada": modelo_pca.explained_variance_ratio_,
        "porcentaje_explicado": modelo_pca.explained_variance_ratio_ * 100,
        "porcentaje_acumulado": (
            modelo_pca.explained_variance_ratio_.cumsum() * 100
        ),
    })
    varianzas.to_csv(
        RUTA_RESULTADOS / "p13_varianza_explicada.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=CARACTERISTICAS,
        index=["PC1", "PC2"],
    )
    cargas.to_csv(
        RUTA_RESULTADOS / "p13_cargas_componentes.csv",
        encoding="utf-8-sig",
    )
    joblib.dump({
        "modelo_pca": modelo_pca,
        "escalador": escalador,
        "caracteristicas": CARACTERISTICAS,
    }, ARCHIVO_MODELO)

    figura, eje = plt.subplots(figsize=(9, 6))
    dispersion = eje.scatter(
        coordenadas["PC1"],
        coordenadas["PC2"],
        c=coordenadas["cluster"],
        cmap="viridis",
        alpha=0.75,
        s=30,
    )
    eje.set(
        title="RENTCONTROL - Proyeccion PCA de inquilinos",
        xlabel="Componente principal 1 (PC1)",
        ylabel="Componente principal 2 (PC2)",
    )
    figura.colorbar(dispersion, ax=eje, label="Cluster K-Means")
    eje.grid(alpha=0.2)
    figura.tight_layout()
    figura.savefig(RUTA_RESULTADOS / "p13_pca_2d.png", dpi=180)
    plt.close(figura)

    print("=" * 60)
    print("RENTCONTROL - PRACTICA 13: PCA")
    print("=" * 60)
    print(varianzas.round(2).to_string(index=False))
    print(
        "\nVarianza acumulada en 2D:",
        f"{modelo_pca.explained_variance_ratio_.sum() * 100:.2f}%",
    )
    print("Cargas de los componentes:")
    print(cargas.round(3).to_string())
    print("\nCoordenadas, grafica y modelo guardados en:", RUTA_RESULTADOS)


if __name__ == "__main__":
    main()
