import os

import numpy as np
from pymongo import MongoClient


VARIABLES = [
    "pagos_puntuales_pct",
    "dias_retraso_promedio",
    "antiguedad_meses",
    "renta_ingreso_pct",
]


def generar_inquilinos(seed=42):
    rng = np.random.default_rng(seed)
    perfiles = (
        (60, (88, 100), (0, 4), (12, 60), (15, 35)),
        (60, (70, 94), (2, 10), (4, 50), (25, 52)),
        (60, (40, 82), (5, 25), (0, 48), (35, 75)),
    )
    valores = {
        variable: []
        for variable in VARIABLES
    }

    for cantidad, *rangos in perfiles:
        for variable, (minimo, maximo) in zip(VARIABLES, rangos):
            valores[variable].extend(
                rng.uniform(minimo, maximo, cantidad)
            )

    import pandas as pd

    data = pd.DataFrame(valores)
    data.insert(
        0,
        "id_inquilino",
        [f"INQ-{indice:04d}" for indice in range(1, len(data) + 1)],
    )
    return data


def guardar_en_mongo(dataframe, coleccion):
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("Configura MONGODB_URI para guardar los resultados.")

    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        resultado = client["rentcontrol"][coleccion].insert_many(
            dataframe.to_dict(orient="records")
        )
    return resultado.inserted_ids
