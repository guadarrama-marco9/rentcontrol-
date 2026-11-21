# ==============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 24: ENTRENAMIENTO DE UNA CNN 1D (PYTORCH)
#
# Archivo:
# p24entrenamiento_cnn1d.py
#
# Objetivo:
# Entrenar una red neuronal convolucional 1D que clasifica la
# tendencia de pago de un contrato en tres categorías:
#   0 = Deterioro, 1 = Estable, 2 = Mejora.
#
# Entrada: [lote, 2 canales (puntualidad y cobertura), 3 meses]
#
# Incluye: DataLoader, Adam, ReduceLROnPlateau, Early Stopping,
# evaluación con accuracy, precision, recall y F1 (macro).
#
# Archivos de salida (carpeta resultados_cnn):
#   rentcontrol_cnn1d.pth, historial_entrenamiento_cnn.csv,
#   matriz_confusion_cnn.csv, reporte_clasificacion_cnn.txt,
#   metricas_cnn.json
# ==============================================================

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_DATOS = RUTA_PROYECTO / "resultados_cnn"

ARCHIVO_DATOS = RUTA_DATOS / "rentcontrol_datos_cnn.npz"

ARCHIVO_METADATOS = RUTA_DATOS / "metadatos_cnn.json"

RUTA_RESULTADOS = RUTA_DATOS

SEMILLA = 42
NUM_CLASES = 3
CANALES_ENTRADA = 2
LONGITUD_SECUENCIA = 3

TAMANO_LOTE = 256
NUM_EPOCAS = 40
PACIENCIA_EARLY_STOPPING = 8
TASA_APRENDIZAJE = 0.001

MAPA_ETIQUETAS = {"Deterioro": 0, "Estable": 1, "Mejora": 2}

NOMBRES_CLASES = ["Deterioro", "Estable", "Mejora"]


def configurar_semillas():
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Semillas aleatorias configuradas.")


def seleccionar_dispositivo():
    print("\n1. CONFIGURANDO DISPOSITIVO")
    print("-" * 60)

    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print("GPU disponible:", torch.cuda.get_device_name(0))
    else:
        dispositivo = torch.device("cpu")
        print("CUDA no está disponible. Se utilizará la CPU.")

    print(f"Dispositivo seleccionado: {dispositivo}")

    return dispositivo


def cargar_datos():
    print("\n2. CARGANDO DATOS")
    print("-" * 60)

    for ruta in (ARCHIVO_DATOS, ARCHIVO_METADATOS):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}. Ejecuta primero "
                "p23preparacion_datos_cnn.py"
            )

    datos = np.load(ARCHIVO_DATOS, allow_pickle=False)

    conjuntos = {
        "Entrenamiento": (datos["X_train"], datos["y_train"]),
        "Validación": (datos["X_val"], datos["y_val"]),
        "Prueba": (datos["X_test"], datos["y_test"])
    }

    with open(ARCHIVO_METADATOS, "r", encoding="utf-8") as archivo:
        metadatos = json.load(archivo)

    for nombre, (X, y) in conjuntos.items():
        print(f"{nombre}: X={X.shape}, y={y.shape}")

        if X.shape[1:] != (CANALES_ENTRADA, LONGITUD_SECUENCIA):
            raise ValueError(f"{nombre}: dimensiones incorrectas en X.")

        if len(X) != len(y):
            raise ValueError(f"{nombre}: X e y tienen tamaños diferentes.")

        if not np.isfinite(X).all():
            raise ValueError(f"{nombre}: existen valores inválidos en X.")

        if not np.isin(y, [0, 1, 2]).all():
            raise ValueError(f"{nombre}: existen etiquetas inválidas.")

    print("Datos cargados y dimensiones verificadas.")

    return (
        *conjuntos["Entrenamiento"],
        *conjuntos["Validación"],
        *conjuntos["Prueba"],
        metadatos
    )


def crear_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, dispositivo):
    print("\n3. CONSTRUYENDO DATASET Y DATALOADER")
    print("-" * 60)

    def dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    pin_memory = dispositivo.type == "cuda"

    generador = torch.Generator()
    generador.manual_seed(SEMILLA)

    ds_train = dataset(X_train, y_train)

    loader_train = DataLoader(
        ds_train,
        batch_size=TAMANO_LOTE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generador
    )

    loader_val = DataLoader(
        dataset(X_val, y_val),
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    loader_test = DataLoader(
        dataset(X_test, y_test),
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    print(f"Muestras de entrenamiento: {len(ds_train):,}")
    print(f"Lotes de entrenamiento: {len(loader_train)}")

    return loader_train, loader_val, loader_test


class RentControlCNN(nn.Module):
    """
    CNN 1D.
    Entrada:  [lote, 2 canales, 3 meses]
    Capas:    Conv1d -> BatchNorm1d -> ReLU (x2) ->
              AdaptiveAvgPool1d -> Dropout -> Linear
    Salida:   3 puntuaciones (una por clase).
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=2
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(p=0.25)

        self.clasificador = nn.Linear(32, NUM_CLASES)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        return self.clasificador(x)


def entrenar_epoca(modelo, loader, criterio, optimizador, dispositivo):
    modelo.train()

    perdida_acumulada = 0.0
    predicciones = []
    reales = []
    total = 0

    for entradas, etiquetas in loader:
        entradas = entradas.to(dispositivo, non_blocking=True)
        etiquetas = etiquetas.to(dispositivo, non_blocking=True)

        optimizador.zero_grad(set_to_none=True)

        salidas = modelo(entradas)

        perdida = criterio(salidas, etiquetas)

        perdida.backward()

        optimizador.step()

        tamano = entradas.size(0)

        perdida_acumulada += perdida.item() * tamano
        total += tamano

        predicciones.extend(torch.argmax(salidas, dim=1).cpu().numpy())
        reales.extend(etiquetas.cpu().numpy())

    return perdida_acumulada / total, accuracy_score(reales, predicciones)


def evaluar_perdida(modelo, loader, criterio, dispositivo):
    modelo.eval()

    perdida_acumulada = 0.0
    total = 0

    with torch.no_grad():
        for entradas, etiquetas in loader:
            entradas = entradas.to(dispositivo, non_blocking=True)
            etiquetas = etiquetas.to(dispositivo, non_blocking=True)

            perdida = criterio(modelo(entradas), etiquetas)

            tamano = entradas.size(0)

            perdida_acumulada += perdida.item() * tamano
            total += tamano

    return perdida_acumulada / total


def evaluar_prueba(modelo, loader_test, dispositivo):
    print("\n6. EVALUACIÓN FINAL DEL MODELO")
    print("-" * 60)

    modelo.eval()

    predicciones = []
    reales = []

    with torch.no_grad():
        for entradas, etiquetas in loader_test:
            salidas = modelo(entradas.to(dispositivo, non_blocking=True))

            predicciones.extend(torch.argmax(salidas, dim=1).cpu().numpy())
            reales.extend(etiquetas.numpy())

    metricas = {
        "accuracy": float(accuracy_score(reales, predicciones)),
        "precision_macro": float(precision_score(
            reales, predicciones, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            reales, predicciones, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            reales, predicciones, average="macro", zero_division=0
        ))
    }

    matriz = confusion_matrix(reales, predicciones, labels=[0, 1, 2])

    reporte = classification_report(
        reales,
        predicciones,
        labels=[0, 1, 2],
        target_names=NOMBRES_CLASES,
        zero_division=0
    )

    print(f"Accuracy:  {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision_macro']:.4f}")
    print(f"Recall:    {metricas['recall_macro']:.4f}")
    print(f"F1-score:  {metricas['f1_macro']:.4f}")
    print("\nMATRIZ DE CONFUSIÓN")
    print(matriz)
    print("\nREPORTE DE CLASIFICACIÓN")
    print(reporte)

    return metricas, matriz, reporte


def guardar_resultados(
    modelo, historial, matriz, reporte, metricas,
    metadatos, mejor_epoca, mejor_perdida
):
    print("\n7. GUARDANDO MODELO Y RESULTADOS")
    print("-" * 60)

    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    ruta_modelo = RUTA_RESULTADOS / "rentcontrol_cnn1d.pth"

    torch.save({
        "model_state_dict": modelo.state_dict(),
        "arquitectura": "RentControlCNN",
        "canales_entrada": CANALES_ENTRADA,
        "longitud_secuencia": LONGITUD_SECUENCIA,
        "numero_clases": NUM_CLASES,
        "mapa_etiquetas": MAPA_ETIQUETAS,
        "metadatos_preparacion": metadatos,
        "mejor_epoca": mejor_epoca,
        "mejor_perdida_validacion": mejor_perdida,
        "metricas_prueba": metricas,
        "version_pytorch": torch.__version__
    }, ruta_modelo)

    pd.DataFrame(historial).to_csv(
        RUTA_RESULTADOS / "historial_entrenamiento_cnn.csv",
        index=False
    )

    pd.DataFrame(
        matriz,
        index=NOMBRES_CLASES,
        columns=NOMBRES_CLASES
    ).to_csv(RUTA_RESULTADOS / "matriz_confusion_cnn.csv")

    with open(
        RUTA_RESULTADOS / "reporte_clasificacion_cnn.txt",
        "w",
        encoding="utf-8"
    ) as archivo:
        archivo.write(reporte)

    with open(
        RUTA_RESULTADOS / "metricas_cnn.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            {
                "mejor_epoca": mejor_epoca,
                "mejor_perdida_validacion": mejor_perdida,
                "metricas_prueba": metricas
            },
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print(f"Modelo: {ruta_modelo}")
    print(f"Resultados guardados en: {RUTA_RESULTADOS}")


def main():
    print("=" * 65)
    print("RENTCONTROL")
    print("PRÁCTICA 24 - ENTRENAMIENTO CNN 1D")
    print("=" * 65)

    configurar_semillas()

    dispositivo = seleccionar_dispositivo()

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        metadatos
    ) = cargar_datos()

    loader_train, loader_val, loader_test = crear_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, dispositivo
    )

    print("\n4. CONSTRUYENDO CNN 1D")
    print("-" * 60)

    modelo = RentControlCNN().to(dispositivo)

    print(modelo)

    parametros = sum(
        p.numel() for p in modelo.parameters() if p.requires_grad
    )

    print(f"\nParámetros entrenables: {parametros:,}")

    criterio = nn.CrossEntropyLoss()

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=TASA_APRENDIZAJE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )

    print("\n5. INICIANDO ENTRENAMIENTO")
    print("-" * 60)

    historial = []
    mejor_perdida = float("inf")
    mejor_epoca = 0
    mejor_estado = None
    epocas_sin_mejora = 0

    for epoca in range(1, NUM_EPOCAS + 1):

        perdida_train, accuracy_train = entrenar_epoca(
            modelo, loader_train, criterio, optimizador, dispositivo
        )

        perdida_val = evaluar_perdida(
            modelo, loader_val, criterio, dispositivo
        )

        scheduler.step(perdida_val)

        tasa_actual = optimizador.param_groups[0]["lr"]

        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            "accuracy_train": accuracy_train,
            "perdida_val": perdida_val,
            "tasa_aprendizaje": tasa_actual
        })

        print(
            f"Época {epoca:02d}/{NUM_EPOCAS} | "
            f"Loss Train: {perdida_train:.4f} | "
            f"Acc Train: {accuracy_train:.4f} | "
            f"Loss Val: {perdida_val:.4f} | "
            f"LR: {tasa_actual:.6f}"
        )

        if perdida_val < mejor_perdida:
            mejor_perdida = perdida_val
            mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
            print("  -> Nuevo mejor modelo guardado en memoria.")
        else:
            epocas_sin_mejora += 1

        if epocas_sin_mejora >= PACIENCIA_EARLY_STOPPING:
            print("\nEarly Stopping activado: la pérdida de validación "
                  "dejó de mejorar.")
            break

    if mejor_estado is None:
        raise RuntimeError("No se obtuvo un modelo válido.")

    modelo.load_state_dict(mejor_estado)

    print("\nMEJOR MODELO")
    print(f"Época seleccionada: {mejor_epoca}")
    print(f"Mejor pérdida de validación: {mejor_perdida:.4f}")

    metricas, matriz, reporte = evaluar_prueba(
        modelo, loader_test, dispositivo
    )

    guardar_resultados(
        modelo, historial, matriz, reporte, metricas,
        metadatos, mejor_epoca, mejor_perdida
    )

    print("\n" + "=" * 65)
    print("PRÁCTICA 24 FINALIZADA CORRECTAMENTE")
    print("CNN 1D entrenada y evaluada.")
    print("=" * 65)


if __name__ == "__main__":
    main()
