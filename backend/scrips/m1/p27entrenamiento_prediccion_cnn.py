# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 27: CNN 1D PARA PREDECIR LA PUNTUALIDAD DEL MES 3
#
# Archivo:
# p27entrenamiento_prediccion_cnn.py
#
# Objetivo:
# Entrenar una CNN 1D de REGRESIÓN que estima el puntaje de
# puntualidad de pago del mes 3 (0 a 10) a partir de los meses
# 1 y 2 (puntualidad y cobertura de renta).
#
# Métricas: MAE, MSE, RMSE y R2, comparadas contra un modelo
# base que siempre predice el promedio del entrenamiento.
#
# Archivos de salida (carpeta resultados_prediccion_cnn):
#   rentcontrol_cnn_regresion.pth, historial_entrenamiento_regresion.csv,
#   metricas_prediccion_cnn.json, curva_entrenamiento_regresion.png,
#   predicciones_real_vs_predicho.png, predicciones_prueba_cnn.csv
# ============================================================

import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Guardar gráficas sin abrir ventanas.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_DATOS = RUTA_PROYECTO / "resultados_prediccion_cnn"

RUTA_SALIDA = RUTA_DATOS

ARCHIVO_DATOS = RUTA_DATOS / "rentcontrol_prediccion_mes3.npz"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
BATCH_SIZE = 256
MAX_EPOCAS = 60
LEARNING_RATE = 0.001
PACIENCIA = 10

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)
    dispositivo = torch.device("cuda")
else:
    dispositivo = torch.device("cpu")

print("=" * 65)
print("RENTCONTROL")
print("PRÁCTICA 27 - CNN 1D PARA PREDICCIÓN DE PUNTUALIDAD (MES 3)")
print("=" * 65)
print("\nDISPOSITIVO DE ENTRENAMIENTO:", dispositivo)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. CARGAR DATOS
# ============================================================

print("\nCARGANDO DATOS PREPARADOS...")

if not ARCHIVO_DATOS.exists():
    raise FileNotFoundError(
        f"No existe el archivo: {ARCHIVO_DATOS}. "
        "Ejecuta primero p26preparacion_prediccion_cnn.py"
    )

datos = np.load(ARCHIVO_DATOS, allow_pickle=False)

X_train, y_train = datos["X_train"], datos["y_train"]
X_val, y_val = datos["X_val"], datos["y_val"]
X_test, y_test = datos["X_test"], datos["y_test"]

print("Entrenamiento:", X_train.shape)
print("Validación:", X_val.shape)
print("Prueba:", X_test.shape)

for Xc, yc in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
    assert Xc.shape[1:] == (2, 2)
    assert len(Xc) == len(yc)
    assert np.isfinite(Xc).all() and np.isfinite(yc).all()

print("Validaciones correctas.")


class DatasetCobranza(Dataset):
    """
    Cada muestra: 2 canales (puntualidad, cobertura) x 2 meses.
    El objetivo es la puntualidad del mes 3.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, indice):
        return self.X[indice], self.y[indice]


def crear_loader(X, y, mezclar):
    return DataLoader(
        DatasetCobranza(X, y),
        batch_size=BATCH_SIZE,
        shuffle=mezclar,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


loader_train = crear_loader(X_train, y_train, True)
loader_val = crear_loader(X_val, y_val, False)
loader_test = crear_loader(X_test, y_test, False)

print(
    f"\nLotes: entrenamiento={len(loader_train)}, "
    f"validación={len(loader_val)}, prueba={len(loader_test)}"
)

# ============================================================
# 3. MODELO
# ============================================================

class CNNRegresionCobranza(nn.Module):
    """
    Entrada: (lote, 2 canales, 2 meses)
    Salida:  una puntuación de puntualidad estimada por registro.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 16, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.20)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        return self.fc(x).squeeze(1)


print("\nCONSTRUYENDO CNN 1D...")

modelo = CNNRegresionCobranza().to(dispositivo)

total_parametros = sum(
    p.numel() for p in modelo.parameters() if p.requires_grad
)

print(modelo)
print("\nParámetros entrenables:", total_parametros)

funcion_perdida = nn.MSELoss()

optimizador = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizador,
    mode="min",
    factor=0.5,
    patience=4
)

# ============================================================
# 4. ENTRENAMIENTO
# ============================================================

def entrenar_epoca(modelo, loader):
    modelo.train()

    acumulada = 0.0
    total = 0

    for entradas, objetivos in loader:
        entradas = entradas.to(dispositivo)
        objetivos = objetivos.to(dispositivo)

        optimizador.zero_grad()

        perdida = funcion_perdida(modelo(entradas), objetivos)

        perdida.backward()

        optimizador.step()

        acumulada += perdida.item() * len(entradas)
        total += len(entradas)

    return acumulada / total


def validar_epoca(modelo, loader):
    modelo.eval()

    acumulada = 0.0
    total = 0

    with torch.no_grad():
        for entradas, objetivos in loader:
            entradas = entradas.to(dispositivo)
            objetivos = objetivos.to(dispositivo)

            perdida = funcion_perdida(modelo(entradas), objetivos)

            acumulada += perdida.item() * len(entradas)
            total += len(entradas)

    return acumulada / total


print("\nINICIANDO ENTRENAMIENTO...")

historial = []
mejor_perdida = float("inf")
mejor_epoca = 0
mejores_pesos = None
epocas_sin_mejora = 0

inicio_entrenamiento = time.perf_counter()

for epoca in range(1, MAX_EPOCAS + 1):

    perdida_train = entrenar_epoca(modelo, loader_train)

    perdida_val = validar_epoca(modelo, loader_val)

    scheduler.step(perdida_val)

    tasa_actual = optimizador.param_groups[0]["lr"]

    historial.append({
        "epoca": epoca,
        "perdida_train": perdida_train,
        "perdida_val": perdida_val,
        "learning_rate": tasa_actual
    })

    print(
        f"Época {epoca:02d}/{MAX_EPOCAS} | "
        f"Train: {perdida_train:.5f} | "
        f"Val: {perdida_val:.5f} | "
        f"LR: {tasa_actual:.6f}"
    )

    if perdida_val < mejor_perdida:
        mejor_perdida = perdida_val
        mejor_epoca = epoca
        mejores_pesos = copy.deepcopy(modelo.state_dict())
        epocas_sin_mejora = 0
    else:
        epocas_sin_mejora += 1

    if epocas_sin_mejora >= PACIENCIA:
        print("\nEARLY STOPPING ACTIVADO")
        break

tiempo_entrenamiento = time.perf_counter() - inicio_entrenamiento

print("\nENTRENAMIENTO FINALIZADO")
print("Mejor época:", mejor_epoca)
print(f"Mejor pérdida de validación: {mejor_perdida:.6f}")
print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")

if mejores_pesos is None:
    raise RuntimeError("No se encontraron pesos válidos.")

modelo.load_state_dict(mejores_pesos)

modelo.eval()

# ============================================================
# 5. EVALUACIÓN EN PRUEBA
# ============================================================

def obtener_predicciones(modelo, loader):
    modelo.eval()

    predicciones = []
    reales = []

    with torch.no_grad():
        for entradas, objetivos in loader:
            resultado = modelo(entradas.to(dispositivo))

            predicciones.extend(resultado.cpu().numpy())
            reales.extend(objetivos.numpy())

    return np.asarray(predicciones), np.asarray(reales)


print("\nEVALUANDO CONJUNTO DE PRUEBA...")

inicio_inferencia = time.perf_counter()

predicciones_test, reales_test = obtener_predicciones(modelo, loader_test)

tiempo_inferencia = time.perf_counter() - inicio_inferencia

mae = mean_absolute_error(reales_test, predicciones_test)
mse = mean_squared_error(reales_test, predicciones_test)
rmse = float(np.sqrt(mse))
r2 = r2_score(reales_test, predicciones_test)

print("\nRESULTADOS DE LA CNN 1D")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Modelo base: predecir siempre el promedio del entrenamiento.
prediccion_base = np.full(len(y_test), y_train.mean())

mae_base = mean_absolute_error(y_test, prediccion_base)
rmse_base = float(np.sqrt(mean_squared_error(y_test, prediccion_base)))

print("\nMODELO BASE: PREDICCIÓN DEL PROMEDIO")
print(f"MAE base:  {mae_base:.4f}")
print(f"RMSE base: {rmse_base:.4f}")

# ============================================================
# 6. GUARDAR MODELO, MÉTRICAS Y GRÁFICAS
# ============================================================

ARCHIVO_MODELO = RUTA_SALIDA / "rentcontrol_cnn_regresion.pth"

torch.save({
    "model_state_dict": modelo.state_dict(),
    "mejor_epoca": mejor_epoca,
    "mejor_perdida_validacion": mejor_perdida,
    "arquitectura": "CNNRegresionCobranza",
    "canales": 2,
    "meses_entrada": 2,
    "variable_objetivo": "puntualidad_m3",
    "parametros_entrenables": total_parametros
}, ARCHIVO_MODELO)

df_historial = pd.DataFrame(historial)

df_historial.to_csv(
    RUTA_SALIDA / "historial_entrenamiento_regresion.csv",
    index=False
)

metricas = {
    "proyecto": "RentControl",
    "practica": 27,
    "tipo_problema": "Regresión",
    "mejor_epoca": int(mejor_epoca),
    "mejor_perdida_validacion": float(mejor_perdida),
    "metricas_prueba": {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    },
    "modelo_base": {
        "MAE": float(mae_base),
        "RMSE": float(rmse_base)
    },
    "tiempos": {
        "entrenamiento_segundos": float(tiempo_entrenamiento),
        "inferencia_segundos": float(tiempo_inferencia)
    },
    "advertencia": (
        "Datos sintéticos. Los resultados no representan una "
        "validación con inquilinos reales."
    )
}

with open(
    RUTA_SALIDA / "metricas_prediccion_cnn.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(metricas, archivo, indent=4, ensure_ascii=False)

# Curva de entrenamiento.
plt.figure(figsize=(10, 5))
plt.plot(df_historial["epoca"], df_historial["perdida_train"], label="Entrenamiento")
plt.plot(df_historial["epoca"], df_historial["perdida_val"], label="Validación")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("RentControl - Pérdida de entrenamiento")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "curva_entrenamiento_regresion.png", dpi=300)
plt.close()

# Real vs predicho.
plt.figure(figsize=(8, 7))
plt.scatter(reales_test, predicciones_test, alpha=0.25)
plt.plot([0, 10], [0, 10], linestyle="--")
plt.xlabel("Puntualidad real del mes 3")
plt.ylabel("Puntualidad predicha")
plt.title("RentControl - Real vs Predicho")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "predicciones_real_vs_predicho.png", dpi=300)
plt.close()

# Predicciones por inquilino y tipo de inmueble.
df_predicciones = pd.DataFrame({
    "id_inquilino": datos["ids_test"],
    "tipo_inmueble": datos["tipos_test"],
    "puntualidad_real_m3": reales_test,
    "puntualidad_predicha_m3": predicciones_test,
    "error_absoluto": np.abs(reales_test - predicciones_test)
})

df_predicciones.to_csv(
    RUTA_SALIDA / "predicciones_prueba_cnn.csv",
    index=False
)

print("\n" + "=" * 65)
print("PRÁCTICA 27 FINALIZADA")
print("=" * 65)
print(f"\nMAE CNN:  {mae:.4f}  |  MAE base:  {mae_base:.4f}")
print(f"RMSE CNN: {rmse:.4f}  |  RMSE base: {rmse_base:.4f}")
print(f"R2:       {r2:.4f}")
print(f"\nArchivos generados en: {RUTA_SALIDA}")
print("NO SE UTILIZÓ EL MES 3 COMO ENTRADA.")
print("=" * 65)
