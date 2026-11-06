# ============================================================
# PRACTICA 1: BUSQUEDA FRENTE A APRENDIZAJE AUTOMATICO
# PARTE A: RECORRER UN FLUJO DE RENTCONTROL MEDIANTE BFS
# ============================================================

# Objetivo de aprendizaje:
# Distinguir una busqueda determinista de un modelo que aprende
# a partir de datos.
#
# RENTCONTROL administra propiedades, inquilinos, contratos,
# pagos y mantenimientos. En esta parte buscaremos una ruta entre
# modulos conectados del sistema. En la parte B aplicaremos una
# regla fija a un pago ficticio.

from collections import deque

# Cada modulo es un nodo y cada enlace representa una navegacion
# disponible en el flujo de trabajo de ejemplo.
mapa_modulos = {
    "Dashboard": ["Propiedades", "Cobranza", "Alertas"],
    "Propiedades": ["Inquilinos", "Contratos"],
    "Inquilinos": ["Contratos"],
    "Contratos": ["Cobranza", "Mantenimientos"],
    "Cobranza": ["Movimientos", "Alertas"],
    "Alertas": ["Mantenimientos"],
    "Movimientos": [],
    "Mantenimientos": ["Ayuda"],
    "Ayuda": []
}


def buscar_ruta(grafo, inicio, objetivo):
    """Devuelve la ruta mas corta en numero de enlaces, o None."""
    if inicio not in grafo or objetivo not in grafo:
        return None

    cola = deque([[inicio]])
    visitados = {inicio}

    while cola:
        ruta = cola.popleft()
        modulo_actual = ruta[-1]

        if modulo_actual == objetivo:
            return ruta

        for vecino in grafo[modulo_actual]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    return None


inicio = "Dashboard"
objetivo = "Mantenimientos"
ruta = buscar_ruta(mapa_modulos, inicio, objetivo)

print("=== RUTA ENTRE MODULOS DE RENTCONTROL ===")
if ruta is not None:
    print(" -> ".join(ruta))
    print("Numero de enlaces:", len(ruta) - 1)
else:
    print("No existe una ruta hacia el modulo solicitado.")
