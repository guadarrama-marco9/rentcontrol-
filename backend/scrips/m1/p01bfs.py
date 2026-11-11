# ============================================================
# PRACTICA 1: BUSQUEDA FRENTE A APRENDIZAJE AUTOMATICO
# PARTE A: ENCONTRAR UNA RUTA MEDIANTE BFS
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Planteamiento:
# RENTCONTROL es un sistema de administracion de rentas.
# El sistema necesita realizar dos tareas:
#
# Problema A. Encontrar la ruta mas corta para que el administrador
#             vaya desde la oficina hasta un departamento, usando un
#             mapa de edificios y conexiones conocidas.
#
# Problema B. Estimar si un inquilino pagara tarde, usando ejemplos
#             historicos de pagos.
#
# Pregunta central: ¿ambos problemas se resuelven con el mismo
# tipo de algoritmo?
#
# Al terminar, deberas poder:
# - Identificar entradas, estados, objetivos y resultados.
# - Distinguir un algoritmo tradicional de un modelo de ML.
# - Implementar dos soluciones sencillas en Python.
# - Explicar cuando conviene cada enfoque.

from collections import deque

# ------------------------------------------------------------
# 1. REPRESENTAR EL MAPA COMO UN GRAFO
# ------------------------------------------------------------
# Cada lugar es un nodo. Cada conexion es un camino directo.

mapa = {
    "Oficina": ["Edificio A", "Edificio B"],
    "Edificio A": ["Departamento 101", "Pasillo central"],
    "Edificio B": ["Estacionamiento", "Departamento 201"],
    "Pasillo central": ["Departamento 201"],
    "Estacionamiento": ["Departamento 201"],
    "Departamento 101": [],
    "Departamento 201": []
}


# ------------------------------------------------------------
# 2. BUSQUEDA EN ANCHURA (BFS)
# ------------------------------------------------------------
def buscar_ruta(grafo, inicio, objetivo):

    # BFS usa una cola FIFO: primero entra, primero sale.
    cola = deque([[inicio]])

    # Evita explorar dos veces el mismo nodo.
    visitados = {inicio}

    while cola:

        ruta = cola.popleft()
        nodo_actual = ruta[-1]

        if nodo_actual == objetivo:
            return ruta

        for vecino in grafo[nodo_actual]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    # Si se agota la cola, no existe ruta.
    return None


# ------------------------------------------------------------
# 3. EJECUTAR EL ALGORITMO
# ------------------------------------------------------------
inicio = "Oficina"
objetivo = "Departamento 201"

ruta = buscar_ruta(mapa, inicio, objetivo)

print("=== RESULTADO DE LA BUSQUEDA ===")

if ruta is not None:
    print("Ruta encontrada:")
    print(" -> ".join(ruta))
    print("Numero de conexiones:", len(ruta) - 1)
else:
    print("No existe una ruta hacia el destino.")
