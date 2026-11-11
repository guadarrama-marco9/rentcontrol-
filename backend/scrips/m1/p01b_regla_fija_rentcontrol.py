# ============================================================
# PARTE B: TOMAR UNA DECISION MEDIANTE UNA REGLA FIJA
# ADAPTACION AL PROYECTO: RENTCONTROL
# ============================================================
#
# Ahora resolvemos un problema distinto: clasificar el riesgo de
# pago de un inquilino a partir de sus pagos atrasados previos.
#
# Primero usamos una regla fija. Despues (practicas posteriores)
# la sustituiremos por un modelo que aprende de ejemplos.
#
# Los datos son ficticios.

inquilinos = [
    {"nombre": "Ana", "pagos_atrasados_previos": 0},
    {"nombre": "Luis", "pagos_atrasados_previos": 5},
    {"nombre": "Maria", "pagos_atrasados_previos": 1},
    {"nombre": "Carlos", "pagos_atrasados_previos": 3}
]


# ------------------------------------------------------------
# DEFINIR UNA REGLA MANUAL
# ------------------------------------------------------------
# Regla solo didactica: no representa una relacion real entre
# atrasos previos y morosidad.

def predecir_con_regla(pagos_atrasados):

    if pagos_atrasados >= 3:
        return "Riesgo de pago atrasado"

    return "Probablemente paga a tiempo"


# ------------------------------------------------------------
# APLICAR LA REGLA A CADA INQUILINO
# ------------------------------------------------------------
print("\n=== RESULTADO DE LA REGLA FIJA ===")

for inquilino in inquilinos:

    atrasos = inquilino["pagos_atrasados_previos"]
    resultado = predecir_con_regla(atrasos)

    print(
        inquilino["nombre"],
        "| Atrasos previos:", atrasos,
        "| Resultado:", resultado
    )


# ------------------------------------------------------------
# PROBAR CON UN CASO NUEVO
# ------------------------------------------------------------
atrasos_nuevo_inquilino = 2

resultado = predecir_con_regla(atrasos_nuevo_inquilino)

print("\n=== NUEVO CASO ===")
print("Atrasos previos:", atrasos_nuevo_inquilino)
print("Resultado:", resultado)

# Reflexion: la regla ignora ingreso, comunicacion, monto de la renta,
# etc. Un modelo de ML puede combinar varias variables y aprender el
# umbral a partir de los datos en lugar de fijarlo a mano.
