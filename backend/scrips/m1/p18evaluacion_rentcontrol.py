# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 18: EVALUACIÓN INTELIGENTE DE INQUILINOS
#
# Archivo:
# p18evaluacion_rentcontrol.py
#
# Objetivo:
# Aplicación web que evalúa a un inquilino con los modelos
# entrenados (Random Forest, K-Means y PCA), genera
# recomendaciones y guarda el resultado en MongoDB local.
#
# Ejecución (primero entrenar los modelos una sola vez):
#   python rc_motor_ia.py
#   streamlit run p18evaluacion_rentcontrol.py
# ============================================================

import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo.errors import PyMongoError

import rc_mongo
import rc_motor_ia as motor

SISTEMA = "RentControl"

st.set_page_config(
    page_title="RentControl - Evaluación",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 RENTCONTROL")
st.subheader("Sistema inteligente de evaluación de inquilinos")

st.markdown(
    """
    Esta aplicación integra tres técnicas de inteligencia artificial:
    - **Random Forest:** estimación de cumplimiento de pagos.
    - **K-Means:** identificación de perfiles de pago.
    - **PCA:** reducción de dimensionalidad para análisis exploratorio.
    """
)

st.divider()


# ------------------------------------------------------------
# 1. CARGA DE MODELOS (sin entrenar)
# ------------------------------------------------------------

@st.cache_resource
def cargar_modelos_rentcontrol():
    return motor.cargar_modelos()


try:
    modelos = cargar_modelos_rentcontrol()

except FileNotFoundError as error:
    st.error(
        "No se encontraron los modelos. Ejecuta primero "
        "`python rc_motor_ia.py`."
    )
    st.exception(error)
    st.stop()

except Exception as error:
    st.error("No fue posible cargar los modelos.")
    st.exception(error)
    st.stop()

st.success("Los modelos de Machine Learning se cargaron correctamente.")


# ------------------------------------------------------------
# 2. CONEXIÓN A MONGODB
# ------------------------------------------------------------

def conectar_mongodb():
    """
    Conecta con MongoDB local y prepara los índices.
    """

    cliente, base_datos = rc_mongo.conectar()

    coleccion = base_datos[rc_mongo.MONGO_COLLECTION]

    coleccion.create_index("id_analisis", unique=True)
    coleccion.create_index("id_inquilino")
    coleccion.create_index("fecha_registro")

    return cliente, coleccion


# ------------------------------------------------------------
# 3. FORMULARIO
# ------------------------------------------------------------

st.header("1. Información del inquilino")

st.info(
    "Utiliza el mismo identificador en evaluaciones posteriores "
    "para consultar el historial del inquilino."
)

with st.form("formulario_evaluacion"):

    id_inquilino = st.text_input(
        "Identificador permanente del inquilino",
        placeholder="Ejemplo: INQ-2026-001"
    )

    col1, col2 = st.columns(2)

    with col1:
        pagos_a_tiempo = st.number_input(
            "Porcentaje de pagos realizados a tiempo",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

        dias_retraso = st.number_input(
            "Días de retraso promedio",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5
        )

    with col2:
        antiguedad = st.number_input(
            "Antigüedad como inquilino (meses)",
            min_value=0,
            max_value=120,
            value=12,
            step=1
        )

        incidencias = st.number_input(
            "Incidencias registradas",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    enviar = st.form_submit_button(
        "Evaluar inquilino y guardar resultados",
        use_container_width=True
    )


# ------------------------------------------------------------
# 4. ANÁLISIS Y GUARDADO
# ------------------------------------------------------------

if enviar:

    if not id_inquilino.strip():
        st.error("Debes proporcionar el identificador del inquilino.")
        st.stop()

    inquilino = pd.DataFrame(
        [[
            float(pagos_a_tiempo),
            float(dias_retraso),
            float(antiguedad),
            float(incidencias)
        ]],
        columns=motor.COLUMNAS
    )

    st.divider()
    st.header("2. Resultados del análisis inteligente")

    try:
        resultado_prediccion = motor.predecir_cumplimiento(
            inquilino,
            modelos["supervisado"]
        )

        resultado_perfil = motor.identificar_perfil(
            inquilino,
            modelos["kmeans"],
            modelos["escalador_kmeans"],
            modelos["mapa_perfiles"]
        )

        resultado_pca = motor.proyectar_pca(
            inquilino,
            modelos["pca"],
            modelos["escalador_pca"]
        )

        recomendaciones = motor.generar_recomendaciones(
            inquilino,
            resultado_prediccion,
            resultado_perfil
        )

    except Exception as error:
        st.error("Ocurrió un problema durante el análisis.")
        st.exception(error)
        st.stop()

    st.subheader("Predicción de cumplimiento")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Probabilidad de cumplir",
            f"{resultado_prediccion['probabilidad_cumplir'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Probabilidad de incumplir",
            f"{resultado_prediccion['probabilidad_incumplir'] * 100:.2f}%"
        )

    with col3:
        st.metric("Perfil de pago", resultado_perfil["perfil"])

    if resultado_prediccion["prediccion"] == 1:
        st.success(resultado_prediccion["resultado"])
    else:
        st.warning(resultado_prediccion["resultado"])

    st.caption(
        "Las probabilidades son estimaciones del modelo y no "
        "constituyen una decisión definitiva sobre el inquilino."
    )

    st.subheader("Análisis de aprendizaje no supervisado")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Cluster identificado", resultado_perfil["cluster"])
        st.write("Perfil:", resultado_perfil["perfil"])

    with col2:
        st.metric("Componente principal PC1", round(resultado_pca["PC1"], 4))
        st.metric("Componente principal PC2", round(resultado_pca["PC2"], 4))

    st.subheader("Recomendaciones para el administrador")

    for numero, recomendacion in enumerate(recomendaciones, start=1):
        st.write(f"**{numero}.** {recomendacion}")

    # --------------------------------------------------------
    # Documento que se guardará en MongoDB.
    # Las probabilidades se guardan como fracción (0 a 1).
    # --------------------------------------------------------

    id_analisis = str(uuid.uuid4())

    documento = {
        "id_analisis": id_analisis,
        "id_inquilino": id_inquilino.strip(),
        "sistema": SISTEMA,
        "practica": "p18",
        "fecha_registro": datetime.now(timezone.utc),
        "archivo_origen": "p18evaluacion_rentcontrol.py",
        "datos_reporte": {
            "id_inquilino": id_inquilino.strip(),
            "pagos_a_tiempo": float(pagos_a_tiempo),
            "dias_retraso_promedio": float(dias_retraso),
            "antiguedad_meses": int(antiguedad),
            "incidencias": int(incidencias),
            "prediccion": int(resultado_prediccion["prediccion"]),
            "resultado": resultado_prediccion["resultado"],
            "probabilidad_cumplir": float(
                resultado_prediccion["probabilidad_cumplir"]
            ),
            "probabilidad_incumplir": float(
                resultado_prediccion["probabilidad_incumplir"]
            ),
            "cluster": int(resultado_perfil["cluster"]),
            "perfil": resultado_perfil["perfil"],
            "PC1": float(resultado_pca["PC1"]),
            "PC2": float(resultado_pca["PC2"]),
            "recomendaciones": recomendaciones
        }
    }

    cliente = None

    try:
        with st.spinner("Conectando con MongoDB..."):
            cliente, coleccion = conectar_mongodb()
            coleccion.insert_one(documento)

        st.success("La evaluación se almacenó correctamente en MongoDB.")
        st.write("**Identificador del inquilino:**", id_inquilino.strip())
        st.write("**Identificador de la evaluación:**", id_analisis)
        st.write("**Base de datos:**", rc_mongo.MONGO_DB)
        st.write("**Colección:**", rc_mongo.MONGO_COLLECTION)

    except PyMongoError as error:
        st.error("No fue posible guardar la evaluación en MongoDB.")
        st.exception(error)

    except Exception as error:
        st.error("Ocurrió un error inesperado al guardar la evaluación.")
        st.exception(error)

    finally:
        if cliente is not None:
            cliente.close()


# ------------------------------------------------------------
# 5. HISTORIAL
# ------------------------------------------------------------

st.divider()
st.header("3. Historial de evaluaciones")

id_consulta = st.text_input(
    "Identificador del inquilino para consultar su historial",
    key="id_consulta",
    placeholder="Ejemplo: INQ-2026-001"
)

if st.button("Consultar historial", use_container_width=True):

    if not id_consulta.strip():
        st.warning("Escribe el identificador del inquilino.")

    else:
        cliente = None

        try:
            cliente, coleccion = conectar_mongodb()

            registros = list(
                coleccion.find(
                    {
                        "id_inquilino": id_consulta.strip(),
                        "sistema": SISTEMA
                    },
                    {
                        "_id": 0,
                        "id_analisis": 1,
                        "fecha_registro": 1,
                        "datos_reporte": 1
                    }
                ).sort("fecha_registro", -1)
            )

            if registros:
                st.success(f"Se encontraron {len(registros)} evaluaciones.")

                filas = []

                for registro in registros:
                    datos = registro.get("datos_reporte", {})

                    filas.append({
                        "ID análisis": registro.get("id_analisis"),
                        "Fecha": registro.get("fecha_registro"),
                        "Pagos a tiempo (%)": datos.get("pagos_a_tiempo"),
                        "Días de retraso": datos.get("dias_retraso_promedio"),
                        "Antigüedad (meses)": datos.get("antiguedad_meses"),
                        "Incidencias": datos.get("incidencias"),
                        "Predicción": datos.get("resultado"),
                        "Prob. de cumplir": datos.get("probabilidad_cumplir"),
                        "Perfil": datos.get("perfil")
                    })

                st.dataframe(
                    pd.DataFrame(filas),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("No existen evaluaciones para ese inquilino.")

        except PyMongoError as error:
            st.error("No fue posible consultar MongoDB.")
            st.exception(error)

        except Exception as error:
            st.error("Ocurrió un error al consultar el historial.")
            st.exception(error)

        finally:
            if cliente is not None:
                cliente.close()

st.divider()

st.caption(
    "RENTCONTROL | Sistema de análisis y recomendaciones de rentas "
    "basado en Machine Learning."
)
