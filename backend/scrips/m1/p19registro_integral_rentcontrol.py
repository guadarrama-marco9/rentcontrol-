# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 19: REGISTRO INTEGRAL DE INQUILINOS Y CONTRATOS
#
# Archivo:
# p19registro_integral_rentcontrol.py
#
# Objetivo:
# Sistema de registro que integra:
# - Expediente permanente del inquilino.
# - Propiedades rentadas (contratos) y su renta mensual.
# - Control de pagos puntuales y saldo pendiente.
# - Análisis mediante inteligencia artificial.
#
# Colecciones (base rentcontrol):
#   inquilinos              Expediente del inquilino
#   evaluaciones_contrato   Evaluación de contratos y pagos
#   analisis_inquilinos     Análisis de IA (usa p16, p17 y p18)
#
# Ejecución:
#   streamlit run p19registro_integral_rentcontrol.py
# ============================================================

import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo.errors import PyMongoError

import rc_mongo
import rc_motor_ia as motor

SISTEMA = "RentControl"

COLECCION_INQUILINOS = "inquilinos"
COLECCION_EVALUACIONES = "evaluaciones_contrato"
COLECCION_IA = rc_mongo.MONGO_COLLECTION

st.set_page_config(
    page_title="RentControl - Registro Integral",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 RENTCONTROL")
st.subheader("Práctica 19 - Registro integral de inquilinos y contratos")

st.markdown(
    """
    Sistema de registro que integra:
    - Expediente permanente del inquilino.
    - Propiedades rentadas y renta mensual.
    - Control de pagos puntuales y saldo pendiente.
    - Análisis mediante inteligencia artificial.
    """
)

st.divider()


# ------------------------------------------------------------
# 1. MODELOS DE IA
# ------------------------------------------------------------

@st.cache_resource
def cargar_modelos():
    return motor.cargar_modelos()


try:
    modelos = cargar_modelos()
    st.success("Modelos de inteligencia artificial cargados.")

except Exception as error:
    st.error(
        "No fue posible cargar los modelos. Ejecuta primero "
        "`python rc_motor_ia.py`."
    )
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# 2. CONEXIÓN A MONGODB
# ------------------------------------------------------------

def conectar_mongodb():
    """
    Conecta con MongoDB local y prepara las colecciones.
    """

    cliente, base_datos = rc_mongo.conectar()

    c_inquilinos = base_datos[COLECCION_INQUILINOS]
    c_evaluaciones = base_datos[COLECCION_EVALUACIONES]
    c_ia = base_datos[COLECCION_IA]

    c_inquilinos.create_index("id_inquilino", unique=True)

    c_evaluaciones.create_index("id_evaluacion", unique=True)
    c_evaluaciones.create_index([
        ("id_inquilino", 1),
        ("fecha_registro", -1)
    ])

    c_ia.create_index("id_analisis", unique=True)

    return cliente, c_inquilinos, c_evaluaciones, c_ia


# ------------------------------------------------------------
# 3. VALIDACIÓN Y CÁLCULOS
# ------------------------------------------------------------

def procesar_contratos(tabla):
    """
    Valida la tabla de propiedades y calcula:
    - Pagos tardíos.
    - Porcentaje de puntualidad.
    - Monto esperado y porcentaje cobrado.
    """

    contratos = tabla.copy()

    contratos["propiedad"] = (
        contratos["propiedad"].fillna("").astype(str).str.strip()
    )

    contratos = contratos[contratos["propiedad"] != ""].copy()

    if contratos.empty:
        raise ValueError("Debes registrar al menos una propiedad.")

    numericas = [
        "renta_mensual",
        "meses_contratados",
        "pagos_puntuales",
        "saldo_pendiente"
    ]

    for columna in numericas:
        contratos[columna] = pd.to_numeric(
            contratos[columna],
            errors="coerce"
        )

    if contratos[numericas].isna().any().any():
        raise ValueError("Completa todos los datos de renta y pagos.")

    if (contratos["renta_mensual"] <= 0).any():
        raise ValueError("La renta mensual debe ser mayor que cero.")

    if (contratos["meses_contratados"] < 1).any():
        raise ValueError("Los meses contratados deben ser al menos 1.")

    if (contratos["pagos_puntuales"] < 0).any():
        raise ValueError("Los pagos puntuales no pueden ser negativos.")

    if (
        contratos["pagos_puntuales"] > contratos["meses_contratados"]
    ).any():
        raise ValueError(
            "Los pagos puntuales no pueden superar los meses contratados."
        )

    contratos["monto_esperado"] = (
        contratos["renta_mensual"] * contratos["meses_contratados"]
    )

    if (contratos["saldo_pendiente"] < 0).any():
        raise ValueError("El saldo pendiente no puede ser negativo.")

    if (contratos["saldo_pendiente"] > contratos["monto_esperado"]).any():
        raise ValueError(
            "El saldo pendiente no puede superar el monto esperado."
        )

    contratos["pagos_tardios"] = (
        contratos["meses_contratados"] - contratos["pagos_puntuales"]
    )

    contratos["porcentaje_puntualidad"] = (
        contratos["pagos_puntuales"]
        / contratos["meses_contratados"]
        * 100
    ).round(2)

    contratos["porcentaje_cobrado"] = (
        (contratos["monto_esperado"] - contratos["saldo_pendiente"])
        / contratos["monto_esperado"]
        * 100
    ).round(2)

    return contratos


def convertir_contratos_documentos(contratos):
    """
    Convierte las filas del DataFrame en documentos de MongoDB.
    """

    documentos = []

    for _, fila in contratos.iterrows():
        documentos.append({
            "propiedad": str(fila["propiedad"]),
            "renta_mensual": float(fila["renta_mensual"]),
            "meses_contratados": int(fila["meses_contratados"]),
            "pagos": {
                "puntuales": int(fila["pagos_puntuales"]),
                "tardios": int(fila["pagos_tardios"]),
                "porcentaje_puntualidad": float(
                    fila["porcentaje_puntualidad"]
                )
            },
            "cobranza": {
                "monto_esperado": float(fila["monto_esperado"]),
                "saldo_pendiente": float(fila["saldo_pendiente"]),
                "porcentaje_cobrado": float(fila["porcentaje_cobrado"])
            }
        })

    return documentos


def guardar_inquilino(
    coleccion,
    id_inquilino,
    nombre,
    telefono,
    correo,
    ocupacion,
    periodo
):
    """
    Si el inquilino existe actualiza sus datos generales; si no,
    crea el expediente. No borra evaluaciones anteriores.
    """

    ahora = datetime.now(timezone.utc)

    coleccion.update_one(
        {"id_inquilino": id_inquilino},
        {
            "$set": {
                "nombre_completo": nombre,
                "telefono": telefono,
                "correo": correo,
                "ocupacion": ocupacion,
                "periodo_contrato": periodo,
                "fecha_actualizacion": ahora
            },
            "$setOnInsert": {
                "fecha_creacion": ahora
            }
        },
        upsert=True
    )


# ------------------------------------------------------------
# 4. FORMULARIO
# ------------------------------------------------------------

st.header("1. Expediente del inquilino")

with st.form("formulario_registro", clear_on_submit=False):

    col1, col2 = st.columns(2)

    with col1:
        id_inquilino = st.text_input(
            "Identificador permanente",
            placeholder="Ejemplo: INQ-2026-001"
        )

        nombre = st.text_input("Nombre completo")

        telefono = st.text_input("Teléfono")

    with col2:
        correo = st.text_input("Correo electrónico")

        ocupacion = st.text_input("Ocupación")

        periodo = st.text_input(
            "Periodo del contrato",
            placeholder="Ejemplo: 2026-2027"
        )

    st.divider()
    st.header("2. Propiedades rentadas")

    st.write(
        "Agrega una fila por cada propiedad que renta el inquilino. "
        "Puedes añadir filas desde la tabla."
    )

    tabla_inicial = pd.DataFrame({
        "propiedad": [""],
        "renta_mensual": [5000.0],
        "meses_contratados": [1],
        "pagos_puntuales": [1],
        "saldo_pendiente": [0.0]
    })

    tabla_contratos = st.data_editor(
        tabla_inicial,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "propiedad": st.column_config.TextColumn(
                "Propiedad",
                required=True
            ),
            "renta_mensual": st.column_config.NumberColumn(
                "Renta mensual ($)",
                min_value=0.0,
                step=100.0,
                format="%.2f"
            ),
            "meses_contratados": st.column_config.NumberColumn(
                "Meses contratados",
                min_value=1,
                step=1
            ),
            "pagos_puntuales": st.column_config.NumberColumn(
                "Pagos puntuales",
                min_value=0,
                step=1
            ),
            "saldo_pendiente": st.column_config.NumberColumn(
                "Saldo pendiente ($)",
                min_value=0.0,
                step=100.0,
                format="%.2f"
            )
        }
    )

    st.divider()
    st.header("3. Variables para inteligencia artificial")

    st.caption(
        "Estas variables mantienen el formato de entrada de los "
        "modelos. El porcentaje de pagos a tiempo se calcula con la "
        "tabla de propiedades."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
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

    with col3:
        incidencias = st.number_input(
            "Incidencias registradas",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    enviar = st.form_submit_button(
        "Guardar expediente y evaluar inquilino",
        use_container_width=True
    )


# ------------------------------------------------------------
# 5. PROCESAMIENTO Y GUARDADO
# ------------------------------------------------------------

if enviar:

    obligatorios = {
        "El identificador del inquilino": id_inquilino,
        "El nombre del inquilino": nombre,
        "El periodo del contrato": periodo
    }

    for etiqueta, valor in obligatorios.items():
        if not valor.strip():
            st.error(f"{etiqueta} es obligatorio.")
            st.stop()

    try:
        contratos = procesar_contratos(tabla_contratos)
        documentos_contratos = convertir_contratos_documentos(contratos)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    total_propiedades = len(contratos)
    renta_total = float(contratos["renta_mensual"].sum())
    saldo_total = float(contratos["saldo_pendiente"].sum())
    total_meses = int(contratos["meses_contratados"].sum())
    total_puntuales = int(contratos["pagos_puntuales"].sum())
    total_tardios = int(contratos["pagos_tardios"].sum())

    puntualidad_general = round(total_puntuales / total_meses * 100, 2)

    inquilino = pd.DataFrame(
        [[
            float(puntualidad_general),
            float(dias_retraso),
            float(antiguedad),
            float(incidencias)
        ]],
        columns=motor.COLUMNAS
    )

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
        st.error("No fue posible ejecutar los modelos.")
        st.exception(error)
        st.stop()

    st.divider()
    st.header("4. Resultados de cobranza")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Propiedades", total_propiedades)
    col2.metric("Renta mensual total", f"${renta_total:,.2f}")
    col3.metric("Puntualidad general", f"{puntualidad_general:.2f}%")
    col4.metric("Saldo pendiente", f"${saldo_total:,.2f}")

    st.dataframe(contratos, use_container_width=True, hide_index=True)

    st.header("5. Análisis de inteligencia artificial")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Probabilidad de cumplir",
        f"{resultado_prediccion['probabilidad_cumplir'] * 100:.2f}%"
    )
    col2.metric(
        "Probabilidad de incumplir",
        f"{resultado_prediccion['probabilidad_incumplir'] * 100:.2f}%"
    )
    col3.metric("Perfil K-Means", resultado_perfil["perfil"])

    if resultado_prediccion["prediccion"] == 1:
        st.success(resultado_prediccion["resultado"])
    else:
        st.warning(resultado_prediccion["resultado"])

    st.write(
        "**Componentes PCA:**",
        f"PC1 = {resultado_pca['PC1']:.4f}",
        "|",
        f"PC2 = {resultado_pca['PC2']:.4f}"
    )

    st.subheader("Recomendaciones")

    for numero, recomendacion in enumerate(recomendaciones, start=1):
        st.write(f"{numero}. {recomendacion}")

    id_evaluacion = str(uuid.uuid4())
    id_analisis = str(uuid.uuid4())
    fecha_actual = datetime.now(timezone.utc)

    documento_contrato = {
        "id_evaluacion": id_evaluacion,
        "id_inquilino": id_inquilino.strip(),
        "sistema": SISTEMA,
        "fecha_registro": fecha_actual,
        "periodo_contrato": periodo.strip(),
        "propiedades": documentos_contratos,
        "resumen_cobranza": {
            "total_propiedades": total_propiedades,
            "renta_mensual_total": renta_total,
            "meses_contratados": total_meses,
            "pagos_puntuales": total_puntuales,
            "pagos_tardios": total_tardios,
            "porcentaje_puntualidad": puntualidad_general,
            "saldo_pendiente_total": saldo_total
        },
        "variables_comportamiento": {
            "dias_retraso_promedio": float(dias_retraso),
            "antiguedad_meses": int(antiguedad),
            "incidencias": int(incidencias)
        },
        "id_analisis": id_analisis
    }

    documento_ia = {
        "id_analisis": id_analisis,
        "id_evaluacion_contrato": id_evaluacion,
        "id_inquilino": id_inquilino.strip(),
        "sistema": SISTEMA,
        "practica": "p19",
        "fecha_registro": fecha_actual,
        "archivo_origen": "p19registro_integral_rentcontrol.py",
        "datos_reporte": {
            "id_inquilino": id_inquilino.strip(),
            "id_evaluacion_contrato": id_evaluacion,
            "pagos_a_tiempo": float(puntualidad_general),
            "dias_retraso_promedio": float(dias_retraso),
            "antiguedad_meses": int(antiguedad),
            "incidencias": int(incidencias),
            "renta_mensual_total": renta_total,
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
        with st.spinner("Guardando expediente y evaluación en MongoDB..."):
            (
                cliente,
                c_inquilinos,
                c_evaluaciones,
                c_ia
            ) = conectar_mongodb()

            guardar_inquilino(
                c_inquilinos,
                id_inquilino.strip(),
                nombre.strip(),
                telefono.strip(),
                correo.strip(),
                ocupacion.strip(),
                periodo.strip()
            )

            c_evaluaciones.insert_one(documento_contrato)

            c_ia.insert_one(documento_ia)

        st.success(
            "El expediente, la evaluación de contratos y el análisis "
            "de IA se guardaron correctamente."
        )
        st.write("**ID del inquilino:**", id_inquilino.strip())
        st.write("**ID de evaluación de contratos:**", id_evaluacion)
        st.write("**ID de análisis de IA:**", id_analisis)
        st.info(
            "Los registros anteriores del inquilino se conservaron. "
            "Se creó una nueva evaluación."
        )

    except PyMongoError as error:
        st.error("Ocurrió un problema al guardar información en MongoDB.")
        st.exception(error)
        st.warning(
            "Si hubo un error durante el guardado, verifica las "
            "colecciones antes de repetir la operación. Las escrituras "
            "en varias colecciones no forman una transacción "
            "(MongoDB local sin réplica no admite transacciones)."
        )

    except Exception as error:
        st.error("Ocurrió un error inesperado.")
        st.exception(error)

    finally:
        if cliente is not None:
            cliente.close()


# ------------------------------------------------------------
# 6. CONSULTA DEL EXPEDIENTE
# ------------------------------------------------------------

st.divider()
st.header("6. Consulta del expediente")

id_consulta = st.text_input(
    "Identificador del inquilino",
    key="consulta_expediente",
    placeholder="Ejemplo: INQ-2026-001"
)

if st.button("Consultar expediente", use_container_width=True):

    if not id_consulta.strip():
        st.warning("Escribe el identificador del inquilino.")

    else:
        cliente = None

        try:
            (
                cliente,
                c_inquilinos,
                c_evaluaciones,
                c_ia
            ) = conectar_mongodb()

            identificador = id_consulta.strip()

            expediente = c_inquilinos.find_one(
                {"id_inquilino": identificador},
                {"_id": 0}
            )

            evaluaciones = list(
                c_evaluaciones.find(
                    {"id_inquilino": identificador},
                    {"_id": 0}
                ).sort("fecha_registro", -1)
            )

            if expediente is None:
                st.warning("No se encontró un expediente con ese identificador.")

            else:
                st.success("Expediente localizado correctamente.")

                st.subheader("Datos generales")
                st.json(expediente)

                st.subheader("Historial de contratos")
                st.write(f"Evaluaciones registradas: {len(evaluaciones)}")

                for numero, evaluacion in enumerate(evaluaciones, start=1):

                    with st.expander(
                        f"Evaluación {numero} - "
                        f"{evaluacion.get('fecha_registro')}"
                    ):
                        st.write(
                            "**Periodo:**",
                            evaluacion.get("periodo_contrato")
                        )

                        st.write("**Resumen de cobranza:**")
                        st.json(evaluacion.get("resumen_cobranza", {}))

                        st.write("**Propiedades:**")
                        st.json(evaluacion.get("propiedades", []))

                        st.write("**Variables de comportamiento:**")
                        st.json(evaluacion.get("variables_comportamiento", {}))

                        analisis = c_ia.find_one(
                            {"id_analisis": evaluacion.get("id_analisis")},
                            {"_id": 0}
                        )

                        if analisis:
                            st.write("**Análisis de inteligencia artificial:**")
                            st.json(analisis.get("datos_reporte", {}))

        except PyMongoError as error:
            st.error("No fue posible consultar MongoDB.")
            st.exception(error)

        except Exception as error:
            st.error("Ocurrió un error durante la consulta.")
            st.exception(error)

        finally:
            if cliente is not None:
                cliente.close()

st.divider()

st.caption(
    "RENTCONTROL | Registro integral de inquilinos y contratos "
    "con Machine Learning y MongoDB."
)
