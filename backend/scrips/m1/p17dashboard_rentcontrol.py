# ============================================================
# PROYECTO: RENTCONTROL
# PRÁCTICA 17: DASHBOARD INTELIGENTE DE RENTAS
#
# Archivo:
# p17dashboard_rentcontrol.py
#
# Objetivo:
# Construir un dashboard web para consultar y visualizar los
# análisis de inquilinos almacenados en MongoDB local.
#
# Ejecución:
#   streamlit run p17dashboard_rentcontrol.py
#
# Tecnologías: Python, Streamlit, Pandas, Plotly, PyMongo.
# ============================================================

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo.errors import PyMongoError

import rc_mongo
import rc_motor_ia

SISTEMA = "RentControl"

# Debe ejecutarse antes que cualquier otro elemento de Streamlit.
st.set_page_config(
    page_title="RentControl | Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# 1. CONEXIÓN Y CARGA DE DATOS
# ------------------------------------------------------------

@st.cache_resource
def obtener_cliente():
    """
    Cliente reutilizable de MongoDB (solo lectura si existe
    el usuario de reportes en el .env).
    """

    cliente, _ = rc_mongo.conectar(solo_lectura=True)

    return cliente


@st.cache_data(ttl=60)
def cargar_datos():
    """
    Consulta MongoDB y convierte los documentos anidados
    (datos_reporte) en una tabla de Pandas. Caché de 60 s.
    """

    cliente = obtener_cliente()

    coleccion = cliente[rc_mongo.MONGO_DB][rc_mongo.MONGO_COLLECTION]

    documentos = list(
        coleccion.find(
            {"sistema": SISTEMA},
            {"_id": 0}
        ).sort("fecha_registro", -1)
    )

    registros = []

    for documento in documentos:
        datos = documento.get("datos_reporte", {})

        registros.append({
            "id_analisis": documento.get("id_analisis"),
            "id_inquilino": documento.get("id_inquilino"),
            "fecha_registro": documento.get("fecha_registro"),
            "pagos_a_tiempo": datos.get("pagos_a_tiempo"),
            "dias_retraso_promedio": datos.get("dias_retraso_promedio"),
            "antiguedad_meses": datos.get("antiguedad_meses"),
            "incidencias": datos.get("incidencias"),
            "prediccion": datos.get("prediccion"),
            "resultado": datos.get("resultado"),
            "probabilidad_cumplir": datos.get("probabilidad_cumplir"),
            "probabilidad_incumplir": datos.get("probabilidad_incumplir"),
            "cluster": datos.get("cluster"),
            "perfil": datos.get("perfil"),
            "PC1": datos.get("PC1"),
            "PC2": datos.get("PC2")
        })

    return pd.DataFrame(registros)


def preparar_datos(dataframe):
    """
    Convierte columnas numéricas y fechas a tipos apropiados.
    """

    numericas = [
        "pagos_a_tiempo",
        "dias_retraso_promedio",
        "antiguedad_meses",
        "incidencias",
        "prediccion",
        "probabilidad_cumplir",
        "probabilidad_incumplir",
        "cluster",
        "PC1",
        "PC2"
    ]

    for columna in numericas:
        if columna in dataframe.columns:
            dataframe[columna] = pd.to_numeric(
                dataframe[columna],
                errors="coerce"
            )

    dataframe["fecha_registro"] = pd.to_datetime(
        dataframe["fecha_registro"],
        errors="coerce",
        utc=True
    )

    return dataframe


# ------------------------------------------------------------
# 2. COMPONENTES VISUALES
# ------------------------------------------------------------

def mostrar_encabezado():
    st.title("🏠 RENTCONTROL")
    st.subheader("Dashboard de análisis inteligente de rentas")
    st.markdown(
        "Consulta y visualización de los resultados generados con "
        "aprendizaje automático y almacenados en MongoDB local."
    )
    st.divider()


def mostrar_indicadores(dataframe):
    st.subheader("1. Indicadores generales")

    total = len(dataframe)
    riesgos = int((dataframe["prediccion"] == 0).sum())
    cumplimientos = int((dataframe["prediccion"] == 1).sum())

    porcentaje_riesgo = riesgos / total * 100 if total else 0
    promedio_cumplir = (
        dataframe["probabilidad_cumplir"].mean() * 100 if total else 0
    )
    promedio_pagos = dataframe["pagos_a_tiempo"].mean() if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de análisis", total)
    col2.metric(
        "Riesgos de impago",
        riesgos,
        f"{porcentaje_riesgo:.2f}% del total"
    )
    col3.metric("Cumplimientos estimados", cumplimientos)
    col4.metric("Pagos a tiempo (promedio)", f"{promedio_pagos:.2f}%")

    st.metric(
        "Probabilidad promedio estimada de cumplir",
        f"{promedio_cumplir:.2f}%"
    )


def mostrar_predicciones(dataframe):
    st.subheader("2. Distribución de predicciones")

    nombres = {
        0: "Riesgo de impago",
        1: "Cumplimiento estimado"
    }

    conteo = dataframe["prediccion"].value_counts(dropna=False)

    datos = pd.DataFrame({
        "prediccion": conteo.index,
        "cantidad": conteo.values
    })

    datos["resultado"] = (
        datos["prediccion"].map(nombres).fillna("Sin clasificar")
    )

    figura = px.pie(
        datos,
        names="resultado",
        values="cantidad",
        title="Distribución de resultados estimados",
        hole=0.45
    )

    figura.update_traces(
        textposition="inside",
        textinfo="label+percent+value"
    )

    st.plotly_chart(figura, use_container_width=True)


def mostrar_perfiles(dataframe):
    st.subheader("3. Distribución de perfiles de pago")

    datos = dataframe.dropna(subset=["perfil"])

    if datos.empty:
        st.info("No existen perfiles disponibles.")
        return

    conteo = (
        datos.groupby("perfil")
        .size()
        .reset_index(name="cantidad")
    )

    figura = px.bar(
        conteo,
        x="perfil",
        y="cantidad",
        text="cantidad",
        title="Cantidad de análisis por perfil",
        labels={
            "perfil": "Perfil de pago",
            "cantidad": "Cantidad de análisis"
        }
    )

    figura.update_traces(textposition="outside")

    st.plotly_chart(figura, use_container_width=True)


def mostrar_caracteristicas(dataframe):
    st.subheader("4. Características de los inquilinos")

    columnas = [
        "pagos_a_tiempo",
        "dias_retraso_promedio",
        "antiguedad_meses",
        "incidencias"
    ]

    nombres = [
        "Pagos a tiempo (%)",
        "Días de retraso",
        "Antigüedad (meses)",
        "Incidencias"
    ]

    datos = pd.DataFrame({
        "caracteristica": nombres,
        "promedio": dataframe[columnas].mean().values
    })

    figura = px.bar(
        datos,
        x="caracteristica",
        y="promedio",
        text_auto=".2f",
        title="Promedio de características",
        labels={
            "caracteristica": "Característica",
            "promedio": "Promedio"
        }
    )

    st.plotly_chart(figura, use_container_width=True)


def mostrar_pca(dataframe):
    st.subheader("5. Visualización de componentes PCA")

    datos = dataframe.dropna(subset=["PC1", "PC2"])

    if datos.empty:
        st.info("No existen coordenadas PCA disponibles.")
        return

    figura = px.scatter(
        datos,
        x="PC1",
        y="PC2",
        color="perfil",
        hover_data=[
            "id_inquilino",
            "pagos_a_tiempo",
            "dias_retraso_promedio",
            "antiguedad_meses",
            "incidencias",
            "resultado"
        ],
        title="Proyección de inquilinos en dos componentes principales",
        labels={
            "PC1": "Componente principal 1",
            "PC2": "Componente principal 2",
            "perfil": "Perfil de pago"
        }
    )

    st.plotly_chart(figura, use_container_width=True)

    # La varianza explicada se lee de los metadatos del entrenamiento.
    varianza = rc_motor_ia.leer_metadatos().get("varianza_explicada_pca")

    texto = "La visualización es exploratoria."

    if varianza is not None:
        texto = (
            f"La proyección representa el {varianza * 100:.2f}% de la "
            "varianza de las cuatro variables originales. " + texto
        )

    st.caption(texto)


def mostrar_historial(dataframe):
    st.subheader("6. Historial de análisis")

    columnas = [
        "id_analisis",
        "id_inquilino",
        "fecha_registro",
        "pagos_a_tiempo",
        "dias_retraso_promedio",
        "antiguedad_meses",
        "incidencias",
        "resultado",
        "probabilidad_cumplir",
        "perfil"
    ]

    historial = dataframe[
        [c for c in columnas if c in dataframe.columns]
    ].copy()

    st.dataframe(historial, use_container_width=True, hide_index=True)

    csv = historial.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Descargar historial CSV",
        data=csv,
        file_name="rentcontrol_historial.csv",
        mime="text/csv"
    )


def aplicar_filtros(dataframe):
    st.sidebar.header("Filtros de consulta")

    filtrado = dataframe.copy()

    perfiles = sorted(
        dataframe["perfil"].dropna().astype(str).unique().tolist()
    )

    seleccion_perfiles = st.sidebar.multiselect(
        "Seleccionar perfiles",
        options=perfiles,
        default=perfiles
    )

    filtrado = filtrado[
        filtrado["perfil"].astype(str).isin(seleccion_perfiles)
    ]

    resultados = sorted(
        dataframe["resultado"].dropna().astype(str).unique().tolist()
    )

    seleccion_resultados = st.sidebar.multiselect(
        "Seleccionar resultados",
        options=resultados,
        default=resultados
    )

    filtrado = filtrado[
        filtrado["resultado"].astype(str).isin(seleccion_resultados)
    ]

    return filtrado


# ------------------------------------------------------------
# 3. APLICACIÓN PRINCIPAL
# ------------------------------------------------------------

def main():
    mostrar_encabezado()

    if st.sidebar.button("Actualizar información"):
        cargar_datos.clear()
        st.rerun()

    try:
        dataframe = cargar_datos()

        if dataframe.empty:
            st.warning("Todavía no existen análisis almacenados.")
            st.info(
                "Ejecuta primero p18evaluacion_rentcontrol.py o "
                "p19registro_integral_rentcontrol.py."
            )
            return

        dataframe = preparar_datos(dataframe)

        st.sidebar.success("Conexión con MongoDB local activa")
        st.sidebar.write("Base de datos:", rc_mongo.MONGO_DB)
        st.sidebar.write("Colección:", rc_mongo.MONGO_COLLECTION)

        filtrado = aplicar_filtros(dataframe)

        st.sidebar.write("Análisis encontrados:", len(filtrado))

        if filtrado.empty:
            st.warning("No existen análisis para los filtros seleccionados.")
            return

        mostrar_indicadores(filtrado)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            mostrar_predicciones(filtrado)

        with col2:
            mostrar_perfiles(filtrado)

        st.divider()
        mostrar_caracteristicas(filtrado)

        st.divider()
        mostrar_pca(filtrado)

        st.divider()
        mostrar_historial(filtrado)

        st.divider()
        st.info(
            "Las predicciones y probabilidades son estimaciones de "
            "modelos entrenados con datos sintéticos. No sustituyen "
            "el criterio del administrador ni deben usarse como único "
            "elemento para tomar decisiones sobre un inquilino."
        )

    except PyMongoError as error:
        st.error("No fue posible consultar MongoDB. ¿Está iniciado el servicio?")
        st.code(str(error))

    except Exception as error:
        st.error("Se produjo un error inesperado.")
        st.code(str(error))


if __name__ == "__main__":
    main()
