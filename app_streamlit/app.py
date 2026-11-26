import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title = "Uptime Guard", layout = "wide")

# 1. Cargar el modelo y su configuración (la app se lanza desde la raíz del proyecto)
with open("models/final_model.pkl", "rb") as entrada:
    modelo = pickle.load(entrada)
with open("models/config_modelo.json") as f:
    config = json.load(f)
with open("app_streamlit/valores_tipicos.json") as f:
    tipicos = json.load(f)
columnas = config["features_num"] + config["features_cat"]


def semaforo(probabilidad):
    if probabilidad >= config["umbral_sustituir"]:
        return "🔴 Sustituir"
    if probabilidad >= config["umbral_vigilar"]:
        return "🟠 Vigilar"
    return "🟢 OK"


st.title("Uptime Guard")
st.subheader("¿Qué discos de mi centro de datos van a fallar en los próximos 30 días?")
pestana_csv, pestana_disco = st.tabs(["Revisar varios discos (CSV)", "Revisar un disco"])

# 2. Varios discos: se sube un CSV con una fila por disco
with pestana_csv:
    archivo = st.file_uploader("CSV con los discos", type = "csv")
    if st.button("Usar el CSV de ejemplo"):
        archivo = "app_streamlit/ejemplo_discos.csv"
    if archivo is not None:
        discos = pd.read_csv(archivo)
        discos["probabilidad"] = modelo.predict_proba(discos[columnas])[:, 1]
        discos["decision"] = discos["probabilidad"].apply(semaforo)
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Sustituir", (discos["decision"] == "🔴 Sustituir").sum())
        col2.metric("🟠 Vigilar", (discos["decision"] == "🟠 Vigilar").sum())
        col3.metric("🟢 OK", (discos["decision"] == "🟢 OK").sum())
        st.dataframe(discos[["serial_number", "fabricante", "decision", "probabilidad"]]
                     .sort_values("probabilidad", ascending = False), hide_index = True)

# 3. Un disco: formulario con lo más importante; el resto toma valores típicos
with pestana_disco:
    fabricante = st.selectbox("Fabricante", tipicos["fabricantes"])
    datacenter = st.selectbox("Centro de datos", tipicos["datacenters"])
    capacidad = st.slider("Capacidad (TB)", 1, 26, 16)
    anios = st.slider("Años de uso", 0.0, 10.0, 3.5, step = 0.5)
    reasignados = st.number_input("Sectores reasignados (SMART 5)", min_value = 0, value = 0)
    pendientes = st.number_input("Sectores pendientes (SMART 197)", min_value = 0, value = 0)
    pendientes_7 = st.number_input("Sectores pendientes hace 7 días", min_value = 0, value = 0)

    if st.button("Predecir", type = "primary"):
        disco = dict(tipicos["medianas"])
        disco.update({"fabricante": fabricante, "datacenter": datacenter, "capacidad_tb": capacidad,
                      "smart_9_raw": anios * 365 * 24, "smart_5_raw": reasignados, "smart_197_raw": pendientes,
                      "delta7_197": max(pendientes - pendientes_7, 0), "disco_nuevo": 0})
        probabilidad = modelo.predict_proba(pd.DataFrame([disco])[columnas])[0, 1]
        st.header(semaforo(probabilidad))
        st.metric("Probabilidad de fallo en 30 días", str(round(probabilidad * 100)) + " %")
