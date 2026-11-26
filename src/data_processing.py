"""
data_processing.py
De data/processed/01_*_unido.csv a data/train, data/test y data/processed/02_dataset_limpio.csv.

Reproduce el notebook 02_LimpiezaEDA:
1. Split por disco (80/20, estratificado por "el disco falla alguna vez", semilla 42).
2. Missings calculados solo con train: fuera los atributos SMART con más del 50 %.
3. limpiar(): la misma limpieza y feature engineering para train, test y abril.

La extracción desde los zips de Backblaze está en el notebook 01_Fuentes (necesita los zips, más de 3 GB).

Uso desde la raíz del proyecto:  python src/data_processing.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "falla_30d"
FABRICANTES = ["TOSHIBA", "Seagate", "WDC", "HGST"]

SMART_USAR = [1, 3, 4, 5, 7, 9, 12, 192, 193, 194, 197, 198, 199]
SMART_DELTA = [1, 4, 5, 7, 9, 12, 192, 193, 197, 198, 199]
SMART_DELTA_CORTO = [1, 4, 5, 7, 12, 192, 193, 197, 198, 199]

FEATURES_NUM = ["smart_" + str(n) + "_raw" for n in SMART_USAR]
FEATURES_NUM += ["delta_" + str(n) for n in SMART_DELTA]
FEATURES_NUM += ["delta" + str(d) + "_" + str(n) for d in [7, 14] for n in SMART_DELTA_CORTO]
FEATURES_NUM += ["capacidad_tb", "disco_nuevo"]
FEATURES_CAT = ["fabricante", "datacenter"]
GUARDAR = ["serial_number", "fecha_foto", "fecha_fallo", "model"] + FEATURES_NUM + FEATURES_CAT + [TARGET]

POCO_UTILES = ["smart_3_raw", "smart_7_raw", "smart_194_raw", "smart_199_raw", "delta_7", "delta_192",
               "delta_193", "delta_199", "delta7_7", "delta7_12", "delta7_192", "delta7_193", "delta7_199",
               "delta14_7", "delta14_12", "delta14_192", "delta14_193", "delta14_199"]
REDUCIDO = ["smart_1_raw", "smart_5_raw", "smart_9_raw", "smart_12_raw", "smart_193_raw", "smart_197_raw",
            "smart_198_raw", "delta_1", "delta_5", "delta_12", "delta_197", "delta7_5", "delta7_197",
            "capacidad_tb", "disco_nuevo"]


def split_por_disco(df):
    """Cada disco va entero a train o a test (un mismo disco puede salir en varias fechas)."""
    falla_disco = df.groupby("serial_number")[TARGET].max()
    discos_train, discos_test = train_test_split(list(falla_disco.index), test_size = 0.2,
                                                 stratify = falla_disco.values, random_state = 42)
    return df[df["serial_number"].isin(discos_train)], df[df["serial_number"].isin(discos_test)]


def columnas_a_eliminar(train_set):
    """Atributos SMART con más del 50 % de missings, calculado solo con train."""
    smart_raw = [c for c in train_set.columns if c.startswith("smart_") and c.endswith("_raw")]
    missings = train_set[smart_raw].isna().mean() * 100
    return [int(c.split("_")[1]) for c in missings[missings > 50].index]


def poner_fabricante(modelo):
    """Marca del disco a partir del modelo (los Seagate empiezan por ST sin la marca delante)."""
    modelo = str(modelo)
    if " " in modelo:
        marca = modelo.split(" ")[0]
    else:
        marca = modelo[:2]
    if marca == "ST":
        marca = "Seagate"
    if marca not in FABRICANTES:
        marca = "Otros"
    return marca


def limpiar(df, smart_eliminar):
    """Limpieza y feature engineering. No aprende nada de los datos: sirve igual para train, test y abril."""
    df = df.copy()
    smart_raw = [c for c in df.columns if c.startswith("smart_") and c.endswith("_raw")]
    df = df[~df[smart_raw].isna().all(axis = 1)]          # discos que no reportaron nada ese día

    quitar = [c for c in df.columns if c.endswith("_normalized")]
    for n in smart_eliminar + [10]:                        # el 10 casi siempre vale 0
        for sufijo in ["_raw", "_raw_hace7", "_raw_hace14", "_raw_hace30"]:
            if "smart_" + str(n) + sufijo in df.columns:
                quitar.append("smart_" + str(n) + sufijo)
    df = df.drop(columns = quitar)

    df["fabricante"] = df["model"].apply(poner_fabricante)
    df["capacidad_tb"] = df["capacity_bytes"] / 1e12
    hace30 = ["smart_" + str(n) + "_raw_hace30" for n in SMART_USAR]
    df["disco_nuevo"] = df[hace30].isna().all(axis = 1).astype(int)
    for n in SMART_DELTA:
        hoy = "smart_" + str(n) + "_raw"
        df["delta_" + str(n)] = (df[hoy] - df[hoy + "_hace30"].fillna(df[hoy])).clip(lower = 0)
    for d in [7, 14]:
        for n in SMART_DELTA_CORTO:
            hoy = "smart_" + str(n) + "_raw"
            df["delta" + str(d) + "_" + str(n)] = (df[hoy] - df[hoy + "_hace" + str(d)].fillna(df[hoy])).clip(lower = 0)
    return df


def main():
    df = pd.read_csv("data/processed/01_dataset_unido.csv", low_memory = False)
    train_set, test_set = split_por_disco(df)
    print("Train:", train_set.shape, "| Test:", test_set.shape)

    smart_eliminar = columnas_a_eliminar(train_set)
    print("Se eliminan:", sorted(smart_eliminar))

    train_set = limpiar(train_set, smart_eliminar)[GUARDAR]
    test_set = limpiar(test_set, smart_eliminar)[GUARDAR]
    abril = limpiar(pd.read_csv("data/processed/01_abril_unido.csv", low_memory = False), smart_eliminar)[GUARDAR]
    print("Train limpio:", train_set.shape, "| Test limpio:", test_set.shape, "| Abril limpio:", abril.shape)

    conjuntos = {"todas": FEATURES_NUM,
                 "sin_poco_utiles": [c for c in FEATURES_NUM if c not in POCO_UTILES],
                 "reducido": REDUCIDO}
    with open("data/processed/conjuntos_variables.json", "w") as f:
        json.dump({"conjuntos": conjuntos, "categoricas": FEATURES_CAT}, f, indent = 2)
    pd.concat([train_set, test_set]).to_csv("data/processed/02_dataset_limpio.csv", index = False)
    train_set.to_csv("data/train/train.csv", index = False)
    test_set.to_csv("data/test/test.csv", index = False)
    abril.to_csv("data/test/test_abril.csv", index = False)
    print("Guardados train.csv, test.csv, test_abril.csv y 02_dataset_limpio.csv")


if __name__ == "__main__":
    main()
