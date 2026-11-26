"""
evaluation.py
De models/final_model.pkl + data/test a las métricas del modelo.

- classification_report con el umbral de vigilar y PR-AUC
- recall y precisión en los dos umbrales, con intervalos de confianza al 95 % por bootstrap
- lo mismo en la prueba final de abril (data/test/test_abril.csv), un mes que el modelo no ha visto

Uso desde la raíz del proyecto:  python src/evaluation.py
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, precision_score, recall_score

TARGET = "falla_30d"


def intervalo_bootstrap(y_real, y_pred, funcion, n = 1000):
    rng = np.random.default_rng(42)
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    valores = []
    for i in range(n):
        idx = rng.integers(0, len(y_real), len(y_real))
        valores.append(funcion(y_real[idx], y_pred[idx]))
    return np.percentile(valores, 2.5), np.percentile(valores, 97.5)


def evaluar(ruta, nombre, modelo, config):
    datos = pd.read_csv(ruta, low_memory = False)
    X = datos[config["features_num"] + config["features_cat"]]
    y = datos[TARGET]
    proba = modelo.predict_proba(X)[:, 1]

    print("=====", nombre, "|", len(datos), "discos |", int(y.sum()), "fallos")
    print(classification_report(y, (proba >= config["umbral_vigilar"]).astype(int), digits = 3))
    print("PR-AUC:", round(average_precision_score(y, proba), 3))
    for accion in ["vigilar", "sustituir"]:
        pred = (proba >= config["umbral_" + accion]).astype(int)
        ic_r = intervalo_bootstrap(y, pred, recall_score)
        ic_p = intervalo_bootstrap(y, pred, precision_score)
        print(accion, "| recall", round(recall_score(y, pred), 3), "IC95", np.round(ic_r, 3),
              "| precisión", round(precision_score(y, pred), 3), "IC95", np.round(ic_p, 3))
    print()


def main():
    with open("models/final_model.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("models/config_modelo.json") as f:
        config = json.load(f)
    evaluar("data/test/test.csv", "Test (discos que el modelo no ha visto)", modelo, config)
    evaluar("data/test/test_abril.csv", "Prueba final (abril de 2026)", modelo, config)


if __name__ == "__main__":
    main()
