"""
training.py
De data/train/train.csv al modelo final: models/final_model.pkl y models/config_modelo.json.

Reproduce lo que se decidió en el notebook 03_Entrenamiento_Evaluacion:
- conjunto de variables "reducido" (15 numéricas + fabricante y datacenter)
- pipeline: imputar con la mediana -> log(1 + x) en las asimétricas -> One-Hot en las categóricas
- Random Forest con los mejores hiperparámetros del GridSearchCV (300 árboles, max_depth 12,
  min_samples_leaf 5, sin pesos de clase)
- umbrales elegidos con validación cruzada agrupada por disco:
  vigilar = el más alto con recall >= 70 %; sustituir = el más bajo con precisión >= 85 %

Uso desde la raíz del proyecto:  python src/training.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TARGET = "falla_30d"
MEJORES_PARAMETROS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5, "class_weight": None}


def crear_pipeline(train_set, variables, features_cat, todas):
    """Mismo preprocesado que en el notebook 03 (sin escalar: el Random Forest no lo necesita)."""
    asimetria = train_set[todas].skew()
    asimetricas_todas = list(asimetria[asimetria > 1].index)
    asimetricas = [c for c in variables if c in asimetricas_todas]
    resto = [c for c in variables if c not in asimetricas_todas]
    preprocesado = ColumnTransformer([
        ("num_asimetricas", Pipeline([("imputar", SimpleImputer(strategy = "median")),
                                      ("log", FunctionTransformer(np.log1p, feature_names_out = "one-to-one"))]), asimetricas),
        ("num_resto", Pipeline([("imputar", SimpleImputer(strategy = "median"))]), resto),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), features_cat)])
    modelo = RandomForestClassifier(random_state = 42, n_jobs = -1, **MEJORES_PARAMETROS)
    return Pipeline([("preprocesado", preprocesado), ("modelo", modelo)])


def elegir_umbrales(pipeline, X, y, grupos):
    """Umbrales con predicciones de validación cruzada agrupada (sin tocar test)."""
    cv = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 42)
    proba_cv = cross_val_predict(pipeline, X, y, groups = grupos, cv = cv, method = "predict_proba")[:, 1]
    filas = []
    for u in np.round(np.arange(0.1, 0.95, 0.05), 2):
        tn, fp, fn, tp = confusion_matrix(y, (proba_cv >= u).astype(int)).ravel()
        filas.append([u, tp / (tp + fn), tp / (tp + fp)])
    umbrales = pd.DataFrame(filas, columns = ["umbral", "recall", "precision"])
    vigilar = umbrales.loc[umbrales["recall"] >= 0.70, "umbral"].max()
    sustituir = umbrales.loc[umbrales["precision"] >= 0.85, "umbral"].min()
    return float(vigilar), float(sustituir)


def main():
    train_set = pd.read_csv("data/train/train.csv", low_memory = False)
    with open("data/processed/conjuntos_variables.json") as f:
        info = json.load(f)
    variables = info["conjuntos"]["reducido"]
    features_cat = info["categoricas"]
    X = train_set[variables + features_cat]
    y = train_set[TARGET]

    pipeline = crear_pipeline(train_set, variables, features_cat, info["conjuntos"]["todas"])
    vigilar, sustituir = elegir_umbrales(pipeline, X, y, train_set["serial_number"])
    print("Umbral vigilar:", vigilar, "| umbral sustituir:", sustituir)

    pipeline.fit(X, y)
    os.makedirs("models", exist_ok = True)
    with open("models/final_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    config = {"features_num": variables, "features_cat": features_cat,
              "umbral_vigilar": vigilar, "umbral_sustituir": sustituir}
    with open("models/config_modelo.json", "w") as f:
        json.dump(config, f, indent = 2)
    print("Modelo guardado en models/final_model.pkl y configuración en models/config_modelo.json")


if __name__ == "__main__":
    main()
