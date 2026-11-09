"""Riesgo de abandono de clientes (supervisado: clasificacion).

Etiqueta: un cliente "abandona" si no compra en los VENTANA_DIAS siguientes a una fecha de corte. Se entrena con cortes
pasados y se evalua con el corte mas reciente, cuyo resultado ya se conoce: nunca se prueba con datos del futuro.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VENTANA_DIAS = 90
MAX_CORTES_DE_ENTRENAMIENTO = 6
MINIMO_ENTRENAMIENTO = 50
MINIMO_PRUEBA = 20

VARIABLES = ["recencia_dias", "frecuencia", "monto", "ticket_promedio", "antiguedad_dias", "proporcion_online", "devoluciones"]
ETIQUETAS = {
    "recencia_dias": "Días desde su última compra",
    "frecuencia": "Cantidad de compras",
    "monto": "Monto total comprado",
    "ticket_promedio": "Ticket promedio",
    "antiguedad_dias": "Días desde su primera compra",
    "proporcion_online": "Proporción de compras en línea",
    "devoluciones": "Devoluciones",
}


@dataclass
class ResultadoAbandono:
    clientes: pd.DataFrame
    metricas: dict[str, Any]


def variables_al_corte(compras: pd.DataFrame, devoluciones: pd.DataFrame, corte: date) -> pd.DataFrame:
    """Variables de cada cliente con lo que se sabia en la fecha de corte."""
    tope = pd.Timestamp(corte)
    previas = compras[compras["fecha"] <= tope]
    if previas.empty:
        return pd.DataFrame(columns=["id_cliente", *VARIABLES])
    resumen = previas.groupby("id_cliente").agg(
        ultima=("fecha", "max"), primera=("fecha", "min"), frecuencia=("id_nota", "nunique"), monto=("importe", "sum"), proporcion_online=("online", "mean")
    )
    resumen["recencia_dias"] = (tope - resumen["ultima"]).dt.days
    resumen["antiguedad_dias"] = (tope - resumen["primera"]).dt.days
    resumen["ticket_promedio"] = resumen["monto"] / resumen["frecuencia"]
    devueltas = devoluciones[devoluciones["fecha"] <= tope].groupby("id_cliente").size()
    resumen["devoluciones"] = devueltas.reindex(resumen.index, fill_value=0)
    resumen["proporcion_online"] = resumen["proporcion_online"].astype(float)
    return resumen.reset_index()[["id_cliente", *VARIABLES]]


def _etiquetado(compras: pd.DataFrame, devoluciones: pd.DataFrame, corte: date) -> pd.DataFrame:
    filas = variables_al_corte(compras, devoluciones, corte)
    inicio, fin = pd.Timestamp(corte), pd.Timestamp(corte + timedelta(days=VENTANA_DIAS))
    volvieron = compras[(compras["fecha"] > inicio) & (compras["fecha"] <= fin)]["id_cliente"].unique()
    filas["abandona"] = (~filas["id_cliente"].isin(volvieron)).astype(int)
    return filas


def _candidatos() -> dict[str, Any]:
    return {
        "REGRESION_LOGISTICA": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "RANDOM_FOREST": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0),
    }


def _importancias(modelo: Any) -> np.ndarray:
    if isinstance(modelo, RandomForestClassifier):
        return modelo.feature_importances_
    return np.abs(modelo.named_steps["logisticregression"].coef_[0])


def _nivel(probabilidad: float) -> str:
    return "ALTO" if probabilidad >= 0.7 else "MEDIO" if probabilidad >= 0.4 else "BAJO"


def predecir_abandono(compras: pd.DataFrame, devoluciones: pd.DataFrame, hoy: date) -> ResultadoAbandono | None:
    """None si no hay historia suficiente para entrenar y evaluar."""
    if compras.empty:
        return None
    compras = compras.assign(fecha=pd.to_datetime(compras["fecha"]))
    devoluciones = devoluciones.assign(fecha=pd.to_datetime(devoluciones["fecha"]))

    corte_prueba = hoy - timedelta(days=VENTANA_DIAS)
    primera = compras["fecha"].min().date()
    prueba = _etiquetado(compras, devoluciones, corte_prueba)
    tablas: list[pd.DataFrame] = []
    for i in range(1, MAX_CORTES_DE_ENTRENAMIENTO + 1):
        corte = corte_prueba - timedelta(days=VENTANA_DIAS * i)
        if corte <= primera:
            break
        tablas.append(_etiquetado(compras, devoluciones, corte))
    if not tablas:
        return None
    entrenamiento = pd.concat(tablas, ignore_index=True)

    if (
        len(entrenamiento) < MINIMO_ENTRENAMIENTO
        or len(prueba) < MINIMO_PRUEBA
        or entrenamiento["abandona"].nunique() < 2
        or prueba["abandona"].nunique() < 2
    ):
        return None

    mejor_nombre, mejor_auc, mejor_prob = "", -1.0, np.array([])
    for nombre, modelo in _candidatos().items():
        modelo.fit(entrenamiento[VARIABLES], entrenamiento["abandona"])
        probabilidades = modelo.predict_proba(prueba[VARIABLES])[:, 1]
        auc = float(roc_auc_score(prueba["abandona"], probabilidades))
        if auc > mejor_auc:
            mejor_nombre, mejor_auc, mejor_prob = nombre, auc, probabilidades

    predicciones = (mejor_prob >= 0.5).astype(int)
    metricas = {
        "modelo": mejor_nombre,
        "auc": round(mejor_auc, 4),
        "precision": round(float(precision_score(prueba["abandona"], predicciones, zero_division=0)), 4),
        "exhaustividad": round(float(recall_score(prueba["abandona"], predicciones, zero_division=0)), 4),
        "filas_entrenamiento": len(entrenamiento),
        "filas_prueba": len(prueba),
        "tasa_abandono_prueba": round(float(prueba["abandona"].mean()), 4),
    }

    # Evaluado, se reentrena con todo lo etiquetado y se puntua a los clientes de hoy.
    todo = pd.concat([entrenamiento, prueba], ignore_index=True)
    final = _candidatos()[mejor_nombre].fit(todo[VARIABLES], todo["abandona"])
    actuales = variables_al_corte(compras, devoluciones, hoy)
    probabilidades = final.predict_proba(actuales[VARIABLES])[:, 1]

    importancias = _importancias(final)
    medias, desvios = todo[VARIABLES].mean(), todo[VARIABLES].std().replace(0, 1)
    z = ((actuales[VARIABLES] - medias) / desvios).abs().to_numpy() * importancias

    filas = []
    for i, fila in enumerate(actuales.itertuples()):
        principales = np.argsort(z[i])[::-1][:2]
        filas.append(
            {
                "id_cliente": int(fila.id_cliente),
                "probabilidad": round(float(probabilidades[i]), 4),
                "nivel": _nivel(float(probabilidades[i])),
                "factores": [{"variable": ETIQUETAS[VARIABLES[j]], "valor": round(float(actuales[VARIABLES[j]].iloc[i]), 2)} for j in principales],
            }
        )
    return ResultadoAbandono(clientes=pd.DataFrame(filas), metricas=metricas)
