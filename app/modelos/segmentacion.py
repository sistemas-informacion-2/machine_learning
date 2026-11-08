"""Segmentacion de clientes (no supervisado): K-means sobre recencia, frecuencia y monto (RFM).

k se elige por silueta. Los grupos que salen no tienen nombre: se les pone uno con reglas sobre sus centros.
"""
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MINIMO_CLIENTES = 20
VALORES_DE_K = range(3, 7)


@dataclass
class ResultadoSegmentacion:
    clientes: pd.DataFrame
    metricas: dict[str, Any]


def tabla_rfm(compras: pd.DataFrame, hoy: date) -> pd.DataFrame:
    fechas = pd.to_datetime(compras["fecha"])
    resumen = compras.assign(fecha=fechas).groupby("id_cliente").agg(
        ultima=("fecha", "max"), frecuencia=("id_nota", "nunique"), monto=("importe", "sum"), descuento=("descuento", "sum"), importe_lista=("importe_lista", "sum")
    )
    resumen["recencia_dias"] = (pd.Timestamp(hoy) - resumen["ultima"]).dt.days
    resumen["proporcion_descuento"] = (resumen["descuento"] / resumen["importe_lista"].replace(0, np.nan)).fillna(0)
    return resumen.reset_index()[["id_cliente", "recencia_dias", "frecuencia", "monto", "proporcion_descuento"]]


def nombrar_segmento(centro: pd.Series, medianas: pd.Series, p75: pd.Series) -> str:
    """Nombre de negocio de un grupo segun su centro. El orden importa: gana la primera regla que se cumple."""
    if centro["recencia_dias"] > max(180, 2 * medianas["recencia_dias"]):
        return "Inactivos"
    if centro["frecuencia"] >= p75["frecuencia"] and centro["monto"] >= p75["monto"]:
        return "Frecuentes de alto gasto"
    if centro["proporcion_descuento"] >= 0.15:
        return "Cazadores de liquidacion"
    if centro["recencia_dias"] <= medianas["recencia_dias"]:
        return "Activos de buen gasto" if centro["monto"] >= medianas["monto"] else "Recientes ocasionales"
    return "En riesgo"


def segmentar(compras: pd.DataFrame, hoy: date) -> ResultadoSegmentacion | None:
    """None si hay muy pocos clientes para formar grupos."""
    if compras.empty:
        return None
    rfm = tabla_rfm(compras, hoy)
    if len(rfm) < MINIMO_CLIENTES:
        return None

    # Frecuencia y monto tienen colas largas: el logaritmo evita que unos pocos clientes definan los grupos.
    entrada = np.column_stack([rfm["recencia_dias"], np.log1p(rfm["frecuencia"]), np.log1p(rfm["monto"])])
    escalado = StandardScaler().fit_transform(entrada)

    mejor_k, mejor_silueta, mejores_grupos = 0, -1.0, np.array([])
    for k in VALORES_DE_K:
        if k >= len(rfm):
            break
        grupos = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(escalado)
        silueta = float(silhouette_score(escalado, grupos))
        if silueta > mejor_silueta:
            mejor_k, mejor_silueta, mejores_grupos = k, silueta, grupos

    rfm = rfm.assign(grupo=mejores_grupos)
    variables = ["recencia_dias", "frecuencia", "monto", "proporcion_descuento"]
    medianas, p75 = rfm[variables].median(), rfm[variables].quantile(0.75)
    centros = rfm.groupby("grupo")[variables].mean()
    nombres = {grupo: nombrar_segmento(centro, medianas, p75) for grupo, centro in centros.iterrows()}
    rfm["segmento"] = rfm["grupo"].map(nombres)

    salida = rfm[["id_cliente", "segmento", "grupo", "recencia_dias", "frecuencia", "monto"]].copy()
    salida["monto"] = salida["monto"].round(2)
    metricas = {
        "k": mejor_k,
        "silueta": round(mejor_silueta, 4),
        "clientes": len(salida),
        "tamanios": {str(k): int(v) for k, v in salida["segmento"].value_counts().items()},
    }
    return ResultadoSegmentacion(clientes=salida, metricas=metricas)
