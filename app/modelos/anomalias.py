"""Anomalias en la actividad diaria de cada sucursal (no supervisado).

Se mide la cantidad de tickets presenciales, no el importe: una sola prenda cara mueve mucho el importe de un dia
tranquilo, y las ventas en linea no dependen de lo que pase en la tienda (un bloqueo, una feria).

Un conteo varia mas cuanto mas grande es, asi que temporada alta pareceria llena de dias raros. La raiz cuadrada iguala
esa variabilidad. Sobre esa escala, la descomposicion STL separa tendencia y patron semanal (un sabado fuerte no es raro,
un martes con triple de tickets si), Isolation Forest aisla los residuos atipicos y se exige una desviacion robusta
alta para no marcar ruido.
"""
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

MINIMO_DIAS = 56
CONTAMINACION = 0.03
Z_MINIMA = 4.5
Z_ALTA = 5.5
# Ventana larga: el patron semanal de una tienda cambia despacio. Con la de defecto (7) el ajuste copia el ruido, los
# residuos normales quedan muy chicos y cualquier dia comun parece atipico.
VENTANA_ESTACIONAL = 35
AJUSTE_ANSCOMBE = 3 / 8

COLUMNAS = ["fecha", "id_sucursal", "metrica", "valor_real", "valor_esperado", "desviacion", "gravedad"]


@dataclass
class ResultadoAnomalias:
    anomalias: pd.DataFrame
    metricas: dict[str, Any]


def _serie_diaria(grupo: pd.DataFrame, hoy: date) -> pd.Series:
    fechas = pd.to_datetime(grupo["fecha"])
    por_dia = grupo["tickets"].groupby(fechas.values).sum().astype(float)
    # Hoy todavia no termino.
    indice = pd.date_range(por_dia.index.min(), pd.Timestamp(hoy) - pd.Timedelta(days=1), freq="D")
    return por_dia.reindex(indice, fill_value=0.0)


def detectar_anomalias(diario: pd.DataFrame, hoy: date) -> ResultadoAnomalias | None:
    """diario: fecha, id_sucursal, tickets. None si ninguna sucursal tiene historia suficiente."""
    if diario.empty:
        return None
    filas: list[dict] = []
    evaluadas = 0
    for id_sucursal, grupo in diario.groupby("id_sucursal"):
        serie = _serie_diaria(grupo, hoy)
        if len(serie) < MINIMO_DIAS:
            continue
        evaluadas += 1
        estabilizada = np.sqrt(serie + AJUSTE_ANSCOMBE)
        descomposicion = STL(estabilizada, period=7, seasonal=VENTANA_ESTACIONAL, robust=True).fit()
        residuo = descomposicion.resid.to_numpy()
        mad = float(np.median(np.abs(residuo - np.median(residuo)))) or 1.0
        z = (residuo - np.median(residuo)) / (1.4826 * mad)
        atipico = IsolationForest(contamination=CONTAMINACION, random_state=0).fit_predict(z.reshape(-1, 1)) == -1
        esperado = np.clip((descomposicion.trend + descomposicion.seasonal).to_numpy() ** 2 - AJUSTE_ANSCOMBE, 0, None)
        for i in np.flatnonzero(atipico & (np.abs(z) >= Z_MINIMA)):
            filas.append(
                {
                    "fecha": serie.index[i].date(),
                    "id_sucursal": int(id_sucursal),
                    "metrica": "TICKETS",
                    "valor_real": round(float(serie.iloc[i]), 2),
                    "valor_esperado": round(float(esperado[i]), 2),
                    "desviacion": round(float(z[i]), 2),
                    "gravedad": "ALTA" if abs(z[i]) >= Z_ALTA else "MEDIA",
                }
            )
    if evaluadas == 0:
        return None
    return ResultadoAnomalias(anomalias=pd.DataFrame(filas, columns=COLUMNAS), metricas={"sucursales": evaluadas, "anomalias": len(filas)})
