"""Sugerencia de compra (prescriptivo).

De arriba hacia abajo: una variante vende muy pocas unidades por mes para pronosticarla sola, asi que se pronostica la
demanda de cada categoria en cada sucursal y se reparte entre sus variantes segun lo que vendio cada una ultimamente.
"""
import math
from datetime import date

import pandas as pd

from app.modelos.pronostico import pronosticar, serie_completa

HORIZONTE_MESES = 2
Z_SERVICIO = 1.28  # ~90% de nivel de servicio


def sugerir_compra(diario: pd.DataFrame, recientes: pd.DataFrame, stock: pd.DataFrame, categorias: pd.DataFrame, hoy: date) -> pd.DataFrame:
    """
    diario: fecha, id_sucursal, categoria, unidades.
    recientes: id_sucursal, id_variante, categoria, unidades (ultimos meses).
    stock: id_sucursal, id_variante, stock.
    categorias: id_variante, categoria.
    """
    columnas = ["id_variante", "id_sucursal", "demanda_esperada", "stock_actual", "stock_seguridad", "cantidad_sugerida", "motivo"]
    if diario.empty:
        return pd.DataFrame(columns=columnas)

    variantes = pd.concat(
        [recientes[["id_sucursal", "id_variante", "categoria"]], stock[["id_sucursal", "id_variante"]].merge(categorias, on="id_variante")]
    ).drop_duplicates(["id_sucursal", "id_variante"])
    variantes = variantes.merge(recientes[["id_sucursal", "id_variante", "unidades"]], on=["id_sucursal", "id_variante"], how="left")
    variantes["unidades"] = variantes["unidades"].fillna(0)
    variantes = variantes.merge(stock, on=["id_sucursal", "id_variante"], how="left")
    variantes["stock"] = variantes["stock"].fillna(0).astype(int)

    filas: list[dict] = []
    for (id_sucursal, categoria), grupo in variantes.groupby(["id_sucursal", "categoria"]):
        historia = diario[(diario["id_sucursal"] == id_sucursal) & (diario["categoria"] == categoria)]
        serie = serie_completa(historia, "unidades", "MENSUAL", hoy)
        pronostico = pronosticar(serie, "MENSUAL", HORIZONTE_MESES)
        if pronostico is None:
            continue
        demanda = sum(pronostico.valores)
        desvio = pronostico.rmse * math.sqrt(HORIZONTE_MESES)

        pesos = grupo["unidades"] + 0.5
        participacion = pesos / pesos.sum()
        for fila, parte in zip(grupo.itertuples(), participacion, strict=True):
            demanda_variante = demanda * parte
            seguridad = Z_SERVICIO * desvio * parte
            cantidad = math.ceil(demanda_variante + seguridad - fila.stock)
            if cantidad <= 0:
                continue
            filas.append(
                {
                    "id_variante": int(fila.id_variante),
                    "id_sucursal": int(id_sucursal),
                    "demanda_esperada": round(demanda_variante, 2),
                    "stock_actual": int(fila.stock),
                    "stock_seguridad": round(seguridad, 2),
                    "cantidad_sugerida": cantidad,
                    "motivo": f"Se esperan {demanda_variante:.1f} unidades en {HORIZONTE_MESES} meses y hay {int(fila.stock)} en stock",
                }
            )
    return pd.DataFrame(filas, columns=columnas)
