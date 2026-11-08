"""Corre todos los modelos de una empresa y guarda resultados y metricas.

Cada modelo es independiente: si uno falla, conserva sus resultados anteriores y los demas siguen.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from app import datos, db
from app.modelos.abandono import predecir_abandono
from app.modelos.anomalias import detectar_anomalias
from app.modelos.canasta import reglas_de_asociacion
from app.modelos.pronostico import pronosticar, periodos_siguientes, serie_completa
from app.modelos.reposicion import sugerir_compra
from app.modelos.segmentacion import segmentar

log = logging.getLogger("ml.entrenamiento")

HORIZONTES = {"MENSUAL": 6, "SEMANAL": 12}
DIAS_PARA_PARTICIPACION = 180


@dataclass
class Contexto:
    id_empresa: int
    hoy: date
    ventas: pd.DataFrame
    compras: pd.DataFrame
    devoluciones: pd.DataFrame


def _series(ventas: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    series = [("TOTAL", "TOTAL", ventas)]
    series += [("SUCURSAL", str(clave), grupo) for clave, grupo in ventas.groupby("id_sucursal")]
    series += [("CATEGORIA", str(clave), grupo) for clave, grupo in ventas.groupby("categoria")]
    return series


def _pronosticos(ctx: Contexto) -> dict[str, Any]:
    filas: list[dict] = []
    errores: list[float] = []
    errores_ingenuo: list[float] = []
    modelos: dict[str, int] = {}
    omitidas = 0
    for ambito, clave, grupo in _series(ctx.ventas):
        for frecuencia, horizonte in HORIZONTES.items():
            serie = serie_completa(grupo, "importe", frecuencia, ctx.hoy)
            resultado = pronosticar(serie, frecuencia, horizonte) if not serie.empty else None
            if resultado is None:
                omitidas += 1
                continue
            modelos[resultado.modelo] = modelos.get(resultado.modelo, 0) + 1
            if ambito == "TOTAL" and frecuencia == "MENSUAL" and resultado.mape is not None:
                errores.append(resultado.mape)
                if resultado.mape_ingenuo is not None:
                    errores_ingenuo.append(resultado.mape_ingenuo)
            for i, periodo in enumerate(periodos_siguientes(serie.index[-1], frecuencia, horizonte)):
                filas.append(
                    {
                        "ambito": ambito,
                        "clave": clave,
                        "frecuencia": frecuencia,
                        "periodo": periodo,
                        "valor": resultado.valores[i],
                        "bajo": resultado.bajo[i],
                        "alto": resultado.alto[i],
                        "modelo": resultado.modelo,
                    }
                )
    db.reemplazar_resultados("ml_pronostico", ctx.id_empresa, filas)
    return {
        "filas": len(filas),
        "series_omitidas_por_pocos_datos": omitidas,
        "modelos_elegidos": modelos,
        "mape_total_mensual": errores[0] if errores else None,
        "mape_ingenuo_total_mensual": errores_ingenuo[0] if errores_ingenuo else None,
    }


def _reposicion(ctx: Contexto) -> dict[str, Any]:
    desde = (ctx.hoy - timedelta(days=DIAS_PARA_PARTICIPACION)).isoformat()
    sugerencias = sugerir_compra(
        ctx.ventas, datos.ventas_recientes_por_variante(ctx.id_empresa, desde), datos.stock_actual(ctx.id_empresa), datos.categorias_de_variantes(ctx.id_empresa), ctx.hoy
    )
    db.reemplazar_resultados("ml_sugerencia_compra", ctx.id_empresa, sugerencias.to_dict("records"))
    return {"filas": len(sugerencias), "unidades": int(sugerencias["cantidad_sugerida"].sum()) if len(sugerencias) else 0}


def _abandono(ctx: Contexto) -> dict[str, Any]:
    resultado = predecir_abandono(ctx.compras, ctx.devoluciones, ctx.hoy)
    if resultado is None:
        db.reemplazar_resultados("ml_riesgo_abandono", ctx.id_empresa, [])
        return {"omitido": "pocos datos"}
    db.reemplazar_resultados("ml_riesgo_abandono", ctx.id_empresa, resultado.clientes.to_dict("records"))
    return {**resultado.metricas, "filas": len(resultado.clientes)}


def _segmentos(ctx: Contexto) -> dict[str, Any]:
    resultado = segmentar(ctx.compras, ctx.hoy)
    if resultado is None:
        db.reemplazar_resultados("ml_segmento_cliente", ctx.id_empresa, [])
        return {"omitido": "pocos datos"}
    db.reemplazar_resultados("ml_segmento_cliente", ctx.id_empresa, resultado.clientes.to_dict("records"))
    return resultado.metricas


def _canasta(ctx: Contexto) -> dict[str, Any]:
    resultado = reglas_de_asociacion(datos.lineas_de_notas(ctx.id_empresa))
    if resultado is None:
        db.reemplazar_resultados("ml_asociacion", ctx.id_empresa, [])
        return {"omitido": "pocos datos"}
    db.reemplazar_resultados("ml_asociacion", ctx.id_empresa, resultado.reglas.to_dict("records"))
    return resultado.metricas


def _anomalias(ctx: Contexto) -> dict[str, Any]:
    resultado = detectar_anomalias(datos.tickets_presenciales(ctx.id_empresa), ctx.hoy)
    if resultado is None:
        db.reemplazar_resultados("ml_anomalia", ctx.id_empresa, [])
        return {"omitido": "pocos datos"}
    db.reemplazar_resultados("ml_anomalia", ctx.id_empresa, resultado.anomalias.to_dict("records"))
    return resultado.metricas


MODELOS: dict[str, Callable[[Contexto], dict[str, Any]]] = {
    "pronostico": _pronosticos,
    "reposicion": _reposicion,
    "abandono": _abandono,
    "segmentacion": _segmentos,
    "canasta": _canasta,
    "anomalias": _anomalias,
}


def entrenar(id_empresa: int) -> dict[str, Any]:
    id_ejecucion = db.iniciar_ejecucion(id_empresa)
    metricas: dict[str, Any] = {}
    try:
        ctx = Contexto(
            id_empresa=id_empresa,
            hoy=db.hoy(),
            ventas=datos.ventas_diarias(id_empresa),
            compras=datos.compras_de_clientes(id_empresa),
            devoluciones=datos.devoluciones_de_clientes(id_empresa),
        )
        fallidos = 0
        for nombre, modelo in MODELOS.items():
            try:
                metricas[nombre] = modelo(ctx)
            except Exception as error:  # noqa: BLE001 - un modelo que falla no debe tirar a los demas
                log.exception("Fallo el modelo %s de la empresa %s", nombre, id_empresa)
                metricas[nombre] = {"error": type(error).__name__}
                fallidos += 1
        estado = "OK" if fallidos == 0 else "ERROR" if fallidos == len(MODELOS) else "PARCIAL"
        db.terminar_ejecucion(id_ejecucion, id_empresa, estado, metricas, None)
        log.info("Modelos de la empresa %s: %s", id_empresa, estado)
    except Exception as error:
        db.terminar_ejecucion(id_ejecucion, id_empresa, "ERROR", metricas, type(error).__name__)
        raise
    return metricas
