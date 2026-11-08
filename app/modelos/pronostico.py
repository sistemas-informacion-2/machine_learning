"""Pronostico de series (supervisado: series de tiempo).

Prueba varios candidatos sobre los ultimos periodos guardados para prueba y se queda con el de menor error. Siempre
compite contra el modelo ingenuo (repetir el mismo periodo del anio anterior): asi se ve si el modelo aporta algo.
"""
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Frecuencia = str  # 'MENSUAL' | 'SEMANAL'

TEMPORADA = {"MENSUAL": 12, "SEMANAL": 52}
PERIODOS_DE_PRUEBA = {"MENSUAL": 3, "SEMANAL": 8}
MINIMO_DE_PERIODOS = {"MENSUAL": 6, "SEMANAL": 12}
Z_80 = 1.2816

Candidato = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class Pronostico:
    valores: list[float]
    bajo: list[float]
    alto: list[float]
    modelo: str
    mape: float | None
    mape_ingenuo: float | None
    rmse: float


def inicio_de_periodo(fecha: date, frecuencia: Frecuencia) -> date:
    if frecuencia == "MENSUAL":
        return fecha.replace(day=1)
    return fecha - timedelta(days=fecha.weekday())


def serie_completa(diario: pd.DataFrame, columna: str, frecuencia: Frecuencia, hoy: date) -> pd.Series:
    """Serie regular por periodo, con ceros donde no hubo ventas.

    Solo periodos completos: se descarta el que esta en curso y el primero si la historia empezo a mitad de periodo.
    """
    if diario.empty:
        return pd.Series(dtype=float)
    fechas = pd.to_datetime(diario["fecha"])
    regla = "MS" if frecuencia == "MENSUAL" else "7D"
    periodo = fechas.dt.to_period("M").dt.start_time if frecuencia == "MENSUAL" else fechas - pd.to_timedelta(fechas.dt.day_of_week, unit="D")
    agrupada = diario[columna].groupby(periodo.values).sum().astype(float)

    primera_fecha = fechas.min()
    primer_periodo = agrupada.index.min()
    tolerancia = 2 if frecuencia == "MENSUAL" else 1
    if (primera_fecha - primer_periodo).days > tolerancia:
        primer_periodo = (primer_periodo + pd.offsets.MonthBegin(1)) if frecuencia == "MENSUAL" else primer_periodo + pd.Timedelta(days=7)

    ultimo = pd.Timestamp(inicio_de_periodo(hoy, frecuencia))
    indice = pd.date_range(primer_periodo, ultimo, freq=regla, inclusive="left")
    return agrupada.reindex(indice, fill_value=0.0)


def periodos_siguientes(ultimo: pd.Timestamp, frecuencia: Frecuencia, horizonte: int) -> list[date]:
    if frecuencia == "MENSUAL":
        fechas = pd.date_range(ultimo + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS")
    else:
        fechas = pd.date_range(ultimo + pd.Timedelta(days=7), periods=horizonte, freq="7D")
    return [f.date() for f in fechas]


def _ingenuo(temporada: int) -> Candidato:
    def predecir(entrenamiento: np.ndarray, pasos: int) -> np.ndarray:
        if len(entrenamiento) >= temporada:
            return np.array([entrenamiento[-temporada + (i % temporada)] for i in range(pasos)], dtype=float)
        return np.repeat(entrenamiento[-1], pasos).astype(float)

    return predecir


def _estacional_con_tendencia(temporada: int, ventana: int) -> Candidato:
    """El mismo periodo del anio anterior, corregido por cuanto crecio el negocio en los ultimos periodos."""

    def predecir(entrenamiento: np.ndarray, pasos: int) -> np.ndarray:
        reciente = entrenamiento[-ventana:].sum()
        anterior = entrenamiento[-temporada - ventana : -temporada].sum()
        crecimiento = float(np.clip(reciente / anterior, 0.5, 2.0)) if anterior > 0 else 1.0
        return _ingenuo(temporada)(entrenamiento, pasos) * crecimiento

    return predecir


def _holt(entrenamiento: np.ndarray, pasos: int) -> np.ndarray:
    ajuste = ExponentialSmoothing(entrenamiento, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return np.asarray(ajuste.forecast(pasos), dtype=float)


def _holt_winters(tipo: str, temporada: int) -> Candidato:
    def predecir(entrenamiento: np.ndarray, pasos: int) -> np.ndarray:
        ajuste = ExponentialSmoothing(
            entrenamiento, trend="add", damped_trend=True, seasonal=tipo, seasonal_periods=temporada, initialization_method="heuristic"
        ).fit()
        return np.asarray(ajuste.forecast(pasos), dtype=float)

    return predecir


def _candidatos(frecuencia: Frecuencia, n: int, y: np.ndarray) -> dict[str, Candidato]:
    temporada = TEMPORADA[frecuencia]
    ventana = PERIODOS_DE_PRUEBA[frecuencia]
    candidatos: dict[str, Candidato] = {"INGENUO": _ingenuo(temporada)}
    if n >= 10:
        candidatos["HOLT"] = _holt
    if n >= temporada + ventana:
        candidatos["ESTACIONAL_CON_TENDENCIA"] = _estacional_con_tendencia(temporada, ventana)
    # Con frecuencia semanal serian 52 factores estacionales: demasiados para 2 anios de datos.
    if frecuencia == "MENSUAL" and n >= 2 * temporada:
        candidatos["HOLT_WINTERS_ADITIVO"] = _holt_winters("add", temporada)
        if (y > 0).all():
            candidatos["HOLT_WINTERS_MULTIPLICATIVO"] = _holt_winters("mul", temporada)
    return candidatos


def _mape(real: np.ndarray, predicho: np.ndarray) -> float | None:
    positivos = real > 0
    if not positivos.any():
        return None
    return float(np.mean(np.abs(real[positivos] - predicho[positivos]) / real[positivos]))


def pronosticar(valores: pd.Series | np.ndarray, frecuencia: Frecuencia, horizonte: int) -> Pronostico | None:
    """None si no hay historia suficiente."""
    y = np.asarray(valores, dtype=float)
    if len(y) < MINIMO_DE_PERIODOS[frecuencia]:
        return None

    prueba = PERIODOS_DE_PRUEBA[frecuencia]
    entrenamiento, real = y[:-prueba], y[-prueba:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", RuntimeWarning)

        evaluados: dict[str, tuple[float, np.ndarray]] = {}
        for nombre, candidato in _candidatos(frecuencia, len(entrenamiento), entrenamiento).items():
            try:
                predicho = np.clip(candidato(entrenamiento, prueba), 0, None)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(predicho).all():
                continue
            error = _mape(real, predicho)
            evaluados[nombre] = (float("inf") if error is None else error, predicho)

        modelo = min(evaluados, key=lambda nombre: evaluados[nombre][0])
        rmse = float(np.sqrt(np.mean((real - evaluados[modelo][1]) ** 2)))

        # El ganador se reajusta con toda la serie. Si no se puede (por ejemplo, un cero que impide el
        # multiplicativo), se usa el ingenuo, que siempre funciona, y asi queda informado.
        finales = _candidatos(frecuencia, len(y), y)
        try:
            final = finales[modelo](y, horizonte)
            if not np.isfinite(final).all():
                raise ValueError("pronostico no finito")
        except (KeyError, ValueError, np.linalg.LinAlgError):
            modelo = "INGENUO"
            final = finales[modelo](y, horizonte)

    valores_finales = np.clip(np.asarray(final, dtype=float), 0, None)
    # El RMSE viene de la prueba; el margen crece de a poco con la distancia al ultimo dato.
    margen = Z_80 * rmse * np.sqrt(1 + np.arange(horizonte) / prueba)
    mape = evaluados[modelo][0]
    mape_ingenuo = evaluados["INGENUO"][0]
    return Pronostico(
        valores=[round(float(v), 2) for v in valores_finales],
        bajo=[round(float(v), 2) for v in np.clip(valores_finales - margen, 0, None)],
        alto=[round(float(v), 2) for v in valores_finales + margen],
        modelo=modelo,
        mape=None if np.isinf(mape) else round(mape, 4),
        mape_ingenuo=None if np.isinf(mape_ingenuo) else round(mape_ingenuo, 4),
        rmse=round(rmse, 2),
    )
