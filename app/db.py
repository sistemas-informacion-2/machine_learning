"""Acceso a Postgres. Toda lectura y toda escritura va limitada a una empresa."""
import json
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, text

from app.config import ajustes

# Una tabla de resultados solo se puede reemplazar si esta en esta lista: el nombre va dentro del SQL.
TABLAS_RESULTADO = {
    "ml_pronostico",
    "ml_sugerencia_compra",
    "ml_riesgo_abandono",
    "ml_segmento_cliente",
    "ml_asociacion",
    "ml_anomalia",
}


@lru_cache
def motor() -> Engine:
    url = ajustes().ml_database_url
    if not url:
        raise RuntimeError("Falta ML_DATABASE_URL")
    return create_engine(url, pool_pre_ping=True, connect_args={"options": "-c TimeZone=America/La_Paz"})


def leer(sql: str, id_empresa: int, **parametros: Any) -> pd.DataFrame:
    """SELECT a un DataFrame. La consulta tiene que filtrar por `:id_empresa`, o no se ejecuta."""
    if ":id_empresa" not in sql:
        raise ValueError("Toda consulta del servicio tiene que filtrar por :id_empresa")
    with motor().connect() as conexion:
        return pd.read_sql_query(text(sql), conexion, params={"id_empresa": id_empresa, **parametros})


def hoy() -> date:
    """Fecha actual en La Paz, segun la base."""
    with motor().connect() as conexion:
        return conexion.execute(text("SELECT CURRENT_DATE")).scalar_one()


def nativo(valor: Any) -> Any:
    """Tipos de numpy/pandas a tipos de Python, y NaN a None, para el driver y para JSON."""
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return None if np.isnan(valor) else float(valor)
    if isinstance(valor, float) and np.isnan(valor):
        return None
    if isinstance(valor, pd.Timestamp):
        return valor.date()
    if isinstance(valor, (dict, list)):
        return json.dumps(valor, default=nativo)
    return valor


def reemplazar_resultados(tabla: str, id_empresa: int, filas: Sequence[Mapping[str, Any]]) -> int:
    """Cambia los resultados de un modelo para una empresa en una transaccion: o quedan todos los nuevos, o los de antes."""
    if tabla not in TABLAS_RESULTADO:
        raise ValueError(f"Tabla de resultados desconocida: {tabla}")
    with motor().begin() as conexion:
        conexion.execute(text(f"DELETE FROM dw.{tabla} WHERE id_empresa = :id_empresa"), {"id_empresa": id_empresa})
        if filas:
            columnas = list(filas[0].keys())
            sql = f"INSERT INTO dw.{tabla} (id_empresa, {', '.join(columnas)}) VALUES (:id_empresa, {', '.join(':' + c for c in columnas)})"
            conexion.execute(text(sql), [{"id_empresa": id_empresa, **{c: nativo(f[c]) for c in columnas}} for f in filas])
    return len(filas)


def iniciar_ejecucion(id_empresa: int) -> int:
    with motor().begin() as conexion:
        return conexion.execute(
            text("INSERT INTO dw.ml_ejecucion (id_empresa, estado) VALUES (:id_empresa, 'EN_CURSO') RETURNING id"),
            {"id_empresa": id_empresa},
        ).scalar_one()


def terminar_ejecucion(id_ejecucion: int, id_empresa: int, estado: str, metricas: dict[str, Any], error: str | None) -> None:
    with motor().begin() as conexion:
        conexion.execute(
            text(
                "UPDATE dw.ml_ejecucion SET estado = :estado, fin = now(), metricas = CAST(:metricas AS jsonb), error = :error "
                "WHERE id = :id AND id_empresa = :id_empresa"
            ),
            {"id": id_ejecucion, "id_empresa": id_empresa, "estado": estado, "metricas": json.dumps(metricas, default=nativo), "error": error},
        )
