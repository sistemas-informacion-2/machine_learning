"""Servicio interno de machine learning del BI. Solo lo llama NestJS, con la clave interna."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from fastapi import Depends, FastAPI, status
from pydantic import BaseModel, Field

from app import datos, db
from app.entrenamiento import entrenar
from app.modelos.pronostico import periodos_siguientes, pronosticar, serie_completa
from app.seguridad import verificar_clave

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ml")

app = FastAPI(title="ML del BI", docs_url=None, redoc_url=None, openapi_url=None)

# Un entrenamiento a la vez; varias empresas hacen cola. Una empresa que ya esta en cola no se vuelve a encolar.
_ejecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entrenamiento")
_en_cola: set[int] = set()
_candado = threading.Lock()


def _entrenar_y_liberar(id_empresa: int) -> None:
    with _candado:
        _en_cola.discard(id_empresa)
    try:
        entrenar(id_empresa)
    except Exception:
        log.exception("Fallo el entrenamiento de la empresa %s", id_empresa)


class PedidoEntrenar(BaseModel):
    idEmpresa: int = Field(gt=0)


class PedidoPronostico(BaseModel):
    idEmpresa: int = Field(gt=0)
    ambito: Literal["TOTAL", "SUCURSAL", "CATEGORIA"] = "TOTAL"
    clave: str | None = None
    frecuencia: Literal["MENSUAL", "SEMANAL"] = "MENSUAL"
    horizonte: int = Field(default=6, ge=1, le=24)


@app.get("/salud")
def salud() -> dict[str, str]:
    return {"estado": "ok"}


@app.post("/entrenar", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verificar_clave)])
def pedir_entrenamiento(pedido: PedidoEntrenar) -> dict[str, bool]:
    """Responde enseguida: el entrenamiento corre aparte y su resultado queda en dw.ml_ejecucion."""
    with _candado:
        if pedido.idEmpresa in _en_cola:
            return {"encolado": False}
        _en_cola.add(pedido.idEmpresa)
    _ejecutor.submit(_entrenar_y_liberar, pedido.idEmpresa)
    return {"encolado": True}


@app.post("/pronostico", dependencies=[Depends(verificar_clave)])
def pronostico(pedido: PedidoPronostico) -> dict[str, Any]:
    """Pronostico al momento, para preguntas que no cubre lo ya calculado."""
    ventas = datos.ventas_diarias(pedido.idEmpresa)
    if pedido.ambito == "SUCURSAL":
        ventas = ventas[ventas["id_sucursal"].astype(str) == (pedido.clave or "")]
    elif pedido.ambito == "CATEGORIA":
        ventas = ventas[ventas["categoria"] == (pedido.clave or "")]

    serie = serie_completa(ventas, "importe", pedido.frecuencia, db.hoy())
    resultado = pronosticar(serie, pedido.frecuencia, pedido.horizonte) if not serie.empty else None
    if resultado is None:
        return {"disponible": False, "motivo": "No hay historia suficiente para pronosticar esta serie"}

    periodos = periodos_siguientes(serie.index[-1], pedido.frecuencia, pedido.horizonte)
    return {
        "disponible": True,
        "modelo": resultado.modelo,
        "mape": resultado.mape,
        "mapeIngenuo": resultado.mape_ingenuo,
        "puntos": [
            {"periodo": periodos[i].isoformat(), "valor": resultado.valores[i], "bajo": resultado.bajo[i], "alto": resultado.alto[i]}
            for i in range(pedido.horizonte)
        ],
    }
