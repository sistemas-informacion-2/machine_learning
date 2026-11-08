from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config, db, main

CLAVE = {"X-Clave-Interna": "secreta"}


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setenv("ML_CLAVE_INTERNA", "secreta")
    config.ajustes.cache_clear()
    yield TestClient(main.app)
    config.ajustes.cache_clear()


class EjecutorFalso:
    def __init__(self):
        self.trabajos = []

    def submit(self, funcion, *argumentos):
        self.trabajos.append((funcion, argumentos))


def test_salud_no_pide_clave(cliente):
    assert cliente.get("/salud").json() == {"estado": "ok"}


@pytest.mark.parametrize("cabeceras", [{}, {"X-Clave-Interna": "otra"}])
def test_sin_la_clave_correcta_responde_401(cliente, cabeceras):
    assert cliente.post("/entrenar", json={"idEmpresa": 1}, headers=cabeceras).status_code == 401
    assert cliente.post("/pronostico", json={"idEmpresa": 1}, headers=cabeceras).status_code == 401


def test_sin_clave_configurada_rechaza_todo(monkeypatch):
    monkeypatch.setenv("ML_CLAVE_INTERNA", "")
    config.ajustes.cache_clear()
    respuesta = TestClient(main.app).post("/entrenar", json={"idEmpresa": 1}, headers={"X-Clave-Interna": ""})
    assert respuesta.status_code == 401


def test_no_expone_documentacion(cliente):
    assert cliente.get("/docs").status_code == 404
    assert cliente.get("/openapi.json").status_code == 404


def test_entrenar_responde_enseguida_y_no_encola_dos_veces(cliente, monkeypatch):
    ejecutor = EjecutorFalso()
    monkeypatch.setattr(main, "_ejecutor", ejecutor)
    main._en_cola.clear()

    primera = cliente.post("/entrenar", json={"idEmpresa": 5}, headers=CLAVE)
    segunda = cliente.post("/entrenar", json={"idEmpresa": 5}, headers=CLAVE)
    otra = cliente.post("/entrenar", json={"idEmpresa": 6}, headers=CLAVE)

    assert primera.status_code == 202 and primera.json() == {"encolado": True}
    assert segunda.json() == {"encolado": False}
    assert otra.json() == {"encolado": True}
    assert [argumentos for _, argumentos in ejecutor.trabajos] == [(5,), (6,)]
    main._en_cola.clear()


def test_al_empezar_libera_la_cola_para_aceptar_un_pedido_nuevo(monkeypatch):
    monkeypatch.setattr(main, "entrenar", lambda _: None)
    main._en_cola.add(5)
    main._entrenar_y_liberar(5)
    assert 5 not in main._en_cola


def test_un_fallo_de_entrenamiento_no_rompe_el_hilo(monkeypatch):
    def falla(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "entrenar", falla)
    main._entrenar_y_liberar(5)


def test_id_de_empresa_invalido(cliente):
    assert cliente.post("/entrenar", json={"idEmpresa": 0}, headers=CLAVE).status_code == 422


def test_pronostico_al_momento_pide_los_datos_de_la_empresa_y_filtra(cliente, monkeypatch):
    pedidas = []

    def ventas(id_empresa):
        pedidas.append(id_empresa)
        fechas = pd.date_range("2025-01-01", "2026-10-31", freq="D")
        return pd.DataFrame({"fecha": fechas.date, "id_sucursal": 3, "categoria": "Jeans", "importe": 100.0, "unidades": 1})

    monkeypatch.setattr(main.datos, "ventas_diarias", ventas)
    monkeypatch.setattr(main.db, "hoy", lambda: date(2026, 11, 8))

    respuesta = cliente.post("/pronostico", json={"idEmpresa": 9, "ambito": "SUCURSAL", "clave": "3", "horizonte": 2}, headers=CLAVE).json()
    assert pedidas == [9]
    assert respuesta["disponible"] is True
    assert [p["periodo"] for p in respuesta["puntos"]] == ["2026-11-01", "2026-12-01"]

    otra = cliente.post("/pronostico", json={"idEmpresa": 9, "ambito": "SUCURSAL", "clave": "99"}, headers=CLAVE).json()
    assert otra == {"disponible": False, "motivo": "No hay historia suficiente para pronosticar esta serie"}


def test_ninguna_lectura_sin_filtro_de_empresa():
    with pytest.raises(ValueError, match="id_empresa"):
        db.leer("SELECT * FROM dw.fact_ventas", 1)


def test_solo_se_escriben_tablas_de_resultados():
    with pytest.raises(ValueError, match="desconocida"):
        db.reemplazar_resultados("fact_ventas", 1, [])
