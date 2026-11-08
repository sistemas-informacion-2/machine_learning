from datetime import date

import numpy as np
import pandas as pd

from app.modelos.pronostico import periodos_siguientes, pronosticar, serie_completa


def estacional(meses: int, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    t = np.arange(meses)
    return (1000 + 15 * t) * (1 + 0.5 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, 15, meses)


def test_serie_estacional_le_gana_al_modelo_ingenuo():
    resultado = pronosticar(estacional(36), "MENSUAL", 6)
    assert resultado is not None
    assert resultado.modelo != "INGENUO"
    assert resultado.mape < resultado.mape_ingenuo
    assert resultado.mape < 0.1


def test_con_menos_de_dos_anios_en_entrenamiento_no_intenta_holt_winters():
    resultado = pronosticar(estacional(24), "MENSUAL", 3)
    assert resultado is not None
    assert not resultado.modelo.startswith("HOLT_WINTERS")


def test_intervalo_contiene_al_pronostico_y_se_ensancha():
    resultado = pronosticar(estacional(36), "MENSUAL", 6)
    for bajo, valor, alto in zip(resultado.bajo, resultado.valores, resultado.alto, strict=True):
        assert 0 <= bajo <= valor <= alto
    assert resultado.alto[5] - resultado.valores[5] > resultado.alto[0] - resultado.valores[0]


def test_sin_historia_suficiente_no_pronostica():
    assert pronosticar(np.array([1, 2, 3, 4, 5.0]), "MENSUAL", 3) is None
    assert pronosticar(np.ones(11), "SEMANAL", 3) is None


def test_serie_con_ceros_no_falla_ni_da_negativos():
    y = np.array([0, 0, 5, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0.0])
    resultado = pronosticar(y, "SEMANAL", 4)
    assert resultado is not None
    assert all(v >= 0 and np.isfinite(v) for v in resultado.valores)


def test_multiplicativo_elegido_con_un_cero_al_final_usa_el_respaldo():
    y = estacional(40)
    y[-1] = 0
    resultado = pronosticar(y, "MENSUAL", 3)
    assert resultado is not None and all(np.isfinite(resultado.valores))


def ventas(*filas):
    return pd.DataFrame(filas, columns=["fecha", "importe"])


def test_serie_completa_descarta_el_mes_en_curso_y_rellena_con_ceros():
    diario = ventas((date(2026, 1, 1), 10.0), (date(2026, 1, 20), 5.0), (date(2026, 3, 2), 7.0), (date(2026, 4, 3), 99.0))
    serie = serie_completa(diario, "importe", "MENSUAL", date(2026, 4, 15))
    assert list(serie.index.strftime("%Y-%m")) == ["2026-01", "2026-02", "2026-03"]
    assert list(serie) == [15.0, 0.0, 7.0]


def test_serie_completa_descarta_el_primer_mes_si_empezo_a_mitad():
    diario = ventas((date(2026, 1, 20), 5.0), (date(2026, 2, 2), 7.0))
    serie = serie_completa(diario, "importe", "MENSUAL", date(2026, 3, 15))
    assert list(serie.index.strftime("%Y-%m")) == ["2026-02"]


def test_serie_semanal_agrupa_de_lunes_a_domingo():
    # 2026-11-02 es lunes.
    diario = ventas((date(2026, 11, 2), 1.0), (date(2026, 11, 8), 2.0), (date(2026, 11, 9), 4.0), (date(2026, 11, 16), 8.0))
    serie = serie_completa(diario, "importe", "SEMANAL", date(2026, 11, 18))
    assert list(serie) == [3.0, 4.0]


def test_periodos_siguientes():
    assert periodos_siguientes(pd.Timestamp(2026, 10, 1), "MENSUAL", 3) == [date(2026, 11, 1), date(2026, 12, 1), date(2027, 1, 1)]
    assert periodos_siguientes(pd.Timestamp(2026, 11, 2), "SEMANAL", 2) == [date(2026, 11, 9), date(2026, 11, 16)]
