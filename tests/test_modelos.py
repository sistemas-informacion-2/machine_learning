from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.modelos.abandono import predecir_abandono, variables_al_corte
from app.modelos.anomalias import detectar_anomalias
from app.modelos.canasta import reglas_de_asociacion
from app.modelos.reposicion import sugerir_compra
from app.modelos.segmentacion import segmentar

HOY = date(2026, 11, 8)


# ---------------------------------------------------------------- abandono


def compras_con_abandono(clientes: int = 300, semilla: int = 0) -> pd.DataFrame:
    """Cada cliente compra cada ~25 dias hasta que, a veces, se va para siempre."""
    rng = np.random.default_rng(semilla)
    filas = []
    nota = 0
    for cliente in range(1, clientes + 1):
        dia = HOY - timedelta(days=int(rng.integers(500, 730)))
        se_va = dia + timedelta(days=int(rng.integers(30, 700))) if rng.random() < 0.6 else HOY
        while dia < min(se_va, HOY):
            nota += 1
            filas.append({"id_cliente": cliente, "id_nota": nota, "fecha": dia, "importe": float(rng.integers(50, 400)), "online": bool(rng.random() < 0.2), "descuento": 0.0, "importe_lista": 100.0})
            dia += timedelta(days=int(rng.integers(10, 40)))
    return pd.DataFrame(filas)


SIN_DEVOLUCIONES = pd.DataFrame({"id_cliente": pd.Series(dtype=int), "fecha": pd.Series(dtype="datetime64[ns]")})


def test_abandono_aprende_y_se_evalua_con_el_ultimo_corte():
    resultado = predecir_abandono(compras_con_abandono(), SIN_DEVOLUCIONES, HOY)
    assert resultado is not None
    assert resultado.metricas["auc"] > 0.8
    assert resultado.metricas["modelo"] in {"RANDOM_FOREST", "REGRESION_LOGISTICA"}
    assert resultado.clientes["probabilidad"].between(0, 1).all()
    assert set(resultado.clientes["nivel"]) <= {"ALTO", "MEDIO", "BAJO"}
    assert all(len(f) == 2 for f in resultado.clientes["factores"])


def test_el_que_dejo_de_comprar_hace_mucho_tiene_mas_riesgo_que_el_activo():
    compras = compras_con_abandono()
    resultado = predecir_abandono(compras, SIN_DEVOLUCIONES, HOY)
    ultima = pd.to_datetime(compras["fecha"]).groupby(compras["id_cliente"]).max()
    recencia = (pd.Timestamp(HOY) - ultima).dt.days
    tabla = resultado.clientes.set_index("id_cliente")["probabilidad"]
    assert tabla[recencia[recencia > 200].index].mean() > tabla[recencia[recencia < 40].index].mean() + 0.3


def test_las_variables_no_ven_el_futuro():
    compras = pd.DataFrame(
        [
            {"id_cliente": 1, "id_nota": 1, "fecha": pd.Timestamp(2026, 1, 1), "importe": 10.0, "online": False},
            {"id_cliente": 1, "id_nota": 2, "fecha": pd.Timestamp(2026, 6, 1), "importe": 990.0, "online": True},
        ]
    )
    fila = variables_al_corte(compras, SIN_DEVOLUCIONES, date(2026, 3, 1)).iloc[0]
    assert fila["frecuencia"] == 1 and fila["monto"] == 10.0 and fila["proporcion_online"] == 0


def test_abandono_sin_datos_suficientes():
    assert predecir_abandono(compras_con_abandono(5), SIN_DEVOLUCIONES, HOY) is None
    assert predecir_abandono(pd.DataFrame(columns=["id_cliente", "id_nota", "fecha", "importe", "online"]), SIN_DEVOLUCIONES, HOY) is None


# ---------------------------------------------------------------- segmentacion


def compras_por_grupos() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    filas = []
    nota = 0
    grupos = [(40, (1, 25), 20, 400), (40, (300, 500), 2, 100), (40, (1, 30), 1, 60)]
    cliente = 0
    for cantidad, (reciente_min, reciente_max), compras, monto in grupos:
        for _ in range(cantidad):
            cliente += 1
            base = int(rng.integers(reciente_min, reciente_max))
            for j in range(compras):
                nota += 1
                filas.append({"id_cliente": cliente, "id_nota": nota, "fecha": HOY - timedelta(days=base + j * 5), "importe": float(monto), "online": False, "descuento": 0.0, "importe_lista": float(monto)})
    return pd.DataFrame(filas)


def test_segmentacion_encuentra_los_grupos_y_los_nombra():
    resultado = segmentar(compras_por_grupos(), HOY)
    assert resultado is not None
    assert resultado.metricas["silueta"] > 0.4
    por_cliente = resultado.clientes.set_index("id_cliente")["segmento"]
    assert por_cliente[1] == "Frecuentes de alto gasto"
    assert por_cliente[41] == "Inactivos"
    assert por_cliente[81] != por_cliente[1]
    assert len(resultado.clientes) == 120


def test_segmentacion_con_pocos_clientes():
    compras = compras_por_grupos()
    assert segmentar(compras[compras["id_cliente"] <= 10], HOY) is None


# ---------------------------------------------------------------- canasta


def lineas_con_par() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    filas = []
    for nota in range(1, 501):
        if rng.random() < 0.3:
            filas += [(nota, 1, "Jean"), (nota, 2, "Polera")]
        else:
            filas.append((nota, int(rng.integers(3, 8)), "Otro"))
    tabla = pd.DataFrame(filas, columns=["id_nota", "id_producto", "producto"])
    tabla.loc[tabla["producto"] == "Otro", "producto"] = "Prod " + tabla["id_producto"].astype(str)
    return tabla


def test_canasta_encuentra_el_par_que_se_compra_junto():
    resultado = reglas_de_asociacion(lineas_con_par())
    assert resultado is not None and len(resultado.reglas) == 2
    regla = resultado.reglas.iloc[0]
    assert {regla["producto_a"], regla["producto_b"]} == {"Jean", "Polera"}
    assert regla["lift"] > 2 and regla["confianza"] > 0.9 and regla["notas"] > 100


def test_canasta_sin_pares_devuelve_cero_reglas():
    lineas = pd.DataFrame([(i, (i % 5) + 1, f"P{(i % 5) + 1}") for i in range(100)], columns=["id_nota", "id_producto", "producto"])
    resultado = reglas_de_asociacion(lineas)
    assert resultado is not None and resultado.reglas.empty


def test_canasta_con_pocas_notas():
    assert reglas_de_asociacion(pd.DataFrame([(1, 1, "A")], columns=["id_nota", "id_producto", "producto"])) is None


# ---------------------------------------------------------------- anomalias


def tickets_diarios(dias: int, picos: dict[int, float] | None = None, base: float = 30) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    filas = []
    for i in range(dias):
        dia = HOY - timedelta(days=dias - i)
        media = (base + base * 0.6 * (dia.weekday() >= 4)) * (picos or {}).get(i, 1.0)
        filas.append({"fecha": dia, "id_sucursal": 1, "tickets": int(rng.poisson(media))})
    return pd.DataFrame(filas)


def test_detecta_un_dia_muy_por_encima_y_otro_muy_por_debajo():
    resultado = detectar_anomalias(tickets_diarios(200, {100: 3.0, 150: 0.1}), HOY)
    assert resultado is not None
    fechas = set(resultado.anomalias["fecha"])
    assert HOY - timedelta(days=100) in fechas and HOY - timedelta(days=50) in fechas
    assert (resultado.anomalias["gravedad"] == "ALTA").sum() >= 2
    fila = resultado.anomalias[resultado.anomalias["fecha"] == HOY - timedelta(days=100)].iloc[0]
    assert fila["valor_real"] > 2 * fila["valor_esperado"] and fila["desviacion"] > 0


def test_el_patron_semanal_normal_no_es_anomalia():
    resultado = detectar_anomalias(tickets_diarios(200), HOY)
    assert resultado is not None and len(resultado.anomalias) == 0


def test_temporada_alta_no_se_llena_de_falsas_alarmas():
    # La misma tienda, con el triple de actividad la mitad del tiempo: en conteos, mas nivel es mas variabilidad.
    temporada = {i: 3.0 for i in range(100, 200)}
    resultado = detectar_anomalias(tickets_diarios(300, temporada, base=8), HOY)
    assert resultado is not None and len(resultado.anomalias) <= 1


def test_anomalias_con_poca_historia():
    assert detectar_anomalias(tickets_diarios(20), HOY) is None


# ---------------------------------------------------------------- reposicion


def test_sugiere_comprar_segun_la_demanda_y_el_stock():
    filas = [{"fecha": date(2025, 1, 1) + timedelta(days=30 * i), "id_sucursal": 1, "categoria": "Jeans", "unidades": 100} for i in range(20)]
    diario = pd.DataFrame(filas)
    recientes = pd.DataFrame([(1, 10, "Jeans", 90), (1, 11, "Jeans", 10)], columns=["id_sucursal", "id_variante", "categoria", "unidades"])
    stock = pd.DataFrame([(1, 10, 5), (1, 11, 500)], columns=["id_sucursal", "id_variante", "stock"])
    categorias = pd.DataFrame([(10, "Jeans"), (11, "Jeans")], columns=["id_variante", "categoria"])

    resultado = sugerir_compra(diario, recientes, stock, categorias, date(2026, 9, 15))

    # La variante con 500 en stock no necesita compra; la que mas vende y tiene 5, si.
    assert list(resultado["id_variante"]) == [10]
    fila = resultado.iloc[0]
    assert fila["demanda_esperada"] > 150 and fila["cantidad_sugerida"] > 100 and fila["stock_actual"] == 5
    assert "unidades" in fila["motivo"]


def test_sin_historia_no_sugiere_nada():
    vacio = pd.DataFrame(columns=["fecha", "id_sucursal", "categoria", "unidades"])
    recientes = pd.DataFrame(columns=["id_sucursal", "id_variante", "categoria", "unidades"])
    stock = pd.DataFrame(columns=["id_sucursal", "id_variante", "stock"])
    assert sugerir_compra(vacio, recientes, stock, pd.DataFrame(columns=["id_variante", "categoria"]), HOY).empty
