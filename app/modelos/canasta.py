"""Canasta de mercado (no supervisado: reglas de asociacion con Apriori): que productos se compran juntos."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

MINIMO_NOTAS = 30
MINIMO_NOTAS_CON_EL_PAR = 5
LIFT_MINIMO = 1.2
CONFIANZA_MINIMA = 0.05


@dataclass
class ResultadoCanasta:
    reglas: pd.DataFrame
    metricas: dict[str, Any]


COLUMNAS = ["id_producto_a", "producto_a", "id_producto_b", "producto_b", "soporte", "confianza", "lift", "notas"]


def reglas_de_asociacion(lineas: pd.DataFrame) -> ResultadoCanasta | None:
    """lineas: id_nota, id_producto, producto. None si hay muy pocas notas."""
    notas = lineas["id_nota"].nunique() if not lineas.empty else 0
    if notas < MINIMO_NOTAS:
        return None

    nombres = lineas.drop_duplicates("id_producto").set_index("id_producto")["producto"]
    matriz = pd.crosstab(lineas["id_nota"], lineas["id_producto"]).astype(bool)
    frecuentes = apriori(matriz, min_support=MINIMO_NOTAS_CON_EL_PAR / notas, use_colnames=True, max_len=2)
    multiples = int((matriz.sum(axis=1) >= 2).sum())
    metricas = {"notas": notas, "notas_con_varios_productos": multiples}
    if frecuentes.empty or (frecuentes["itemsets"].map(len) == 2).sum() == 0:
        return ResultadoCanasta(reglas=pd.DataFrame(columns=COLUMNAS), metricas={**metricas, "reglas": 0})

    reglas = association_rules(frecuentes, num_itemsets=len(matriz), metric="lift", min_threshold=LIFT_MINIMO)
    reglas = reglas[reglas["confidence"] >= CONFIANZA_MINIMA].sort_values("lift", ascending=False)

    filas = []
    for regla in reglas.itertuples():
        a, b = next(iter(regla.antecedents)), next(iter(regla.consequents))
        filas.append(
            {
                "id_producto_a": int(a),
                "producto_a": str(nombres[a]),
                "id_producto_b": int(b),
                "producto_b": str(nombres[b]),
                "soporte": round(float(regla.support), 4),
                "confianza": round(float(regla.confidence), 4),
                "lift": round(float(regla.lift), 4),
                "notas": int(round(regla.support * notas)),
            }
        )
    return ResultadoCanasta(reglas=pd.DataFrame(filas, columns=COLUMNAS), metricas={**metricas, "reglas": len(filas)})
