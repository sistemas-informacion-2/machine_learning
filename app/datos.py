"""Lecturas del Data Warehouse que alimentan los modelos. Todas filtradas por empresa."""
import pandas as pd

from app.db import leer


def ventas_diarias(id_empresa: int) -> pd.DataFrame:
    return leer(
        """
        SELECT f.fecha, v.id_sucursal, COALESCE(p.categoria, 'Sin categoria') AS categoria,
               SUM(v.importe_neto)::float8 AS importe, SUM(v.cantidad)::int AS unidades
          FROM dw.fact_ventas v
          JOIN dw.dim_fecha f ON f.fecha_key = v.fecha_key
          LEFT JOIN dw.dim_producto p ON p.id_variante = v.id_variante
         WHERE v.id_empresa = :id_empresa
         GROUP BY 1, 2, 3
         ORDER BY 1
        """,
        id_empresa,
    )


def tickets_presenciales(id_empresa: int) -> pd.DataFrame:
    """Tickets por dia y sucursal, sin las ventas en linea."""
    return leer(
        """
        SELECT f.fecha, v.id_sucursal, COUNT(DISTINCT v.id_nota)::int AS tickets
          FROM dw.fact_ventas v
          JOIN dw.dim_fecha f ON f.fecha_key = v.fecha_key
         WHERE v.id_empresa = :id_empresa AND v.canal <> 'ONLINE'
         GROUP BY 1, 2
        """,
        id_empresa,
    )


def compras_de_clientes(id_empresa: int) -> pd.DataFrame:
    """Una fila por nota de cada cliente real (sin el consumidor final)."""
    return leer(
        """
        SELECT v.id_cliente, v.id_nota, f.fecha, SUM(v.importe_neto)::float8 AS importe,
               BOOL_OR(v.canal = 'ONLINE') AS online,
               SUM(v.descuento_producto + v.descuento_nota)::float8 AS descuento,
               SUM(v.importe_bruto + v.descuento_producto)::float8 AS importe_lista
          FROM dw.fact_ventas v
          JOIN dw.dim_fecha f ON f.fecha_key = v.fecha_key
          JOIN dw.dim_cliente c ON c.id_cliente = v.id_cliente
         WHERE v.id_empresa = :id_empresa AND NOT c.es_consumidor_final
         GROUP BY 1, 2, 3
        """,
        id_empresa,
    )


def devoluciones_de_clientes(id_empresa: int) -> pd.DataFrame:
    return leer(
        """
        SELECT d.id_cliente, f.fecha
          FROM dw.fact_devoluciones d
          JOIN dw.dim_fecha f ON f.fecha_key = d.fecha_key
         WHERE d.id_empresa = :id_empresa
        """,
        id_empresa,
    )


def lineas_de_notas(id_empresa: int) -> pd.DataFrame:
    return leer(
        """
        SELECT DISTINCT v.id_nota, p.id_producto, p.producto
          FROM dw.fact_ventas v
          JOIN dw.dim_producto p ON p.id_variante = v.id_variante
         WHERE v.id_empresa = :id_empresa
        """,
        id_empresa,
    )


def stock_actual(id_empresa: int) -> pd.DataFrame:
    """Stock disponible de hoy por sucursal y variante, de almacenes y variantes activos."""
    return leer(
        """
        SELECT a.id_sucursal, u.id_variante, SUM(u.stock_disponible)::int AS stock
          FROM (SELECT DISTINCT ON (id_almacen, id_variante) id_almacen, id_variante, stock_disponible
                  FROM dw.fact_inventario_diario
                 WHERE id_empresa = :id_empresa
                 ORDER BY id_almacen, id_variante, fecha_key DESC) u
          JOIN dw.dim_almacen a ON a.id_almacen = u.id_almacen AND a.activo
          JOIN dw.dim_producto p ON p.id_variante = u.id_variante AND p.activo
         GROUP BY 1, 2
        """,
        id_empresa,
    )


def ventas_recientes_por_variante(id_empresa: int, desde: str) -> pd.DataFrame:
    return leer(
        """
        SELECT v.id_sucursal, v.id_variante, COALESCE(p.categoria, 'Sin categoria') AS categoria, SUM(v.cantidad)::int AS unidades
          FROM dw.fact_ventas v
          JOIN dw.dim_fecha f ON f.fecha_key = v.fecha_key
          JOIN dw.dim_producto p ON p.id_variante = v.id_variante AND p.activo
         WHERE v.id_empresa = :id_empresa AND f.fecha >= CAST(:desde AS date)
         GROUP BY 1, 2, 3
        """,
        id_empresa,
        desde=desde,
    )


def categorias_de_variantes(id_empresa: int) -> pd.DataFrame:
    return leer(
        """
        SELECT id_variante, COALESCE(categoria, 'Sin categoria') AS categoria
          FROM dw.dim_producto
         WHERE id_empresa = :id_empresa AND activo
        """,
        id_empresa,
    )
