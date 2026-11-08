# Servicio de machine learning del BI

Servicio interno en Python (FastAPI + pandas + scikit-learn + statsmodels + mlxtend). Después de cada carga del Data Warehouse, NestJS le avisa y entrena los modelos de esa empresa. Los resultados quedan en las tablas `dw.ml_*`, que leen el agente y los dashboards.

No se expone a internet: solo lo llama NestJS, con la clave `X-Clave-Interna`. Sin clave configurada rechaza todos los pedidos.

## Modelos

| Modelo | Tipo | Técnica | Tabla |
|---|---|---|---|
| Pronóstico de ventas | Supervisado | Holt, Holt-Winters o estacional con tendencia; gana el de menor error en los últimos períodos, y siempre compite contra el modelo ingenuo | `ml_pronostico` |
| Sugerencia de compra | Prescriptivo | Pronóstico por categoría y sucursal, repartido entre variantes según sus ventas recientes, menos stock actual, más stock de seguridad | `ml_sugerencia_compra` |
| Riesgo de abandono | Supervisado | Random Forest o regresión logística sobre recencia, frecuencia, monto, ticket, antigüedad, canal y devoluciones | `ml_riesgo_abandono` |
| Segmentación | No supervisado | K-means sobre RFM; *k* por silueta | `ml_segmento_cliente` |
| Canasta de mercado | No supervisado | Apriori | `ml_asociacion` |
| Anomalías | No supervisado | Tickets presenciales por día en escala raíz cuadrada, descomposición STL e Isolation Forest | `ml_anomalia` |

Cada modelo guarda sus métricas en `ml_ejecucion`. El abandono se evalúa con una fecha de corte pasada cuyo resultado ya se conoce, sin usar datos del futuro. Holt-Winters necesita al menos 27 meses completos (24 de entrenamiento y 3 de prueba).

## Puesta en marcha

Requiere Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Completa `.env`:

- `ML_DATABASE_URL`: conexión con el usuario `bi_ml`. Su creación está en `backend/docs/BI-DW.md`.
- `ML_CLAVE_INTERNA`: la misma clave que en `backend/.env`.

Para levantarlo:

```
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

| Ruta | Clave | Qué hace |
|---|---|---|
| `GET /salud` | No | Comprueba que el servicio responde |
| `POST /entrenar` `{idEmpresa}` | Sí | Encola el entrenamiento y responde 202; el resultado queda en `dw.ml_ejecucion` |
| `POST /pronostico` `{idEmpresa, ambito, clave, frecuencia, horizonte}` | Sí | Pronóstico al momento |

## Tests

```
pytest
```

Los modelos son funciones puras y se prueban con datos sintéticos, sin base de datos.
