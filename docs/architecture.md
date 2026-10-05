# Arquitectura de AlphaQuant (v3, en construcción)

## Estructura del repositorio

```
alphaquant/
├── backend/            Paquete Python `alphaquant` (pyproject propio)
│   ├── src/alphaquant/
│   │   ├── core/       Rutas, configuración, logging, semillas
│   │   ├── data/       ETL: proveedores, capa raw, validación, procesado, informe
│   │   ├── features/   Indicadores técnicos
│   │   ├── risk/       GARCH, regímenes, VaR/CVaR de portafolio, pipeline
│   │   ├── evaluation/ Backtests y validación
│   │   ├── ml/         Baselines y backbone multiactivo (pendiente)
│   │   └── api/        FastAPI de solo lectura
│   └── tests/
├── frontend/           React + Vite
├── notebooks/          Análisis revisables paso a paso (01_data_audit, ...)
├── config/config.yaml  Única fuente de verdad del universo de activos
├── data/               raw/ (inmutable) y processed/ (locales, no versionados)
├── artifacts/          Salidas generadas: riesgo, modelos, logs (no versionado)
├── docs/               Arquitectura y decisiones
└── .github/workflows/  CI: lint + tests del backend, build del frontend
```

## Decisiones

**Rutas resueltas desde la raíz del repo, no desde el directorio actual.**
En v0.2 todas las rutas eran relativas (`data/raw`, `models/risk`). Ejecutar
un script desde la carpeta equivocada creaba copias de `data/`, `models/` y
`logs/` fuera del repo. `alphaquant.core.paths` localiza la raíz buscando
`config/config.yaml` (o usa `ALPHAQUANT_ROOT`) y falla si no la encuentra.

**`artifacts/` en lugar de `models/` para las salidas.**
Evita confundir código de modelos con archivos generados, y deja una sola
carpeta ignorada por git para todo lo regenerable.

**Configuración estricta.**
Si falta `config/config.yaml`, el sistema falla. Antes caía en silencio a
una configuración por defecto de 4 tickers. Un test exige que los tickers
configurados coincidan con los archivos en `data/raw/`.

**Se eliminó el clasificador direccional y la regresión de retornos (v0.2).**
La auditoría mostró 51 % de acierto en 112 predicciones (±9 pp) y que la
regresión perdía contra "predecir cero" en los 18 tickers. El modelado se
reconstruye con baselines explícitos y validación purgada. El estado
anterior queda en el tag `v0.2-legacy`.

**Correlación promedio sobre calendario común.**
Mezclar cripto (7 días) con acciones (5 días) dejaba NaN los fines de
semana y solo sobrevivía el par BTC–ETH: v0.2 reportaba 0.876. Con
calendario común el valor es ~0.19. Los tickers con menos de 252
observaciones se excluyen del nivel de portafolio en vez de recortar la
historia de todos.

## Capa de datos (ETL)

```
proveedor (Tiingo / Coinbase)
   │  fetch: historia completa en cada corrida
   ▼
data/raw/<proveedor>/<TICKER>/<fecha>.json.gz + .meta.json   ← inmutable, con sha256
   │  normalize: esquema canónico común
   ▼
validación
   1. contrato (Pandera): tipos, nulos, precios > 0, fechas únicas y ordenadas → FAIL
   2. calendario: fechas faltantes vs NYSE (o todos los días en cripto)  → WARN / FAIL
   3. flags por fila: OHLC, volumen 0, precio congelado, outlier, salto sin
      evento corporativo, ajuste sin evento                              → WARN
   │
   ▼
data/processed/<TICKER>_1d.csv   ret = log retorno total (adj_close), flags incluidos
artifacts/data_quality/latest.md informe por corrida (PASS / WARN / FAIL)
```

**Fuentes.** Tiingo para acciones y ETFs: precios crudos y ajustados,
dividendos y splits. Coinbase para BTC y ETH: velas diarias públicas en USD
real. El plan gratuito de Tiingo es de uso interno, así que el dashboard
público muestra métricas derivadas y no precios.

**Marcar, no recortar.** v0.2 recortaba retornos y precios a los cuantiles
0.1 %/99.9 % de toda la muestra: borraba los días extremos (SPY -11.6 % se
guardaba como -6.2 %, BTC -46 % como -18 %) y filtraba información del
futuro. Ahora los días sospechosos se marcan y se conservan. El outlier se
mide con mediana y MAD de los 252 días ANTERIORES, sin mirar el futuro.

**Retorno total.** `ret` sale de `adj_close`, que incluye dividendos. v0.2
usaba el cierre sin ajustar y perdía ~5 pp/año en HYG.

**Historia completa en cada corrida.** Los precios ajustados se recalculan
hacia atrás con cada dividendo nuevo; agregar solo los días nuevos mezclaría
bases de ajuste distintas. Son 16 peticiones a Tiingo y ~30 a Coinbase.

**Un ticker que falla no se publica** y conserva su último archivo bueno.

**Cripto y acciones en el mismo portafolio.** Los retornos de cripto se
llevan al calendario de NYSE sumando los log-retornos del fin de semana en
el lunes, en vez de descartarlos. Limitación: el cierre diario de cripto es
00:00 UTC, unas 4 horas después del cierre de NYSE. La volatilidad anual de
cripto se anualiza con 365 días, no con 252.

## Pendiente (orden de trabajo)

1. Datos macro de FRED y contraste de días marcados con una segunda fuente.
2. Baselines y evaluación walk-forward purgada con embargo.
3. Backbone multiactivo y fine-tuning.
4. Backtest de VaR (Kupiec, Christoffersen).
5. Asistente RAG.
6. Autenticación y frontend (bienvenida, login, dashboard).
