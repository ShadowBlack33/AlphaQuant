# Arquitectura de AlphaQuant (v3, en construcción)

## Estructura del repositorio

```
alphaquant/
├── backend/            Paquete Python `alphaquant` (pyproject propio)
│   ├── src/alphaquant/
│   │   ├── core/       Rutas, configuración, logging, semillas
│   │   ├── data/       ETL (se reconstruye en el paso 1)
│   │   ├── features/   Indicadores técnicos
│   │   ├── risk/       GARCH, regímenes, VaR/CVaR de portafolio, pipeline
│   │   ├── evaluation/ Backtests y validación
│   │   ├── ml/         Baselines y backbone multiactivo (pendiente)
│   │   └── api/        FastAPI de solo lectura
│   └── tests/
├── frontend/           React + Vite
├── config/config.yaml  Única fuente de verdad del universo de activos
├── data/               raw/ y processed/ (locales, no versionados)
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

## Pendiente (orden de trabajo)

1. ETL: proveedores (Tiingo, Coinbase, FRED), capa raw inmutable, validación
   con Pandera, marcar en vez de recortar, retornos totales.
2. Baselines y evaluación walk-forward purgada con embargo.
3. Backbone multiactivo y fine-tuning.
4. Backtest de VaR (Kupiec, Christoffersen).
5. Asistente RAG.
6. Autenticación y frontend (bienvenida, login, dashboard).
