# AlphaQuant

Plataforma de analítica de riesgo financiero: volatilidad GARCH, detección
de regímenes de mercado y riesgo de portafolio (VaR, CVaR, drawdown),
servida por una API FastAPI y un dashboard en React.

> **Estado: v3 en reconstrucción.** Una auditoría encontró errores de datos
> y de evaluación en v0.2 (ver [docs/architecture.md](docs/architecture.md)).
> La versión anterior queda en el tag `v0.2-legacy`.

## Estructura

| Carpeta | Contenido |
|---|---|
| `backend/` | Paquete Python `alphaquant`: datos, riesgo, evaluación, API |
| `frontend/` | Dashboard React + Vite |
| `notebooks/` | Análisis paso a paso (auditoría de datos, modelos) |
| `config/` | `config.yaml`, universo de activos y parámetros |
| `data/` | Datos locales (no versionados) |
| `artifacts/` | Salidas generadas (no versionadas) |
| `docs/` | Arquitectura y decisiones |

## Instalación (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".\backend[dev,api,regime]"
python -m pytest backend\tests
```

## Ejecución

Requiere `TIINGO_API_KEY` en un archivo `.env` en la raíz (ver `.env.example`).

```powershell
python -m alphaquant.data.pipeline                                 # ETL -> data\processed + informe
python -m alphaquant.risk.pipeline                                 # riesgo -> artifacts\risk
python -m uvicorn alphaquant.api.main:app --reload --port 8000     # API
cd frontend; npm install; npm run dev                              # dashboard
```

## Uso de IA

Parte del código, las revisiones y la documentación se desarrollaron con
asistencia de IA (Claude). Las decisiones de diseño, la validación de los
resultados y la responsabilidad sobre el código son del autor.
