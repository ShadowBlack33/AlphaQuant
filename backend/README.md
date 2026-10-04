# AlphaQuant backend

Python package `alphaquant`: data pipeline, risk models, evaluation and the
read-only FastAPI service consumed by `../frontend`.

```
src/alphaquant/
├── core/        paths (repo-root resolution), config, logging, seeding
├── data/        ETL (extract, transform, load) -- being rebuilt in v3
├── features/    technical indicators
├── risk/        GARCH volatility, regime detection, portfolio VaR/CVaR, pipeline
├── evaluation/  signal backtest (next: purged walk-forward, VaR backtest)
├── ml/          modeling layer (next: baselines and multi-asset backbone)
└── api/         FastAPI app
tests/           pytest suite
```

All paths resolve from the repository root via `alphaquant.core.paths`,
so commands work from any working directory. Install and run instructions
are in the root README.
