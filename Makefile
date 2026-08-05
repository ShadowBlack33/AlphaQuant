.PHONY: venv install etl tests metrics tune notebook docker-build docker-test

venv:
	python -m venv .venv

install: venv
	. .venv/bin/activate && pip install -e ".[dev]"

install-tuning: venv
	. .venv/bin/activate && pip install -e . && pip install -r requirements-tuning.txt

etl:
	. .venv/bin/activate && python -m alphaquant.cli.menu

tests:
	. .venv/bin/activate && pytest -q

metrics:
	. .venv/bin/activate && python -c "from alphaquant.training.train_regression import run_for_folder; run_for_folder('data/raw')"

tune:
	. .venv/bin/activate && python -c "from alphaquant.training.tune import tune_file; import glob; [tune_file(p, 'svr', 30) for p in glob.glob('data/raw/*_1d.csv')]"

notebook:
	@echo "Abre notebooks/metrics_report.ipynb en tu entorno preferido."

docker-build:
	docker build -t alphaquant:latest .

docker-test:
	docker run --rm alphaquant:latest