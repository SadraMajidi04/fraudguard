.PHONY: install test lint data train serve drift

install:
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest

lint:
	ruff check src tests

data:
	python -m fraudguard.cli generate --n 100000 --fraud-rate 0.01 --out data/transactions.csv.gz

train: data
	python -m fraudguard.cli train --data data/transactions.csv.gz --out-dir models

serve:
	python -m fraudguard.cli serve --model models/model.joblib --warm-data data/transactions.csv.gz

drift:
	python -m fraudguard.cli drift --model models/model.joblib --data data/transactions.csv.gz
