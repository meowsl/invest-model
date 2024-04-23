
run:
	@poetry run python main.py

install:
	python -m venv .venv
	poetry env use 3.11.0
	poetry install --no-root