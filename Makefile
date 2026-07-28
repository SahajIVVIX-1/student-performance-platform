.PHONY: setup lint test train serve-api serve-dashboard docker-build docker-up

PYTHON := P:/LabX-Global/Scripts/python.exe
PIP    := P:/LabX-Global/Scripts/pip.exe
PYTEST := P:/LabX-Global/Scripts/pytest.exe
RUFF   := P:/LabX-Global/Scripts/ruff.exe

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Setup complete."

lint:
	$(RUFF) check src/ tests/

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

train:
	$(PYTHON) -m student_perf.data.ingest
	$(PYTHON) -m student_perf.features.pipeline
	$(PYTHON) -m student_perf.models.train_all

serve-api:
	$(PYTHON) -m uvicorn student_perf.api.main:app --reload --host 0.0.0.0 --port 8000

serve-dashboard:
	$(PYTHON) -m streamlit run src/student_perf/dashboard/app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up --build
