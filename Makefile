.PHONY: install preprocess train evaluate inference serve test docker-build docker-up clean

install:
	pip install -r requirements.txt

preprocess:
	python scripts/run_preprocessing.py

train:
	python scripts/run_training.py

evaluate:
	python scripts/run_evaluation.py

inference:
	python scripts/run_inference.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

clean:
	rm -rf artifacts/processed/* artifacts/features/* artifacts/models/* artifacts/reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
