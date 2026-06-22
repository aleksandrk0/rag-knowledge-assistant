.PHONY: install demo api test lint eval security

install:
	pip install -r requirements.txt

demo:
	python demo.py

api:
	uvicorn api.main:app --reload

test:
	pytest

lint:
	ruff check .

eval:
	python evaluation/run_eval.py

security:
	python evaluation/run_security_eval.py
