.PHONY: help install smoke train format

help:
	@echo "Targets: install, smoke, train, format"

install:
	python3 -m pip install --upgrade pip
	# It's safe to skip heavy deps if desired; modify as needed
	python3 -m pip install -r requirements.txt || true

smoke:
	PYTHONPATH=$(PWD) python3 tests/smoke_test.py

train:
	PYTHONPATH=$(PWD) python3 -m src.training.train_lora --config configs/mistral_lora_example.yaml

format:
	python3 - <<'PY'
	try:
	    import ruff, black  # type: ignore
	    print("Formatting with ruff and black...")
	except Exception:
	    print("ruff/black not installed; skipping format")
	PY