# RankAgent — reproducible entry points.
PY ?= .venv/bin/python
DATA_DIR ?= data/KuaiRand-Pure/data

.PHONY: help venv data baseline test sanity agent submission clean

help:
	@echo "make venv        create .venv and install dependencies"
	@echo "make data        download and extract KuaiRand-Pure"
	@echo "make sanity      harness self-check (random scoring must hit primary ~0.4753 on test)"
	@echo "make baseline    reproduce the official FM baseline (expect valid primary 0.6016)"
	@echo "make test        run the harness / leak / convergence tests"
	@echo "make agent       run the autonomous loop end to end"
	@echo "make submission  export the validation-best checkpoint and validate it"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	mkdir -p data
	cd data && curl -L -O https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
		&& tar xzf KuaiRand-Pure.tar.gz && rm KuaiRand-Pure.tar.gz
	@ls $(DATA_DIR)

sanity:
	cd kuairand-starter-kit && ../$(PY) baseline.py --model random --data_dir ../$(DATA_DIR)

baseline:
	$(PY) -m pipeline.train --model fm --data_dir $(DATA_DIR)

test:
	$(PY) -m pytest tests/ -q

agent:
	$(PY) main.py --data_dir $(DATA_DIR)

submission:
	$(PY) -m pipeline.submit --generate --checkpoint $(CKPT) --data_dir $(DATA_DIR)

clean:
	rm -rf checkpoints/*.pt checkpoints/*.npz checkpoints/*.npy checkpoints/*.json __pycache__
