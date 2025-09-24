SHELL := /bin/bash

# -------- Config --------
PY        ?= python
PORT      ?= 3000
BENTONAME ?= w11d1_bento_hello
override BENTONAME := $(strip $(BENTONAME))
VM_IP     ?= 54.234.80.17

# AWS (for ECR)
AWS_ACCOUNT_ID ?=
AWS_REGION     ?=
ECR_REPO       ?= $(BENTONAME)

# -------- VENV --------
.PHONY: setup
setup:
	@test -d .venv || $(PY) -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip
	. .venv/bin/activate; pip install -r requirements.txt || true

# -------- TRAIN & TEST --------
.PHONY: train test
train:
	. .venv/bin/activate; $(PY) train.py

test:
	. .venv/bin/activate; pytest -q

# -------- SERVE (no container) --------
.PHONY: serve
serve:
	. .venv/bin/activate; bentoml serve service.py:svc --port $(PORT)

# -------- Bento build/list (optional, still fine) --------
.PHONY: bento.build bento.list
bento.build:
	. .venv/bin/activate; bentoml build

bento.list:
	. .venv/bin/activate; bentoml list

# -------- Docker (manual, robust) --------
.PHONY: docker.build docker.run
docker.build:
	@echo "Building Docker image $(BENTONAME):manual using Dockerfile.bento"
	docker build -f Dockerfile.bento -t $(BENTONAME):manual .

docker.run:
	@echo "Running $(BENTONAME):manual on http://localhost:$(PORT)"
	docker run --rm -p $(PORT):3000 $(BENTONAME):manual

# -------- CURL helpers --------
.PHONY: curl.readyz curl.predict curl.predict.vm
curl.readyz:
	curl -s http://localhost:$(PORT)/readyz | cat

curl.predict:
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' | jq .

curl.predict.vm:
	curl -s -X POST http://$(VM_IP):$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' | jq .

# -------- ECR (login, ensure repo, tag, push) --------
.PHONY: ecr.login ecr.ensure_repo docker.tag docker.push
ecr.login:
	@if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "ERROR: Set AWS_ACCOUNT_ID and AWS_REGION env vars."; exit 1; fi
	aws ecr get-login-password --region $(AWS_REGION) \
	| docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

ecr.ensure_repo:
	@if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "ERROR: Set AWS_ACCOUNT_ID and AWS_REGION env vars."; exit 1; fi
	@aws ecr describe-repositories --repository-names $(ECR_REPO) --region $(AWS_REGION) >/dev/null 2>&1 \
	|| aws ecr create-repository --repository-name $(ECR_REPO) --region $(AWS_REGION) >/dev/null

# Tag/push the MANUAL image (not the containerize one)
docker.tag: docker.build
	@set -euo pipefail; \
	if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "ERROR: Set AWS_ACCOUNT_ID and AWS_REGION env vars."; exit 1; fi; \
	SRC_IMG="$(BENTONAME):manual"; \
	DEST_BASE="$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)"; \
	TAGVER=$$(date +%Y%m%d%H%M%S); \
	echo "Tagging $$SRC_IMG -> $$DEST_BASE:$$TAGVER and :latest"; \
	docker tag "$$SRC_IMG" "$$DEST_BASE:$$TAGVER"; \
	docker tag "$$SRC_IMG" "$$DEST_BASE:latest"; \
	printf "%s\n" "$$DEST_BASE:$$TAGVER" > .ecr_image_tag; \
	printf "%s\n" "$$DEST_BASE:latest"  > .ecr_image_latest

docker.push: ecr.ensure_repo ecr.login
	@set -euo pipefail; \
	if [ ! -f .ecr_image_tag ] || [ ! -f .ecr_image_latest ]; then \
		echo "ERROR: Run 'make docker.tag' first."; exit 1; fi; \
	IMG_VER=$$(cat .ecr_image_tag); \
	IMG_LATEST=$$(cat .ecr_image_latest); \
	echo "Pushing $$IMG_VER"; docker push "$$IMG_VER"; \
	echo "Pushing $$IMG_LATEST"; docker push "$$IMG_LATEST"

